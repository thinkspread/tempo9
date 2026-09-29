// Copyright (c) 2026 Jiejing Zhang.
//
// JSON <-> Jinja Value, with the two properties chat templates depend on.
//
// Both exist because the rendered prompt has to be byte-identical to what
// transformers produces, and both were found by the parity gate rather than
// by reading anything:
//
//  1. KEY ORDER. `Value(any:)` sorts dictionary keys, and JSONSerialization
//     loses the document order before that anyway. But a template that emits
//     `{{ tool | tojson }}` puts the result straight into the prompt, and
//     Python dicts preserve insertion order, so transformers renders the
//     client's order. Sorted keys are a different prompt.
//
//  2. BOOLEANS. `Value(any:)` matches `as Int` before `as Bool`, and on
//     Darwin an NSNumber holding a boolean bridges to Int successfully. So
//     `true` arriving through JSONSerialization becomes `.int(1)`, and the
//     Qwen3.5 template's `{%- if enable_thinking is defined and
//     enable_thinking is true %}` quietly takes the wrong branch: thinking
//     stays off no matter what the caller asked for.
//
// Hence a small parser that builds Value directly, in order, with real bools.

import Foundation
import Jinja
import OrderedCollections

public enum OrderedJSON {

    // MARK: - parse

    public static func parse(_ text: String) throws -> Value {
        var scanner = Scanner(text: Array(text.unicodeScalars))
        scanner.skipWhitespace()
        let value = try scanner.parseValue()
        scanner.skipWhitespace()
        guard scanner.atEnd else {
            throw ChatTemplateError.render("trailing content in JSON")
        }
        return value
    }

    struct Scanner {
        let text: [Unicode.Scalar]
        var index = 0

        var atEnd: Bool { index >= text.count }
        var current: Unicode.Scalar? { index < text.count ? text[index] : nil }

        mutating func skipWhitespace() {
            while let c = current, c == " " || c == "\n" || c == "\t" || c == "\r" {
                index += 1
            }
        }

        mutating func expect(_ scalar: Unicode.Scalar) throws {
            guard current == scalar else {
                throw ChatTemplateError.render(
                    "expected \(scalar) at offset \(index)")
            }
            index += 1
        }

        mutating func parseValue() throws -> Value {
            skipWhitespace()
            guard let c = current else {
                throw ChatTemplateError.render("unexpected end of JSON")
            }
            switch c {
            case "{": return try parseObject()
            case "[": return try parseArray()
            case "\"": return .string(try parseString())
            case "t":
                try literal("true"); return .boolean(true)
            case "f":
                try literal("false"); return .boolean(false)
            case "n":
                try literal("null"); return .null
            default: return try parseNumber()
            }
        }

        mutating func literal(_ word: String) throws {
            for scalar in word.unicodeScalars {
                guard current == scalar else {
                    throw ChatTemplateError.render("bad literal near \(index)")
                }
                index += 1
            }
        }

        mutating func parseObject() throws -> Value {
            try expect("{")
            var dict = OrderedDictionary<ObjectKey, Value>()
            skipWhitespace()
            if current == "}" { index += 1; return .object(dict) }
            while true {
                skipWhitespace()
                let key = try parseString()
                skipWhitespace()
                try expect(":")
                dict[.string(key)] = try parseValue()   // insertion order kept
                skipWhitespace()
                if current == "," { index += 1; continue }
                try expect("}")
                return .object(dict)
            }
        }

        mutating func parseArray() throws -> Value {
            try expect("[")
            var items = [Value]()
            skipWhitespace()
            if current == "]" { index += 1; return .array(items) }
            while true {
                items.append(try parseValue())
                skipWhitespace()
                if current == "," { index += 1; continue }
                try expect("]")
                return .array(items)
            }
        }

        mutating func parseString() throws -> String {
            try expect("\"")
            var out = String.UnicodeScalarView()
            while let c = current {
                index += 1
                if c == "\"" { return String(out) }
                if c != "\\" { out.append(c); continue }
                guard let esc = current else { break }
                index += 1
                switch esc {
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                case "/": out.append("/")
                case "b": out.append(Unicode.Scalar(8))
                case "f": out.append(Unicode.Scalar(12))
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "u":
                    let code = try hex4()
                    if code >= 0xD800, code <= 0xDBFF,
                       current == "\\", index + 1 < text.count,
                       text[index + 1] == "u" {
                        index += 2
                        let low = try hex4()
                        let combined = 0x10000
                            + ((code - 0xD800) << 10) + (low - 0xDC00)
                        out.append(Unicode.Scalar(combined) ?? " ")
                    } else {
                        out.append(Unicode.Scalar(code) ?? " ")
                    }
                default:
                    throw ChatTemplateError.render("bad escape \\\(esc)")
                }
            }
            throw ChatTemplateError.render("unterminated JSON string")
        }

        mutating func hex4() throws -> Int {
            var value = 0
            for _ in 0..<4 {
                guard let c = current,
                      let digit = Character(c).hexDigitValue else {
                    throw ChatTemplateError.render("bad \\u escape")
                }
                value = value << 4 | digit
                index += 1
            }
            return value
        }

        mutating func parseNumber() throws -> Value {
            let start = index
            if current == "-" { index += 1 }
            var isDouble = false
            while let c = current {
                if c >= "0" && c <= "9" { index += 1; continue }
                if c == "." || c == "e" || c == "E" || c == "+" || c == "-" {
                    isDouble = isDouble || c == "." || c == "e" || c == "E"
                    index += 1
                    continue
                }
                break
            }
            let literal = String(String.UnicodeScalarView(text[start..<index]))
            if !isDouble, let int = Int(literal) { return .int(int) }
            guard let double = Double(literal) else {
                throw ChatTemplateError.render("bad number \(literal)")
            }
            return .double(double)
        }
    }

    // MARK: - serialize (transformers-compatible tojson)

    /// Matches `json.dumps(x, ensure_ascii=False, sort_keys=False)`, which is
    /// what transformers installs as the `tojson` filter: insertion order,
    /// `", "` and `": "` separators, non-ASCII left as-is.
    ///
    /// swift-jinja's built-in tojson sorts keys and omits the spaces, so a
    /// tools block rendered with it is a different prompt.
    public static func serialize(_ value: Value, indent: Int? = nil,
                                 depth: Int = 0) -> String {
        switch value {
        case .null: return "null"
        case .boolean(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d):
            // Python prints integral floats as "1.0"; Swift's default
            // description agrees, and non-integral values round-trip.
            return d == d.rounded() && d.magnitude < 1e16
                ? String(format: "%.1f", d) : String(d)
        case .string(let s): return quote(s)
        case .array(let items):
            if items.isEmpty { return "[]" }
            let parts = items.map { serialize($0, indent: indent, depth: depth + 1) }
            return wrap(parts, open: "[", close: "]", indent: indent, depth: depth)
        case .object(let dict):
            if dict.isEmpty { return "{}" }
            let parts = dict.map { key, item -> String in
                let name: String
                switch key {
                case .string(let s): name = s
                case .int(let i): name = String(i)
                }
                return quote(name) + ": "
                    + serialize(item, indent: indent, depth: depth + 1)
            }
            return wrap(parts, open: "{", close: "}", indent: indent, depth: depth)
        default:
            return "null"
        }
    }

    private static func wrap(_ parts: [String], open: String, close: String,
                             indent: Int?, depth: Int) -> String {
        guard let indent, indent > 0 else {
            return open + parts.joined(separator: ", ") + close
        }
        let pad = String(repeating: " ", count: indent * (depth + 1))
        let closePad = String(repeating: " ", count: indent * depth)
        return open + "\n" + parts.map { pad + $0 }.joined(separator: ",\n")
            + "\n" + closePad + close
    }

    private static func quote(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    // ensure_ascii=False: CJK and emoji stay literal.
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
