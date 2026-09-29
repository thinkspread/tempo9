// Copyright (c) 2026 Jiejing Zhang.
// The parser tolerances below are taken from vLLM's Qwen3 tool parser,
// Apache-2.0, Copyright the vLLM contributors. See NOTICE.
//
// The tool-call format these models actually emit.
//
// Not JSON. Every Qwen3.5/3.6/3.8 and AgentWorld .gguf in this project ships
// the same chat template, and it instructs the model to answer in an
// XML-ish form:
//
//     <tool_call>
//     <function=get_weather>
//     <parameter=city>
//     北京
//     </parameter>
//     </function>
//     </tool_call>
//
// This matters because an evaluation that asks for JSON instead is not
// measuring tool use — it is measuring whether the model will follow an
// ad-hoc instruction that contradicts its training. That is a different
// skill, and it penalises exactly the models tuned hardest on the real
// protocol.
//
// The tolerances below are taken from vLLM's Qwen3 parser
// (vllm/parser/qwen3.py), which drives this format in production and has a
// named transition for every way it comes out wrong. Each one here earned
// its place there:
//
//   * a bare <function=...> with NO <tool_call> around it. vLLM has an
//     explicit CONTENT -> TOOL_NAME fallback for this. It is not a rare
//     case for us: <tool_call> is a special token (id 248058), and the
//     first version of our decoder stripped it, so five correct calls
//     parsed as zero. A parser that requires the wrapper is one silently
//     dropped token away from reporting "no tool call" forever.
//   * a missing </parameter> before the next <parameter=. vLLM's regex
//     alternates the close tag with a lookahead at the next open tag.
//   * whitespace inside the tags: <parameter = city >.
//   * a stray </think> in the middle of content, which AgentWorld emits.
//     vLLM absorbs a duplicate silently rather than re-entering reasoning.
//   * back-to-back calls with no </tool_call> between them.
//
// Values are taken verbatim minus one leading and one trailing newline,
// which is the template's own layout rather than part of the value —
// vLLM's _trim_wrapping_newlines does exactly this.
//
// Not ported: streaming partial-argument events (vLLM emits ARG_VALUE_CHUNK
// deltas so a UI can render arguments as they arrive) and typing arguments
// against the schema. Both belong with a caller that dispatches tools, and
// there is not one yet.

import Foundation

public struct ToolCall: Sendable, Equatable {
    public var name: String
    /// Raw strings. Typing them against the schema is the caller's job: the
    /// schema is what says whether "42" is a number or a string, and this
    /// parser deliberately does not read schemas.
    public var arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }
}

public enum ToolCallParser {
    private static let funcRE = try! NSRegularExpression(
        pattern: "<\\s*function\\s*=\\s*([^>]*)>", options: [])
    private static let paramRE = try! NSRegularExpression(
        pattern: "<\\s*parameter\\s*=\\s*([^>]*)>(.*?)"
               + "(?:<\\s*/\\s*parameter\\s*>|(?=<\\s*parameter\\s*=)"
               + "|(?=<\\s*/\\s*function\\s*>)|(?=<\\s*/\\s*tool_call\\s*>)|\\z)",
        options: [.dotMatchesLineSeparators])

    /// Every call in `text`, in whichever dialect produced it.
    ///
    /// Dispatched on the ANCHOR, not on a model name the caller has to know.
    /// The two dialects share no syntax, so the text says which it is:
    ///
    ///   Qwen3.5   <function=name><parameter=key>value</parameter>
    ///   Gemma 4   <|tool_call>call:name{key:value}<tool_call|>
    ///
    /// This existed reading only the first, and Gemma emitted a perfectly
    /// correct call of the second kind while the parser reported none —
    /// which is the identical failure this file's own header records for
    /// Qwen, arriving through a family it had never been shown.
    public static func parse(_ text: String) -> [ToolCall] {
        let gemma = parseGemma(text)
        if !gemma.isEmpty { return gemma }
        return parseXML(text)
    }

    /// `<|tool_call>call:NAME{k:v,k2:v2}<tool_call|>`
    ///
    /// Values, per the template's own format_argument macro: a string is
    /// wrapped in `<|"|>`, null/true/false and numbers are bare, a mapping is
    /// `{k:v}` and a sequence `[a,b]`, with keys UNescaped at the top level.
    ///
    /// The wrapping is accepted but not required. The template writes it when
    /// it renders a call back into history; the model, generating, drops it —
    /// the first real call observed was `{city:Paris}`, not
    /// `{city:<|"|>Paris<|"|>}`. A parser that insisted on the documented
    /// form would read nothing at all.
    static func parseGemma(_ text: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var rest = Substring(text)
        while let open = rest.range(of: "<|tool_call>") {
            var cursor = rest[open.upperBound...]
            // "call:" is the only verb the template emits, and skipping it
            // when absent costs nothing.
            if cursor.hasPrefix("call:") { cursor = cursor.dropFirst(5) }
            guard let brace = cursor.firstIndex(of: "{") else {
                rest = cursor
                continue
            }
            let name = cursor[..<brace]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let (args, after) = scanObject(cursor[brace...])
            if !name.isEmpty {
                calls.append(ToolCall(name: name, arguments: args))
            }
            rest = after
        }
        return calls
    }

    /// Scan `{k:v,...}` from its opening brace. Returns the pairs and what
    /// follows the matching close.
    ///
    /// Hand-written rather than regex because a value can be a nested object,
    /// an array, or a `<|"|>`-wrapped string that may itself contain a comma
    /// or a brace — all three of which make a comma-splitting regex wrong in
    /// a way that only shows up on the argument that matters.
    private static func scanObject(_ s: Substring)
        -> (args: [String: String], rest: Substring) {
        var args: [String: String] = [:]
        var i = s.index(after: s.startIndex)     // past "{"
        while i < s.endIndex, s[i] != "}" {
            while i < s.endIndex, s[i] == "," || s[i].isWhitespace {
                i = s.index(after: i)
            }
            guard i < s.endIndex, s[i] != "}" else { break }
            guard let colon = s[i...].firstIndex(of: ":") else { break }
            let key = unquote(String(s[i..<colon])
                .trimmingCharacters(in: .whitespacesAndNewlines))
            let valueStart = s.index(after: colon)
            let end = scanValueEnd(s, from: valueStart)
            let raw = String(s[valueStart..<end])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !key.isEmpty { args[key] = unquote(raw) }
            i = end
        }
        let after = i < s.endIndex ? s.index(after: i) : s.endIndex
        return (args, s[after...])
    }

    /// Where the value starting at `from` ends: the first `,` or `}` that is
    /// not inside a nested `{}`/`[]` or a `<|"|>` string.
    private static func scanValueEnd(_ s: Substring,
                                     from: Substring.Index) -> Substring.Index {
        var depth = 0
        var inString = false
        var i = from
        while i < s.endIndex {
            if s[i..<s.endIndex].hasPrefix("<|\"|>") {
                inString.toggle()
                i = s.index(i, offsetBy: 5)
                continue
            }
            if !inString {
                let c = s[i]
                if c == "{" || c == "[" { depth += 1 }
                else if c == "}" || c == "]" {
                    if depth == 0 { return i }
                    depth -= 1
                } else if c == "," && depth == 0 { return i }
            }
            i = s.index(after: i)
        }
        return s.endIndex
    }

    private static func unquote(_ v: String) -> String {
        guard v.hasPrefix("<|\"|>"), v.hasSuffix("<|\"|>"), v.count >= 10
        else { return v }
        return String(v.dropFirst(5).dropLast(5))
    }

    private static func parseXML(_ text: String) -> [ToolCall] {
        let ns = text as NSString
        let whole = NSRange(location: 0, length: ns.length)
        let funcs = funcRE.matches(in: text, options: [], range: whole)
        guard !funcs.isEmpty else { return [] }

        var calls: [ToolCall] = []
        for (i, m) in funcs.enumerated() {
            let name = ns.substring(with: m.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { continue }

            let bodyStart = m.range.location + m.range.length
            var bodyEnd = i + 1 < funcs.count
                ? funcs[i + 1].range.location : ns.length
            // </function> ends the argument list before the next call does.
            let tail = NSRange(location: bodyStart, length: bodyEnd - bodyStart)
            if let close = ns.range(of: "</function>", options: [], range: tail)
                .location as Int?, close != NSNotFound {
                bodyEnd = close
            }

            let body = NSRange(location: bodyStart, length: bodyEnd - bodyStart)
            var args: [String: String] = [:]
            for p in paramRE.matches(in: text, options: [], range: body) {
                let key = ns.substring(with: p.range(at: 1))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !key.isEmpty else { continue }
                args[key] = trimWrappingNewlines(ns.substring(with: p.range(at: 2)))
            }
            calls.append(ToolCall(name: name, arguments: args))
        }
        return calls
    }

    /// One newline each side is the template's layout, not the value.
    private static func trimWrappingNewlines(_ s: String) -> String {
        var v = s
        if v.hasPrefix("\n") { v.removeFirst() }
        if v.hasSuffix("\n") { v.removeLast() }
        return v
    }
}
