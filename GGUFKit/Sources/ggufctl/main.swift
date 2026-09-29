// Copyright (c) 2026 Jiejing Zhang.
//
// ggufctl -- inspect a .gguf and exercise its tokenizer without Python.
//
//   ggufctl dump    --gguf f.gguf
//   ggufctl encode  --gguf f.gguf --text "hello"        (or --stdin)
//   ggufctl decode  --gguf f.gguf --ids 1,2,3
//   ggufctl template --gguf f.gguf                       (prints chat_template)
//   ggufctl render  --gguf f.gguf --json '{"messages": [...], ...}'
//   ggufctl render  --gguf f.gguf --batch                (one JSON per line)

import Foundation
import ChatTemplateKit
import GGUFKit

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("ggufctl: \(message)\n".utf8))
    exit(1)
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    fail("usage: ggufctl <dump|encode|decode|template> --gguf <file> [...]")
}
args.removeFirst()

var ggufPath: String?
var text: String?
var idList: String?
var readStdin = false
var batchLines = false
var jsonArg: String?

while let flag = args.first {
    args.removeFirst()
    func value() -> String {
        guard let v = args.first else { fail("\(flag) needs a value") }
        args.removeFirst()
        return v
    }
    switch flag {
    case "--gguf": ggufPath = value()
    case "--text": text = value()
    case "--ids": idList = value()
    case "--stdin": readStdin = true
    case "--batch": batchLines = true
    case "--json": jsonArg = value()
    default: fail("unknown flag \(flag)")
    }
}
guard let ggufPath else { fail("--gguf is required") }

do {
    // dump only needs the hyperparameters; the tokenizer commands need the
    // vocabulary, which is ~250k strings and worth skipping when unused.
    let needsVocab = command == "encode" || command == "decode"
    let started = Date()
    let file = try GGUFFile(path: ggufPath, readArrays: needsVocab)
    let loadSeconds = Date().timeIntervalSince(started)

    switch command {
    case "dump":
        var out: [String: Any] = [
            "architecture": file.architecture,
            "tensors": file.tensors.count,
            "kv": file.kv.count,
            "data_offset": Int(file.dataOffset),
            "alignment": Int(file.alignment),
            "has_chat_template": file.chatTemplate != nil,
            "load_seconds": loadSeconds,
        ]
        for suffix in ["block_count", "embedding_length", "feed_forward_length",
                       "attention.head_count", "attention.head_count_kv",
                       "context_length"] {
            if let v = file.arch(suffix)?.intValue { out[suffix] = v }
        }
        print(String(data: try JSONSerialization.data(
            withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)

    case "template":
        guard let template = file.chatTemplate else {
            fail("this file carries no tokenizer.chat_template")
        }
        print(template)

    case "encode":
        let tokenizer = try BPETokenizer(gguf: file)
        if batchLines {
            // One JSON array per input line, vocabulary loaded once. The
            // parity harness needs hundreds of strings and a process per
            // string would spend all its time re-reading 250k tokens.
            let raw = String(
                data: FileHandle.standardInput.readDataToEndOfFile(),
                encoding: .utf8) ?? ""
            var lines = raw.components(separatedBy: "\n")
            if lines.last == "" { lines.removeLast() }
            for line in lines {
                // Literal \n and \t so a case can carry newlines on one line.
                let decoded = line
                    .replacingOccurrences(of: "\\n", with: "\n")
                    .replacingOccurrences(of: "\\t", with: "\t")
                let ids = tokenizer.encode(decoded)
                print(String(data: try JSONSerialization.data(
                    withJSONObject: ids), encoding: .utf8)!)
            }
            break
        }
        var input = text
        if readStdin {
            input = String(data: FileHandle.standardInput.readDataToEndOfFile(),
                           encoding: .utf8)
        }
        guard let input else { fail("encode needs --text, --stdin or --batch") }
        let ids = tokenizer.encode(input)
        print(String(data: try JSONSerialization.data(withJSONObject: [
            "ids": ids,
            "count": ids.count,
            "vocab": tokenizer.vocabulary.count,
            "eos": tokenizer.eosTokenID as Any,
            "load_seconds": loadSeconds,
        ], options: [.sortedKeys]), encoding: .utf8)!)

    case "render":
        // Renders the model's own chat template. The context is passed
        // through as-is rather than through a fixed set of parameters:
        // every model family invents its own switches (enable_thinking,
        // tools, ...) and the template is the only thing that knows them.
        let chat = try ChatTemplate(gguf: file)

        // Straight from the JSON text: going through JSONSerialization would
        // lose the key order that {{ tool | tojson }} puts in the prompt.
        func renderOne(_ payload: String) throws -> String {
            try chat.render(contextJSON: payload)
        }

        if batchLines {
            let raw = String(
                data: FileHandle.standardInput.readDataToEndOfFile(),
                encoding: .utf8) ?? ""
            for line in raw.split(separator: "\n", omittingEmptySubsequences: true) {
                // One JSON string per line so a rendered prompt full of
                // newlines survives the line-based wire format.
                let rendered = try renderOne(String(line))
                let escaped = try JSONSerialization.data(
                    withJSONObject: [rendered])
                print(String(data: escaped, encoding: .utf8)!)
            }
            break
        }
        guard let jsonArg else { fail("render needs --json or --batch") }
        FileHandle.standardOutput.write(Data(try renderOne(jsonArg).utf8))

    case "decode":
        let tokenizer = try BPETokenizer(gguf: file)
        guard let idList else { fail("decode needs --ids") }
        let ids = idList.split(separator: ",").compactMap { Int($0) }
        // Raw, with no trailing newline: the text may legitimately end in one
        // and a round-trip check cannot tell ours from its own.
        FileHandle.standardOutput.write(
            Data(tokenizer.decode(ids, skipSpecialTokens: false).utf8))

    default:
        fail("unknown command \(command)")
    }
} catch {
    fail(error.localizedDescription)
}
