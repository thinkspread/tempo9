// Copyright (c) 2026 Jiejing Zhang.
//
// Render a model's chat template — the one carried inside the .gguf.
//
// This is the last per-request piece that kept a native host tied to Python.
// It is not written here: chat templates are real Jinja, and the Qwen3.5 one
// alone uses macros, namespace(), loop variables, four tests, tojson/trim and
// raise_exception across 154 lines. huggingface/swift-jinja exists for
// exactly this job, so the work is wiring and verification, not
// implementation.
//
// Verified against transformers' own apply_chat_template by
// tools/check_gguf_chat_template_parity.py, at exact string equality. That
// bar matters: a template that renders *almost* right still produces a
// plausible prompt, and the model just answers slightly worse.

import Foundation
import GGUFKit
import Jinja

public enum ChatTemplateError: LocalizedError {
    case noTemplate
    case render(String)

    public var errorDescription: String? {
        switch self {
        case .noTemplate:
            return "this model carries no chat template "
                + "(tokenizer.chat_template)"
        case .render(let detail):
            return "chat template failed to render: \(detail)"
        }
    }
}

public struct ChatTemplate {
    public let source: String
    private let template: Template
    /// `bos_token` and friends, as the TEXT a template interpolates.
    ///
    /// Templates read these as plain variables -- Gemma 4's opens with
    /// `{{- bos_token -}}` -- and an undefined variable renders as nothing
    /// rather than failing. So a missing one is not a crash, it is a prompt
    /// that is one token short of what the model was trained on, and Gemma
    /// without its BOS degenerates: the same "Hello." that llama.cpp answers
    /// in fourteen tokens ran to nine hundred here, emitting `.` forever.
    ///
    /// Qwen never showed it because its template does not use them.
    private let specialTokens: [String: String]

    public init(source: String, specialTokens: [String: String] = [:]) throws {
        self.source = source
        self.specialTokens = specialTokens
        do {
            self.template = try Template(source)
        } catch {
            throw ChatTemplateError.render(String(describing: error))
        }
    }

    public init(gguf: GGUFFile) throws {
        guard let source = gguf.chatTemplate else {
            throw ChatTemplateError.noTemplate
        }
        // Resolved from the vocabulary rather than hard-coded, because the
        // ids are per-model and the SPELLING is what a template splices in.
        let tokens = gguf.kv["tokenizer.ggml.tokens"]?.stringsValue ?? []
        func text(_ key: String) -> String? {
            guard let id = gguf.kv[key]?.intValue,
                  id >= 0, id < tokens.count else { return nil }
            return tokens[id]
        }
        var specials: [String: String] = [:]
        for (name, key) in [("bos_token", "tokenizer.ggml.bos_token_id"),
                            ("eos_token", "tokenizer.ggml.eos_token_id"),
                            ("pad_token", "tokenizer.ggml.padding_token_id"),
                            ("unk_token", "tokenizer.ggml.unknown_token_id")] {
            if let t = text(key) { specials[name] = t }
        }
        try self.init(source: source, specialTokens: specials)
    }

    /// Render messages into the prompt string the model was trained on.
    ///
    /// `messages` is the OpenAI shape: `[["role": ..., "content": ...]]`,
    /// where content is a String or an array of parts. `extra` carries the
    /// per-model switches a template may read (`enable_thinking`, `tools`,
    /// ...), which are passed through untouched rather than enumerated --
    /// every family invents its own.
    public func render(messages: [[String: Any]],
                       addGenerationPrompt: Bool = true,
                       extra: [String: Any] = [:]) throws -> String {
        var context: [String: Any] = [
            "messages": messages,
            "add_generation_prompt": addGenerationPrompt,
        ]
        // Before `extra`, so a caller can still override one deliberately.
        for (key, value) in specialTokens { context[key] = value }
        for (key, value) in extra { context[key] = value }
        // Route through JSON so the ordered parser handles it: see
        // OrderedJSON for why Value(any:) is not usable here.
        //
        // .sortedKeys is CORRECTNESS, not cosmetics: this entry point takes
        // Swift dictionaries, whose serialization order is unstable per
        // call.  A `tools` array re-serialized by {{ tool | tojson }} lands
        // in the prompt HEAD, so two identical agent turns rendered
        // different prompts from char ~116 on — and the prefix cache could
        // never match past the tools block (codex: hit_requests=0/20 while
        // the wire JSON was byte-identical).  Canonical order fixes the
        // prompt; clients that need THEIR order preserved use the
        // render(contextJSON:) entry, which is untouched.
        let json = try JSONSerialization.data(withJSONObject: context,
                                              options: [.sortedKeys])
        return try render(contextJSON: String(decoding: json, as: UTF8.self))
    }

    /// Render from a JSON object, preserving its key order.
    ///
    /// This is the primary entry point, not a convenience: an OpenAI request
    /// *is* JSON, and a `tools` entry re-serialized by `{{ tool | tojson }}`
    /// goes straight into the prompt, so the client's key order has to
    /// survive the whole way through.
    public func render(contextJSON: String) throws -> String {
        let parsed = try OrderedJSON.parse(contextJSON)
        guard case .object(let parsedFields) = parsed else {
            throw ChatTemplateError.render("context must be a JSON object")
        }
        let fields = Self.decodeToolCallArguments(parsedFields)

        do {
            return try renderParsed(fields)
        } catch {
            // Some templates reject a system role outright -- Mistral v0.3
            // and the Llama-2 line raise "Conversation roles must alternate
            // user/assistant/...". Every agent front end sends a system
            // prompt, so without this those models are reachable by curl and
            // not by an agent. llama.cpp and Ollama make the same
            // accommodation: fold the system text into the first user turn.
            //
            // Only on failure, so a template that accepts system never takes
            // this path and no prompt changes for the models that worked.
            guard let folded = Self.foldSystemIntoFirstUser(fields) else {
                throw error
            }
            return try renderParsed(folded)
        }
    }

    /// OpenAI sends a past call's `function.arguments` as a JSON STRING; chat
    /// templates iterate it as a mapping (Qwen3.5: `tool_call.arguments|items`,
    /// emitting one `<parameter=...>` per key). transformers raises on a
    /// string there and vLLM decodes it first; swift-jinja yields nothing, so
    /// every historical call was rendered with its arguments silently gone --
    /// up to 1,825 of 4,590 tokens on a SiliconBench agent prompt, and the
    /// model no longer saw what it had run. Decode it, as vLLM does, when the
    /// string is a JSON object; anything else is left exactly as sent.
    /// The key order inside the string is kept (OrderedJSON), since a
    /// template that re-serializes it with `tojson` puts it in the prompt.
    static func decodeToolCallArguments(
        _ fields: OrderedDictionary<ObjectKey, Value>
    ) -> OrderedDictionary<ObjectKey, Value> {
        guard case .array(let msgs)? = fields[.string("messages")] else {
            return fields
        }
        var changed = false
        let decoded: [Value] = msgs.map { msg in
            guard case .object(var m) = msg,
                  case .array(let calls)? = m[.string("tool_calls")] else {
                return msg
            }
            m[.string("tool_calls")] = .array(calls.map { call in
                guard case .object(var c) = call,
                      case .object(var fn)? = c[.string("function")],
                      case .string(let raw)? = fn[.string("arguments")],
                      let obj = try? OrderedJSON.parse(raw),
                      case .object = obj else { return call }
                fn[.string("arguments")] = obj
                c[.string("function")] = .object(fn)
                changed = true
                return .object(c)
            })
            return .object(m)
        }
        guard changed else { return fields }
        var out = fields
        out[.string("messages")] = .array(decoded)
        return out
    }

    /// system + first user -> one user message. Returns nil when there is
    /// nothing to fold, or when the contents are not plain strings -- a
    /// multimodal part list has no defined concatenation and guessing one
    /// would corrupt the prompt rather than fail it.
    static func foldSystemIntoFirstUser(
        _ fields: OrderedDictionary<ObjectKey, Value>
    ) -> OrderedDictionary<ObjectKey, Value>? {
        guard case .array(let msgs)? = fields[.string("messages")],
              !msgs.isEmpty else { return nil }
        func role(_ v: Value) -> String? {
            guard case .object(let o) = v,
                  case .string(let r)? = o[.string("role")] else { return nil }
            return r
        }
        func text(_ v: Value) -> String? {
            guard case .object(let o) = v,
                  case .string(let c)? = o[.string("content")] else { return nil }
            return c
        }
        guard role(msgs[0]) == "system", let sys = text(msgs[0]) else {
            return nil
        }
        guard msgs.count > 1, role(msgs[1]) == "user",
              let user = text(msgs[1]),
              case .object(let firstUser) = msgs[1]
        else { return nil }
        var merged = firstUser
        merged[.string("content")] = .string(sys + "\n\n" + user)
        var out = Array(msgs.dropFirst(2))
        out.insert(.object(merged), at: 0)
        var newFields = fields
        newFields[.string("messages")] = .array(out)
        return newFields
    }

    private func renderParsed(
        _ fields: OrderedDictionary<ObjectKey, Value>
    ) throws -> String {
        do {
            let environment = Environment()
            // Environment filters win over built-ins. swift-jinja's tojson
            // sorts keys and drops the spaces after ',' and ':'; transformers
            // installs json.dumps(ensure_ascii=False, sort_keys=False). Both
            // are valid JSON and they are different prompts.
            environment["tojson"] = .function { args, kwargs, _ in
                guard let value = args.first else { return .string("null") }
                var indent: Int?
                if case .int(let n) = kwargs["indent"] ?? .null { indent = n }
                return .string(OrderedJSON.serialize(value, indent: indent))
            }

            var context = [String: Value]()
            for (key, value) in fields {
                if case .string(let name) = key { context[name] = value }
            }
            return try template.render(context, environment: environment)
        } catch let error as ChatTemplateError {
            throw error
        } catch {
            throw ChatTemplateError.render(String(describing: error))
        }
    }
}
