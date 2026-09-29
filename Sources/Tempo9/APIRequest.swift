// Copyright (c) 2026 Jiejing Zhang.
//
// Request parsing for the local API's three protocols, kept apart from the
// socket so it can be tested without an engine.
//
// Each parser turns a decoded JSON body into the one shape LocalSession
// takes -- template messages, template tools, a SamplingConfig -- or throws
// a RequestRefusal that the route answers with 400 in the caller's own
// error dialect.  Nothing here touches the engine.
//
// The rule these parsers enforce: a parameter the server cannot honour is
// REFUSED, naming the parameter and the value, never accepted and ignored.
// Every one of the checks below used to be a 200 whose reply quietly
// lacked what was asked for -- a tool_choice the model was free to
// disregard, a stop sequence that stopped nothing, an image that was not
// in the prompt.  An agent discovers that kind of downgrade in production,
// one wrong answer at a time; a 400 it discovers on the first call.

import Foundation
import Tempo9Engine

/// A request the server will not serve, and why.
///
/// `param` names the field in the caller's own vocabulary
/// (`tool_choice`, `messages[2].content[0].type`), `message` says what was
/// asked and what would be honoured instead.
struct RequestRefusal: Error, Equatable {
    let param: String
    let message: String

    /// OpenAI dialect: chat completions and /v1/responses.
    var openAI: [String: Any] {
        ["error": ["message": message,
                   "type": "invalid_request_error",
                   "param": param]]
    }

    /// Anthropic dialect: /v1/messages and count_tokens.
    var anthropic: [String: Any] {
        ["type": "error",
         "error": ["type": "invalid_request_error",
                   "message": message]]
    }
}

enum APIRequest {

    // MARK: - shared checks

    /// A JSON value as it appears in an error message.
    static func show(_ v: Any) -> String {
        if let s = v as? String { return "\"\(s)\"" }
        if let d = try? JSONSerialization.data(
               withJSONObject: v, options: [.fragmentsAllowed, .sortedKeys]),
           let s = String(data: d, encoding: .utf8) {
            return s
        }
        return "\(v)"
    }

    private static func present(_ v: Any?) -> Any? {
        guard let v, !(v is NSNull) else { return nil }
        return v
    }

    /// TEMPO9_STRIP_UNSUPPORTED_TOOLS=1: strip server-executed tools
    /// (web_search, web_fetch, code_execution, ...) from the request
    /// instead of refusing it, with a warning per stripped tool in the
    /// server log and in the response.  Default off -- an env var set on
    /// purpose is not a silent downgrade, an absent one must not become one.
    static let stripUnsupportedToolsEnvVar = "TEMPO9_STRIP_UNSUPPORTED_TOOLS"
    static var stripUnsupportedToolsEnv: Bool {
        ProcessInfo.processInfo.environment[stripUnsupportedToolsEnvVar] == "1"
    }

    /// tool_choice.  "auto" -- the model decides -- is the only mode this
    /// server has: tools go to the chat template and the reply is parsed
    /// for calls.  "none", "required" and a named tool are promises about
    /// the reply that nothing downstream enforces, so accepting them and
    /// running auto would hand the caller a guarantee it does not have.
    /// Same words on all three protocols; the object form is Anthropic's
    /// {type:"auto"|"any"|"tool"|"none"} and OpenAI's {type:"function",...}.
    private static func checkToolChoice(_ body: [String: Any]) throws {
        guard let choice = present(body["tool_choice"]) else { return }
        if let s = choice as? String, s == "auto" { return }
        if let o = choice as? [String: Any],
           (o["type"] as? String) == "auto" { return }
        throw RequestRefusal(
            param: "tool_choice",
            message: "tool_choice \(show(choice)) is not supported: this "
                   + "server only runs tools in \"auto\" mode (the model "
                   + "decides), and nothing here would enforce \"none\", "
                   + "\"required\"/\"any\" or a named tool. Omit tool_choice "
                   + "or send \"auto\".")
    }

    /// stop (OpenAI) / stop_sequences (Anthropic).  The engine stops on
    /// token ids, not strings, and the server has no host-side cut: a stop
    /// sequence would be accepted and decoded straight through to
    /// max_tokens, which is what happened.  Empty and null mean none.
    private static func checkStop(_ body: [String: Any], key: String) throws {
        guard let v = present(body[key]) else { return }
        let seqs: [Any]
        if let s = v as? String { seqs = s.isEmpty ? [] : [s] }
        else if let a = v as? [Any] { seqs = a }
        else {
            throw RequestRefusal(
                param: key,
                message: "\(key) must be a string or an array of strings, "
                       + "got \(show(v))")
        }
        guard !seqs.isEmpty else { return }
        throw RequestRefusal(
            param: key,
            message: "\(key) \(show(seqs)) is not supported: the engine "
                   + "stops on token ids only, so a stop string would be "
                   + "accepted and then generated straight through. Drop it "
                   + "and bound the reply with max_tokens.")
    }

    private static func sampling(_ body: [String: Any],
                                 speculationK: Int32,
                                 defaultMaxTokens: Int,
                                 maxTokensKeys: [String],
                                 topP: Bool) -> SamplingConfig {
        var cfg = SamplingConfig()
        cfg.speculationK = speculationK
        cfg.maxTokens = maxTokensKeys.lazy
            .compactMap { body[$0] as? Int }.first ?? defaultMaxTokens
        if let t = body["temperature"] as? Double {
            cfg.temperature = Float(t)
            cfg.doSample = t > 0
            if t == 0 { cfg.topK = 1 }
        }
        if topP, let p = body["top_p"] as? Double { cfg.topP = Float(p) }
        return cfg
    }

    /// Guided decoding from a {type, schema} format object -- chat's
    /// response_format (schema nested under json_schema) and Responses'
    /// text.format (schema at top level) both land here.
    ///
    /// json_schema without a schema is refused rather than downgraded to
    /// json_object: the caller asked for ITS schema to be enforced, and
    /// quietly serving free-form JSON with a 200 is the kind of
    /// weaker-than-advertised guarantee an agent only discovers in
    /// production.  The engine's C API refuses the same combination.
    private static func guided(_ cfg: inout SamplingConfig,
                               type: String, schema: Any?,
                               param: String, schemaParam: String) throws {
        switch type {
        case "text":
            break
        case "json_object":
            cfg.responseFormat = "json_object"
        case "json_schema":
            guard let schema = present(schema),
                  let data = try? JSONSerialization.data(withJSONObject: schema),
                  let text = String(data: data, encoding: .utf8) else {
                throw RequestRefusal(
                    param: schemaParam,
                    message: "\(param).json_schema requires a 'schema' object")
            }
            cfg.responseFormat = "json_schema"
            cfg.responseSchema = text
        default:
            throw RequestRefusal(
                param: "\(param).type",
                message: "\(param).type \(show(type)) is not supported; "
                       + "use \"text\", \"json_object\" or \"json_schema\"")
        }
    }

    /// The one system message a chat template accepts is the first one.
    ///
    /// Claude Code sends a second: after the user turn comes
    /// `{role:"system", content:"Available agent types for the Agent
    /// tool: ..."}`, and OpenAI clients send `developer` turns anywhere.
    /// Passed through, Qwen's template throws "System message must be at
    /// the beginning" and every turn is a 500.  Here every system/developer
    /// message, wherever it sits, is hoisted into ONE leading system
    /// message -- its texts in request order, blank-line separated -- and
    /// the other turns keep their order and their dictionaries (a user
    /// turn with an image part is the same object it was, for the image
    /// extraction that runs on these messages later).
    ///
    /// Array content on a system message is reduced to its text parts
    /// (`text`, or `input_text`); a non-text part there is refused by
    /// index, since dropping it would put a message in front of the
    /// model with a piece missing.
    static func foldSystemMessages(_ messages: [[String: Any]],
                                   at path: String) throws
        -> [[String: Any]] {
        var system: [String] = []
        var rest: [[String: Any]] = []
        for (mi, m) in messages.enumerated() {
            let role = m["role"] as? String ?? "user"
            guard role == "system" || role == "developer" else {
                rest.append(m)
                continue
            }
            if let text = m["content"] as? String {
                system.append(text)
            } else if let parts = m["content"] as? [[String: Any]] {
                var text = ""
                for (pi, part) in parts.enumerated() {
                    let type = part["type"] as? String ?? "?"
                    guard type == "text" || type == "input_text",
                          let t = part["text"] as? String else {
                        throw RequestRefusal(
                            param: "\(path)[\(mi)].content[\(pi)].type",
                            message: "\(path)[\(mi)] is a \(role) message and "
                                   + "its content[\(pi)] is a \(show(type)) "
                                   + "part; only text parts can go in the "
                                   + "system prompt")
                    }
                    text += t
                }
                system.append(text)
            }
        }
        let text = system.filter { !$0.isEmpty }.joined(separator: "\n\n")
        guard !text.isEmpty else { return rest }
        return [["role": "system", "content": text]] + rest
    }

    // MARK: - OpenAI chat completions

    struct Chat {
        var messages: [[String: Any]]
        var tools: [[String: Any]]
        var config: SamplingConfig
        var thinking: Bool
        var stream: Bool
        /// One line per tool stripped under TEMPO9_STRIP_UNSUPPORTED_TOOLS.
        var warnings: [String] = []
    }

    static func chat(_ body: [String: Any], speculationK: Int32,
                     stripUnsupportedTools: Bool = stripUnsupportedToolsEnv)
        throws -> Chat {
        guard let rawMessages = body["messages"] as? [[String: Any]] else {
            throw RequestRefusal(param: "messages",
                                 message: "messages[] required")
        }
        try checkToolChoice(body)
        try checkStop(body, key: "stop")
        var cfg = sampling(body, speculationK: speculationK,
                           defaultMaxTokens: 2048,
                           maxTokensKeys: ["max_tokens",
                                           "max_completion_tokens"],
                           topP: true)
        // OpenAI response_format -> engine guided decoding (xgrammar).
        //   {"type":"json_object"}                       any valid JSON
        //   {"type":"json_schema","json_schema":{"schema":{...}}}  constrained
        if let rf = body["response_format"] as? [String: Any],
           let type = rf["type"] as? String {
            let js = rf["json_schema"] as? [String: Any]
            try guided(&cfg, type: type,
                       schema: js?["schema"] ?? rf["schema"],
                       param: "response_format",
                       schemaParam: "response_format.json_schema.schema")
        }
        // OpenAI's tool shape IS the template's native shape; incoming
        // assistant tool_calls and role:"tool" messages pass through the
        // template untouched for the same reason.
        let tools = try ToolCallBridge.fromChat(
            body["tools"] as? [[String: Any]] ?? [],
            stripHosted: stripUnsupportedTools)
        let messages = try foldSystemMessages(rawMessages, at: "messages")
        return Chat(messages: messages, tools: tools.tools, config: cfg,
                    thinking: body["enable_thinking"] as? Bool ?? false,
                    stream: body["stream"] as? Bool ?? false,
                    warnings: tools.warnings)
    }

    // MARK: - Anthropic messages

    struct Anthropic {
        var messages: [[String: Any]]
        var tools: [[String: Any]]
        var config: SamplingConfig
        var thinking: Bool
        var stream: Bool
        var warnings: [String] = []
    }

    /// Anthropic content is a string or an array of typed blocks; the
    /// template layer wants plain text.  A non-text block (image,
    /// document) is refused by name: the old flatten returned nil for it,
    /// and every caller had a `?? ""` waiting, so an image in a tool_result
    /// reached the model as an empty result and a system prompt with an
    /// image in it vanished entirely.
    private static func flatten(_ content: Any?, at path: String) throws
        -> String? {
        guard let content = present(content) else { return nil }
        if let s = content as? String { return s }
        guard let blocks = content as? [[String: Any]] else {
            throw RequestRefusal(
                param: path,
                message: "\(path) must be a string or an array of content "
                       + "blocks, got \(show(content))")
        }
        var out = ""
        for (i, b) in blocks.enumerated() {
            let type = b["type"] as? String ?? "?"
            guard type == "text", let t = b["text"] as? String else {
                throw RequestRefusal(
                    param: "\(path)[\(i)].type",
                    message: "\(path)[\(i)] is a \(show(type)) block, which "
                           + "this server cannot put in front of the model; "
                           + "only \"text\" blocks are accepted here")
            }
            out += t
        }
        return out
    }

    static func anthropic(_ body: [String: Any], speculationK: Int32,
                          stripUnsupportedTools: Bool = stripUnsupportedToolsEnv)
        throws -> Anthropic {
        guard let rawMessages = body["messages"] as? [[String: Any]] else {
            throw RequestRefusal(param: "messages",
                                 message: "messages[] required")
        }
        try checkToolChoice(body)
        try checkStop(body, key: "stop_sequences")
        var messages: [[String: Any]] = []
        // The top-level system first, then every system-role message in
        // messages[] in order (Claude Code's agent-types preamble is one),
        // as the single leading system message -- see foldSystemMessages.
        var system: [String] = []
        if let top = try flatten(body["system"], at: "system"),
           !top.isEmpty {
            system.append(top)
        }
        for (mi, m) in rawMessages.enumerated() {
            let role = m["role"] as? String ?? "user"
            if role == "system" {
                if let text = try flatten(m["content"],
                                          at: "messages[\(mi)].content"),
                   !text.isEmpty {
                    system.append(text)
                }
                continue
            }
            if let text = m["content"] as? String {
                messages.append(["role": role, "content": text])
                continue
            }
            guard let blocks = m["content"] as? [[String: Any]] else {
                continue
            }
            // Typed blocks: text plus the tool round-trip.  tool_use on an
            // assistant turn becomes tool_calls; tool_result on a user turn
            // becomes a tool-role message.  Anything else (images,
            // documents) is refused by name rather than dropped.
            var text = ""
            var calls: [(id: String, name: String, argumentsJSON: String)] = []
            for (bi, b) in blocks.enumerated() {
                let path = "messages[\(mi)].content[\(bi)]"
                let type = b["type"] as? String ?? "?"
                switch type {
                case "text":
                    text += b["text"] as? String ?? ""
                case "tool_use":
                    let args = (try? JSONSerialization.data(
                        withJSONObject: b["input"] ?? [:])) ?? Data("{}".utf8)
                    calls.append((id: b["id"] as? String ?? "toolu_0",
                                  name: b["name"] as? String ?? "",
                                  argumentsJSON: String(data: args,
                                      encoding: .utf8) ?? "{}"))
                case "tool_result":
                    messages.append(ToolCallBridge.toolResultTurn(
                        callId: b["tool_use_id"] as? String ?? "toolu_0",
                        content: try flatten(b["content"],
                                             at: "\(path).content") ?? ""))
                default:
                    throw RequestRefusal(
                        param: "\(path).type",
                        message: "\(path) is a \(show(type)) block, which "
                               + "this server cannot put in front of the "
                               + "model; only text, tool_use and "
                               + "tool_result blocks are accepted")
                }
            }
            if role == "assistant" {
                if !text.isEmpty || !calls.isEmpty {
                    messages.append(ToolCallBridge.assistantTurn(
                        text: text, calls: calls))
                }
            } else if !text.isEmpty {
                messages.append(["role": role, "content": text])
            }
        }
        if !system.isEmpty {
            messages.insert(["role": "system",
                             "content": system.joined(separator: "\n\n")],
                            at: 0)
        }
        let tools = try ToolCallBridge.fromAnthropic(
            body["tools"] as? [[String: Any]] ?? [],
            stripHosted: stripUnsupportedTools)
        let cfg = sampling(body, speculationK: speculationK,
                           defaultMaxTokens: 1024,
                           maxTokensKeys: ["max_tokens"], topP: true)
        let thinking = ((body["thinking"] as? [String: Any])?["type"]
            as? String) == "enabled"
        return Anthropic(messages: messages, tools: tools.tools, config: cfg,
                         thinking: thinking,
                         stream: body["stream"] as? Bool ?? false,
                         warnings: tools.warnings)
    }

    /// POST /v1/messages/count_tokens.
    ///
    /// The request is converted exactly as /v1/messages converts it and
    /// `tokenize` -- the session's own template + tokenizer -- counts the
    /// prompt that call would prefill.  The chars/3.5 estimate this used to
    /// answer was off by 2x either way on CJK and on code, and Claude Code
    /// budgets its context window from this number.
    static func countTokens(_ body: [String: Any],
                            tokenize: ([[String: Any]], [[String: Any]], Bool)
                                async throws -> Int) async throws -> Int {
        let parsed = try anthropic(body, speculationK: 0)
        return try await tokenize(parsed.messages, parsed.tools,
                                  parsed.thinking)
    }

    // MARK: - OpenAI Responses

    struct Responses {
        var messages: [[String: Any]]
        var tools: [[String: Any]]
        var config: SamplingConfig
        var stream: Bool
        var warnings: [String] = []
    }

    /// Text from Responses content parts.  Anything that is not a text
    /// part -- input_image, input_file, refusal -- is refused by name: the
    /// old loop collected `text` fields and an image simply was not in the
    /// prompt, so the model answered a question about a picture it never
    /// saw.  Vision on this server is the chat-completions image_url part.
    private static func partsText(_ parts: [[String: Any]], at path: String)
        throws -> String {
        var text = ""
        for (pi, p) in parts.enumerated() {
            let type = p["type"] as? String ?? "?"
            switch type {
            case "input_text", "output_text", "text":
                text += p["text"] as? String ?? ""
            default:
                throw RequestRefusal(
                    param: "\(path)[\(pi)].type",
                    message: "\(path)[\(pi)] is \(show(type)), which "
                           + "/v1/responses does not carry to the model; "
                           + "only input_text/output_text parts are. For "
                           + "an image, POST /v1/chat/completions with an "
                           + "image_url part (data: URI).")
            }
        }
        return text
    }

    static func responses(_ body: [String: Any], speculationK: Int32,
                          stripUnsupportedTools: Bool = stripUnsupportedToolsEnv)
        throws -> Responses {
        try checkToolChoice(body)
        var messages: [[String: Any]] = []
        var systemText = ""
        if let inst = body["instructions"] as? String, !inst.isEmpty {
            systemText += inst
        }
        // input is a bare string or an array of typed items.  Message items
        // carry content parts; system and developer roles fold into the
        // single leading system message the chat template accepts.  Tool
        // traffic round-trips: function_call items become assistant turns
        // with tool_calls, function_call_output becomes a tool-role turn.
        if let text = body["input"] as? String {
            messages.append(["role": "user", "content": text])
        } else if let items = body["input"] as? [[String: Any]] {
            for (ii, item) in items.enumerated() {
                let kind = item["type"] as? String ?? "message"
                switch kind {
                case "message":
                    let role = item["role"] as? String ?? "user"
                    var text = ""
                    if let s = item["content"] as? String { text = s }
                    else if let parts = item["content"] as? [[String: Any]] {
                        text = try partsText(parts,
                                             at: "input[\(ii)].content")
                    }
                    if role == "system" || role == "developer" {
                        systemText += (systemText.isEmpty ? "" : "\n") + text
                    } else {
                        messages.append(["role": role, "content": text])
                    }
                case "function_call":
                    messages.append(ToolCallBridge.assistantTurn(
                        text: "",
                        calls: [(id: item["call_id"] as? String ?? "call_0",
                                 name: item["name"] as? String ?? "",
                                 argumentsJSON: item["arguments"] as? String
                                     ?? "{}")]))
                case "function_call_output":
                    var out = item["output"] as? String ?? ""
                    if out.isEmpty,
                       let parts = item["output"] as? [[String: Any]] {
                        out = try partsText(parts, at: "input[\(ii)].output")
                    }
                    messages.append(ToolCallBridge.toolResultTurn(
                        callId: item["call_id"] as? String ?? "call_0",
                        content: out))
                default:
                    continue  // reasoning items etc.
                }
            }
        }
        let tools = try ToolCallBridge.fromResponses(
            body["tools"] as? [[String: Any]] ?? [],
            stripHosted: stripUnsupportedTools)
        if !systemText.isEmpty {
            messages.insert(["role": "system", "content": systemText], at: 0)
        }
        var cfg = sampling(body, speculationK: speculationK,
                           defaultMaxTokens: 2048,
                           maxTokensKeys: ["max_output_tokens"], topP: false)
        // text.format -> the same guided decoding chat's response_format
        // reaches; Responses puts the schema at the format's top level.
        if let text = body["text"] as? [String: Any],
           let fmt = text["format"] as? [String: Any] {
            try guided(&cfg, type: fmt["type"] as? String ?? "text",
                       schema: fmt["schema"], param: "text.format",
                       schemaParam: "text.format.schema")
        }
        return Responses(messages: messages, tools: tools.tools, config: cfg,
                         stream: body["stream"] as? Bool ?? false,
                         warnings: tools.warnings)
    }
}
