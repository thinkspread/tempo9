// Copyright (c) 2026 Jiejing Zhang.
//
// GET /v1/capabilities: what the local API serves, refuses, and strips.
//
// Hand-maintained, next to the refusal code, so it cannot drift silently:
// APIToolPolicyTests throws every capability refusal the parsers know and
// compares the set with this document in both directions -- a refusal
// that is not documented fails, and a documented refusal nothing throws
// fails too.  Validation errors (a tool with no name, a json_schema with
// no schema) are not capabilities and are not listed.

import Foundation

enum APICapabilities {
    /// A request parameter this server refuses today, and why.
    struct Refused {
        /// The parameter as RequestRefusal names it, with indices blanked
        /// (`tools[].type`, `messages[].content[].type`).
        let param: String
        let reason: String
    }

    struct Dialect {
        let id: String
        let routes: [String]
        /// Tool `type` values accepted in `tools[]`.
        let toolTypes: [String]
        let refused: [Refused]
        /// (feature, how it is honoured).
        let implemented: [(String, String)]
    }

    private static let toolChoice = Refused(
        param: "tool_choice",
        reason: "only \"auto\" or absent: the model decides, and nothing "
              + "here enforces \"none\", \"required\"/\"any\" or a named tool")

    private static func stop(_ key: String) -> Refused {
        Refused(param: key,
                reason: "the engine stops on token ids only; a stop string "
                      + "would be accepted and generated straight through")
    }

    /// Server-executed tool families, both vendors' spellings.
    static let serverExecuted: [String] =
        ToolCallBridge.AnthropicBuiltin.serverFamilies.sorted().map { $0 + "_*" }
        + ToolCallBridge.openAIHostedTypes.sorted()

    static let anthropicToolTypes: [String] =
        ["custom"] + ToolCallBridge.AnthropicBuiltin.allVersions

    static let protocols: [Dialect] = [
        Dialect(
            id: "chat_completions",
            routes: ["POST /v1/chat/completions"],
            toolTypes: ["function"],
            refused: [
                toolChoice, stop("stop"),
                Refused(param: "tools[].type",
                        reason: "hosted tools (web_search_preview, "
                              + "file_search, code_interpreter, ...) are not "
                              + "executed here; only \"function\" tools "
                              + "reach the model"),
            ],
            implemented: [
                ("tools (function)", "rendered by the chat template; calls "
                     + "parsed from the reply, arguments typed by the schema"),
                ("response_format", "json_object and json_schema via guided "
                     + "decoding"),
                ("image_url (data: URI)", "one image per request, when a "
                     + "vision tower is loaded"),
                ("stream", "SSE; a final chunk carries usage"),
                ("system/developer messages anywhere in messages[]",
                 "folded, in order, into the one leading system message "
                     + "the chat template accepts"),
                ("max_tokens over --max-length",
                 "clamped to the room the prompt leaves, reported in "
                     + "warnings and as finish_reason \"length\"; a prompt "
                     + "that alone overflows is a 400 naming the numbers"),
            ]),
        Dialect(
            id: "anthropic_messages",
            routes: ["POST /v1/messages", "POST /v1/messages/count_tokens"],
            toolTypes: anthropicToolTypes,
            refused: [
                toolChoice, stop("stop_sequences"),
                Refused(param: "tools[].type",
                        reason: "server-executed tools (web_search_*, "
                              + "web_fetch_*, code_execution_*, "
                              + "tool_search_*) are not executed here; the "
                              + "client toolsets (computer_toolset_*, "
                              + "browser_toolset_*: one entry declaring "
                              + "many member tools, no schema table here) "
                              + "and an unknown version of "
                              + "bash/text_editor/computer/memory are "
                              + "refused too, each with its own message"),
                Refused(param: "system[].type",
                        reason: "only text blocks reach the model"),
                Refused(param: "messages[].content[].type",
                        reason: "only text, tool_use and tool_result "
                              + "blocks; no image or document input on "
                              + "this route"),
                Refused(param: "messages[].content[].content[].type",
                        reason: "a tool_result must be text; an image "
                              + "result would reach the model as an empty "
                              + "string"),
            ],
            implemented: [
                ("tools (custom, bash_*, text_editor_*, computer_*, memory_*)",
                 "custom tools pass through; the client-executed built-ins "
                     + "are given their published input schema and come "
                     + "back as tool_use under the client's name"),
                ("count_tokens", "the real tokenizer over the prompt "
                     + "/v1/messages would prefill"),
                ("thinking", "{type: enabled} turns the model's thinking "
                     + "on; the reply carries a thinking block"),
                ("stream", "typed SSE events with incremental tool_use "
                     + "blocks; a failure after message_start ends the "
                     + "stream with an error event"),
                ("system messages inside messages[]",
                 "folded, in order, after the top-level system into the "
                     + "one leading system message the chat template "
                     + "accepts (Claude Code sends one per turn)"),
                ("max_tokens over --max-length",
                 "clamped to the room the prompt leaves, reported in "
                     + "warnings (message_start when streaming) and as "
                     + "stop_reason max_tokens; a prompt that alone "
                     + "overflows is a 400 naming the numbers"),
            ]),
        Dialect(
            id: "responses",
            routes: ["POST /v1/responses"],
            toolTypes: ["function"],
            refused: [
                toolChoice,
                Refused(param: "tools[].type",
                        reason: "hosted tools (web_search, file_search, "
                              + "code_interpreter, ...) are not executed "
                              + "here; only \"function\" tools reach the "
                              + "model"),
                Refused(param: "input[].content[].type",
                        reason: "input_image / input_file are not carried "
                              + "to the model; for an image use "
                              + "/v1/chat/completions with an image_url "
                              + "part"),
                Refused(param: "input[].output[].type",
                        reason: "a function_call_output must be text"),
            ],
            implemented: [
                ("text.format", "json_object and json_schema via guided "
                     + "decoding"),
                ("function_call / function_call_output", "round-trip as "
                     + "assistant tool_calls and tool-role turns"),
                ("stream", "the canonical output_item / arguments.delta "
                     + "event ladder"),
            ]),
        Dialect(
            id: "ollama",
            routes: ["POST /api/chat", "POST /api/generate", "GET /api/tags",
                     "POST /api/show", "GET /api/ps", "GET /api/version"],
            toolTypes: ["function"],
            refused: [],
            implemented: [
                ("tools", "same rules as chat_completions"),
                ("images", "base64, when a vision tower is loaded"),
                ("options.num_predict over --max-length",
                 "clamped to the room the prompt leaves, reported in a "
                     + "top-level warnings array on the final object (the "
                     + "done:true line when streaming) and as done_reason "
                     + "\"length\"; a prompt that alone overflows is a 400 "
                     + "{\"error\": ...} naming the numbers -- refused, "
                     + "not truncated as Ollama would"),
            ]),
    ]

    /// What /v1/models points at, for clients that only read that.
    static let modelHint: [String: Any] = [
        "capabilities_url": "/v1/capabilities",
        "tool_types": Dictionary(uniqueKeysWithValues: protocols.map {
            ($0.id, $0.toolTypes) }),
    ]

    /// `tools[2].type` -> `tools[].type`, so a thrown param can be compared
    /// with a documented one.
    static func normalise(_ param: String) -> String {
        param.replacingOccurrences(of: #"\[\d+\]"#, with: "[]",
                                   options: .regularExpression)
    }

    static func document(model: String, stripUnsupportedTools: Bool)
        -> [String: Any] {
        [
            "object": "capabilities",
            "model": model,
            "protocols": protocols.map { p -> [String: Any] in
                ["id": p.id, "routes": p.routes, "tool_types": p.toolTypes,
                 "refused": p.refused.map {
                     ["param": $0.param, "reason": $0.reason] },
                 "implemented": p.implemented.map {
                     ["feature": $0.0, "how": $0.1] }]
            },
            "unsupported_tools": [
                "policy": "a request carrying a tool this server cannot "
                    + "run is refused as a whole with 400 naming "
                    + "tools[i].type; nothing is silently dropped",
                "server_executed": serverExecuted,
                "client_executed_passthrough":
                    ToolCallBridge.AnthropicBuiltin.allVersions,
                "strip_env": APIRequest.stripUnsupportedToolsEnvVar,
                "strip_enabled": stripUnsupportedTools,
                "strip_effect": "when set to 1, server-executed tools are "
                    + "removed from the request instead; one warning per "
                    + "tool goes to the server log and to a top-level "
                    + "`warnings` array in the response",
            ] as [String: Any],
            "docs": "manual/claude-code.md",
        ]
    }
}
