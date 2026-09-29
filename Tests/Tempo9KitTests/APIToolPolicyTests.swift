// Copyright (c) 2026 Jiejing Zhang.
//
// Anthropic built-in tools come in two classes, and the server must not
// treat them alike.
//
// A CLIENT-executed built-in (bash_*, text_editor_*, computer_*, memory_*) is a tool
// the model calls and the client runs; the server only needs its input
// schema, which Anthropic publishes per version.  Those pass through as
// ordinary function tools and the reply's call comes back as a tool_use
// with the client's name for it.
//
// A SERVER-executed built-in (web_search_*, web_fetch_*, code_execution_*,
// and the OpenAI hosted tools) is one Anthropic's servers would run
// between generations.  Nothing here runs it, so the request is refused as
// a whole with a message that says which class it is and what is
// supported -- or, only under TEMPO9_STRIP_UNSUPPORTED_TOOLS=1, stripped
// with a warning in the response.
//
// GET /v1/capabilities publishes both lists, and the last test keeps that
// document honest against the refusals the parsers actually throw.
//
// Hermetic: no engine, no model, no socket.

import Testing
import Foundation
import Tempo9Engine
@testable import Tempo9

/// The refusal a parser threw, or nil when it accepted the request.
private func refusal(_ body: () throws -> Any) -> RequestRefusal? {
    do { _ = try body(); return nil }
    catch let r as RequestRefusal { return r }
    catch { return nil }
}

private func fn(_ t: [String: Any]) -> [String: Any] {
    t["function"] as? [String: Any] ?? [:]
}
private func props(_ t: [String: Any]) -> [String: Any] {
    (fn(t)["parameters"] as? [String: Any])?["properties"]
        as? [String: Any] ?? [:]
}
private func required(_ t: [String: Any]) -> [String] {
    (fn(t)["parameters"] as? [String: Any])?["required"] as? [String] ?? []
}
private func enumOf(_ t: [String: Any], _ key: String) -> [String] {
    (props(t)[key] as? [String: Any])?["enum"] as? [String] ?? []
}
private func arguments(_ c: ToolCallBridge.Call) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: Data(c.argumentsJSON.utf8))
                 as? [String: Any])
}

private let user: [[String: Any]] = [["role": "user", "content": "hi"]]
private let custom: [String: Any] = ["name": "get_weather",
                                     "input_schema": ["type": "object"]]

@Suite("Built-in tools: client-executed pass through, server-executed are refused")
struct APIToolPolicyTests {

    // MARK: A. client-executed built-ins pass through

    @Test("anthropic: bash_20250124 becomes a function tool and its call comes back as `bash`")
    func bashPassesThrough() throws {
        let out = try ToolCallBridge.fromAnthropic(
            [["type": "bash_20250124", "name": "bash"]])
        #expect(out.warnings.isEmpty)
        let t = try #require(out.tools.first)
        #expect(fn(t)["name"] as? String == "bash")
        #expect(Set(props(t).keys) == ["command", "restart"])
        #expect((props(t)["restart"] as? [String: Any])?["type"] as? String
                == "boolean")
        #expect(!(fn(t)["description"] as? String ?? "").isEmpty,
                "a local model has never heard of `bash`; tell it")

        // The reply parser maps the call back to that name, typed by the
        // schema the conversion supplied.
        let reply = "<tool_call>\n<function=bash>\n<parameter=command>\n"
            + "git status\n</parameter>\n</function>\n</tool_call>"
        let parsed = ToolCallBridge.parse(reply, tools: out.tools)
        let call = try #require(parsed.calls.first)
        #expect(call.name == "bash")
        #expect(try arguments(call)["command"] as? String == "git status")

        let restart = ToolCallBridge.parse(
            "<tool_call>\n<function=bash>\n<parameter=restart>\ntrue\n"
            + "</parameter>\n</function>\n</tool_call>", tools: out.tools)
        #expect(try arguments(try #require(restart.calls.first))["restart"]
                as? Bool == true, "restart is declared boolean")
    }

    @Test("anthropic: each text_editor version passes through with its own command set")
    func textEditorPassesThrough() throws {
        let latest = try ToolCallBridge.fromAnthropic(
            [["type": "text_editor_20250728",
              "name": "str_replace_based_edit_tool",
              "max_characters": 10000]])
        let t = try #require(latest.tools.first)
        #expect(fn(t)["name"] as? String == "str_replace_based_edit_tool")
        #expect(required(t) == ["command", "path"])
        #expect(enumOf(t, "command") == ["view", "create", "str_replace",
                                         "insert"],
                "undo_edit was removed in text_editor_20250429")
        for k in ["view_range", "file_text", "old_str", "new_str",
                  "insert_line", "insert_text"] {
            #expect(props(t)[k] != nil, "missing \(k)")
        }

        let mid = try ToolCallBridge.fromAnthropic(
            [["type": "text_editor_20250429",
              "name": "str_replace_based_edit_tool"]])
        #expect(!enumOf(try #require(mid.tools.first), "command")
                    .contains("undo_edit"))

        let old = try ToolCallBridge.fromAnthropic(
            [["type": "text_editor_20250124", "name": "str_replace_editor"]])
        let o = try #require(old.tools.first)
        #expect(fn(o)["name"] as? String == "str_replace_editor",
                "the name is the client's, not the family's")
        #expect(enumOf(o, "command").contains("undo_edit"))

        // Typed by the schema: view_range is an array of integers.
        let reply = "<tool_call>\n<function=str_replace_based_edit_tool>\n"
            + "<parameter=command>\nview\n</parameter>\n"
            + "<parameter=path>\n/tmp/a.py\n</parameter>\n"
            + "<parameter=view_range>\n[1, 20]\n</parameter>\n"
            + "</function>\n</tool_call>"
        let parsed = ToolCallBridge.parse(reply, tools: latest.tools)
        let call = try #require(parsed.calls.first)
        #expect(call.name == "str_replace_based_edit_tool")
        let args = try arguments(call)
        #expect(args["command"] as? String == "view")
        #expect(args["path"] as? String == "/tmp/a.py")
        #expect(args["view_range"] as? [Int] == [1, 20])
    }

    @Test("anthropic: computer_20251124 passes through, display size in the description, zoom only when enabled")
    func computerPassesThrough() throws {
        let base: [String: Any] = ["type": "computer_20251124",
                                   "name": "computer",
                                   "display_width_px": 1024,
                                   "display_height_px": 768,
                                   "display_number": 1]
        let out = try ToolCallBridge.fromAnthropic([base])
        let t = try #require(out.tools.first)
        #expect(fn(t)["name"] as? String == "computer")
        #expect(required(t) == ["action"])
        for k in ["action", "coordinate", "text", "scroll_direction",
                  "scroll_amount", "duration", "start_coordinate"] {
            #expect(props(t)[k] != nil, "missing \(k)")
        }
        #expect((fn(t)["description"] as? String)?.contains("1024x768")
                == true, "the model must know the coordinate space")
        let actions = enumOf(t, "action")
        for a in ["screenshot", "left_click", "scroll", "type", "key", "wait"] {
            #expect(actions.contains(a), "missing action \(a)")
        }
        #expect(!actions.contains("zoom"),
                "zoom is an action only when enable_zoom is set")
        #expect(props(t)["region"] == nil)

        var zoomed = base
        zoomed["enable_zoom"] = true
        let z = try #require(try ToolCallBridge.fromAnthropic([zoomed])
                                .tools.first)
        #expect(enumOf(z, "action").contains("zoom"))
        #expect(props(z)["region"] != nil)

        let reply = "<tool_call>\n<function=computer>\n"
            + "<parameter=action>\nleft_click\n</parameter>\n"
            + "<parameter=coordinate>\n[500, 300]\n</parameter>\n"
            + "</function>\n</tool_call>"
        let call = try #require(
            ToolCallBridge.parse(reply, tools: out.tools).calls.first)
        #expect(call.name == "computer")
        #expect(try arguments(call)["coordinate"] as? [Int] == [500, 300])
    }

    @Test("anthropic: memory_20250818 passes through with the six commands, and its call comes back as `memory`")
    func memoryPassesThrough() throws {
        let out = try ToolCallBridge.fromAnthropic(
            [["type": "memory_20250818", "name": "memory"]])
        #expect(out.warnings.isEmpty)
        let t = try #require(out.tools.first)
        #expect(fn(t)["name"] as? String == "memory")
        // The six command types of BetaMemoryTool20250818*Command, and
        // every input any of them takes -- nothing more.
        #expect(enumOf(t, "command") == ["view", "create", "str_replace",
                                          "insert", "delete", "rename"])
        #expect(Set(props(t).keys) == ["command", "path", "view_range",
                                       "file_text", "old_str", "new_str",
                                       "insert_line", "insert_text",
                                       "old_path", "new_path"],
                "\(props(t).keys.sorted())")
        // rename takes old_path/new_path and no path, so only the command
        // is required at the top level.
        #expect(required(t) == ["command"])
        #expect((props(t)["insert_line"] as? [String: Any])?["type"]
                as? String == "integer")
        let range = props(t)["view_range"] as? [String: Any]
        #expect(range?["type"] as? String == "array")
        #expect((range?["items"] as? [String: Any])?["type"] as? String
                == "integer")
        let desc = fn(t)["description"] as? String ?? ""
        #expect(desc.contains("/memories"),
                "a local model must be told where memory lives: \(desc)")

        let reply = "<tool_call>\n<function=memory>\n<parameter=command>\n"
            + "view\n</parameter>\n<parameter=path>\n/memories\n</parameter>\n"
            + "</function>\n</tool_call>"
        let parsed = ToolCallBridge.parse(reply, tools: out.tools)
        let call = try #require(parsed.calls.first)
        #expect(call.name == "memory")
        let args = try arguments(call)
        #expect(args["command"] as? String == "view")
        #expect(args["path"] as? String == "/memories")

        // The whole /v1/messages request goes through with it.
        let whole = try APIRequest.anthropic(
            ["messages": user,
             "tools": [custom, ["type": "memory_20250818", "name": "memory"]]],
            speculationK: 0, stripUnsupportedTools: false)
        #expect(whole.tools.count == 2)
    }

    @Test("anthropic: an unknown version of a known family is refused, listing the versions it knows")
    func unknownBuiltinVersion() throws {
        let r = refusal {
            try ToolCallBridge.fromAnthropic(
                [["type": "bash_20991231", "name": "bash"]])
        }
        #expect(r?.param == "tools[0].type", "got \(r?.param ?? "nil")")
        #expect(r?.message.contains("bash_20991231") == true)
        #expect(r?.message.contains("bash_20250124") == true,
                "must list the versions it knows: \(r?.message ?? "nil")")

        let e = refusal {
            try ToolCallBridge.fromAnthropic(
                [["type": "text_editor_20990101",
                  "name": "str_replace_based_edit_tool"]])
        }
        #expect(e?.param == "tools[0].type")
        for v in ["text_editor_20250124", "text_editor_20250429",
                  "text_editor_20250728"] {
            #expect(e?.message.contains(v) == true, "must list \(v)")
        }

        let m = refusal {
            try ToolCallBridge.fromAnthropic(
                [["type": "memory_20991231", "name": "memory"]])
        }
        #expect(m?.param == "tools[0].type", "got \(m?.param ?? "nil")")
        #expect(m?.message.contains("memory_20250818") == true,
                "must list the memory version it knows: \(m?.message ?? "nil")")
    }

    // MARK: B. server-executed built-ins stay refused, accurately

    @Test("anthropic: a server-executed tool is refused naming the class and the supported types")
    func hostedToolRefusedAccurately() throws {
        let r = refusal {
            try ToolCallBridge.fromAnthropic([
                custom,
                ["type": "bash_20250124", "name": "bash"],
                ["type": "web_search_20260209", "name": "web_search"],
            ])
        }
        #expect(r?.param == "tools[2].type", "got \(r?.param ?? "nil")")
        let m = r?.message ?? ""
        #expect(m.contains("web_search_20260209"))
        #expect(m.contains("server-side"))
        #expect(m.contains("custom"))
        for family in ["bash_", "text_editor_", "computer_", "memory_"] {
            #expect(m.contains(family), "supported list must name \(family): \(m)")
        }
        #expect(!m.contains("neither runs nor has a schema for"),
                "that sentence is false for bash/text_editor/computer")

        for type in ["web_fetch_20260209", "code_execution_20260521",
                     "web_search_20250305", "tool_search_tool_regex_20251119"] {
            let r = refusal {
                try ToolCallBridge.fromAnthropic([["type": type, "name": "x"]])
            }
            #expect(r?.message.contains("server-side") == true,
                    "\(type): \(r?.message ?? "accepted")")
        }

        // The whole /v1/messages request is refused, not degraded.
        let whole = refusal {
            try APIRequest.anthropic(
                ["messages": user,
                 "tools": [["type": "web_search_20260209",
                            "name": "web_search"]]],
                speculationK: 0, stripUnsupportedTools: false)
        }
        #expect(whole?.param == "tools[0].type")
    }

    @Test("anthropic: a client toolset (no schema table here) is not called server-side")
    func toolsetRefusedHonestly() throws {
        // computer_toolset_20260801 / browser_toolset_20260801 are one
        // entry declaring a fixed set of member tools (tool-reference,
        // "Client toolsets"): a different shape from a single built-in,
        // and not tabled.  Client-executed all the same, and the message
        // must say so.
        for type in ["computer_toolset_20260801", "browser_toolset_20260801"] {
            let r = refusal {
                try ToolCallBridge.fromAnthropic([["type": type]])
            }
            #expect(r?.param == "tools[0].type", "\(type)")
            #expect(r?.message.contains("server-side") == false,
                    "\(type) is client-executed: \(r?.message ?? "nil")")
            #expect(r?.message.contains("client-executed") == true,
                    "name the class it IS: \(r?.message ?? "nil")")
            #expect(r?.message.contains("input_schema") == true,
                    "say how to get it through: \(r?.message ?? "nil")")
        }
    }

    @Test("openai: hosted tools on chat and responses name the reason and the function-only rule")
    func openAIHostedRefused() throws {
        let resp = refusal {
            try ToolCallBridge.fromResponses([["type": "web_search"]])
        }
        #expect(resp?.param == "tools[0].type")
        #expect(resp?.message.contains("server-side") == true,
                "\(resp?.message ?? "accepted")")
        #expect(resp?.message.contains("\"function\"") == true)

        let chat = refusal {
            try ToolCallBridge.fromChat([["type": "web_search_preview"]])
        }
        #expect(chat?.param == "tools[0].type")
        #expect(chat?.message.contains("server-side") == true,
                "\(chat?.message ?? "accepted")")
    }

    // MARK: E. the explicit escape hatch

    @Test("TEMPO9_STRIP_UNSUPPORTED_TOOLS=1 strips server-executed tools, warns, and the request proceeds")
    func stripMode() throws {
        let body: [String: Any] = [
            "messages": user,
            "tools": [custom, ["type": "web_search_20260209",
                               "name": "web_search"]],
        ]
        #expect(refusal {
            try APIRequest.anthropic(body, speculationK: 0,
                                     stripUnsupportedTools: false)
        }?.param == "tools[1].type", "off is the default, and off refuses")

        let stripped = try APIRequest.anthropic(body, speculationK: 0,
                                                stripUnsupportedTools: true)
        #expect(stripped.tools.count == 1)
        #expect(fn(try #require(stripped.tools.first))["name"] as? String
                == "get_weather")
        #expect(stripped.warnings.count == 1)
        #expect(stripped.warnings.first?.contains("web_search_20260209")
                == true)
        #expect(stripped.warnings.first?.contains("tools[1]") == true)

        // Only server-executed tools are stripped.  An unknown version of a
        // client-executed family is still a 400: nothing is known to strip.
        let unknown = refusal {
            try APIRequest.anthropic(
                ["messages": user,
                 "tools": [["type": "bash_20991231", "name": "bash"]]],
                speculationK: 0, stripUnsupportedTools: true)
        }
        #expect(unknown?.param == "tools[0].type")

        // Same switch on the OpenAI protocols.
        let resp = try APIRequest.responses(
            ["input": "hi",
             "tools": [["type": "function", "name": "f",
                        "parameters": ["type": "object"]],
                       ["type": "web_search"]]],
            speculationK: 0, stripUnsupportedTools: true)
        #expect(resp.tools.count == 1)
        #expect(resp.warnings.count == 1)
        let chat = try APIRequest.chat(
            ["messages": user,
             "tools": [["type": "function",
                        "function": ["name": "f",
                                     "parameters": ["type": "object"]]],
                       ["type": "web_search_preview"]]],
            speculationK: 0, stripUnsupportedTools: true)
        #expect(chat.tools.count == 1)
        #expect(chat.warnings.count == 1)

        // Nothing stripped, nothing warned.
        let clean = try APIRequest.anthropic(
            ["messages": user, "tools": [custom]],
            speculationK: 0, stripUnsupportedTools: true)
        #expect(clean.warnings.isEmpty)
    }

    // MARK: C. discovery

    @Test("GET /v1/capabilities lists protocols, tool classes and the strip switch")
    func capabilitiesDocument() throws {
        let doc = APICapabilities.document(model: "m",
                                           stripUnsupportedTools: false)
        let protocols = try #require(doc["protocols"] as? [[String: Any]])
        let ids = Set(protocols.compactMap { $0["id"] as? String })
        #expect(ids.isSuperset(of: ["chat_completions", "anthropic_messages",
                                    "responses"]), "\(ids)")

        let a = try #require(protocols.first {
            $0["id"] as? String == "anthropic_messages" })
        let types = a["tool_types"] as? [String] ?? []
        for t in ["custom", "bash_20250124", "text_editor_20250728",
                  "computer_20251124", "memory_20250818"] {
            #expect(types.contains(t), "missing \(t) in \(types)")
        }
        #expect(!types.contains { $0.hasPrefix("web_search") })
        let refused = a["refused"] as? [[String: Any]] ?? []
        #expect(!refused.isEmpty)
        #expect(refused.allSatisfy { $0["param"] is String
                                     && $0["reason"] is String })
        #expect(refused.contains { $0["param"] as? String == "tool_choice" })
        let implemented = a["implemented"] as? [[String: Any]] ?? []
        #expect(implemented.contains {
            ($0["feature"] as? String)?.contains("count_tokens") == true })

        let unsupported = try #require(doc["unsupported_tools"]
                                       as? [String: Any])
        #expect(unsupported["strip_env"] as? String
                == "TEMPO9_STRIP_UNSUPPORTED_TOOLS")
        #expect(unsupported["strip_enabled"] as? Bool == false)
        let on = APICapabilities.document(model: "m",
                                          stripUnsupportedTools: true)
        #expect((on["unsupported_tools"] as? [String: Any])?["strip_enabled"]
                as? Bool == true)

        #expect(APICapabilities.modelHint["capabilities_url"] as? String
                == "/v1/capabilities")
    }

    @Test("the capabilities document and the refusals the parsers throw do not drift")
    func capabilitiesMatchRefusals() throws {
        // One request per capability refusal, per protocol.  Validation
        // errors (a tool with no name, a schema-less json_schema) are not
        // capabilities and are not in this list or in the document.
        let png: [String: Any] = ["type": "image",
                                  "source": ["type": "base64",
                                             "media_type": "image/png",
                                             "data": "iVBORw0KGgo="]]
        let cases: [(String, () throws -> Any)] = [
            ("chat_completions", { try APIRequest.chat(
                ["messages": user, "tool_choice": "required"],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("chat_completions", { try APIRequest.chat(
                ["messages": user, "stop": ["END"]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("chat_completions", { try APIRequest.chat(
                ["messages": user, "tools": [["type": "web_search_preview"]]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("anthropic_messages", { try APIRequest.anthropic(
                ["messages": user, "tool_choice": ["type": "any"]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("anthropic_messages", { try APIRequest.anthropic(
                ["messages": user, "stop_sequences": ["Human:"]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("anthropic_messages", { try APIRequest.anthropic(
                ["messages": user,
                 "tools": [["type": "web_search_20260209", "name": "w"]]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("anthropic_messages", { try APIRequest.anthropic(
                ["messages": user, "system": [png]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("anthropic_messages", { try APIRequest.anthropic(
                ["messages": [["role": "user", "content": [png]]]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("anthropic_messages", { try APIRequest.anthropic(
                ["messages": [["role": "user", "content": [
                    ["type": "tool_result", "tool_use_id": "t",
                     "content": [png]]]]]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("responses", { try APIRequest.responses(
                ["input": "hi", "tool_choice": "required"],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("responses", { try APIRequest.responses(
                ["input": "hi", "tools": [["type": "file_search"]]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("responses", { try APIRequest.responses(
                ["input": [["type": "message", "role": "user", "content": [
                    ["type": "input_image", "image_url": "data:,"]]]]],
                speculationK: 0, stripUnsupportedTools: false) }),
            ("responses", { try APIRequest.responses(
                ["input": [["type": "function_call_output", "call_id": "c",
                            "output": [["type": "input_image",
                                        "image_url": "data:,"]]]]],
                speculationK: 0, stripUnsupportedTools: false) }),
        ]
        var thrown: [String: Set<String>] = [:]
        for (i, (proto, body)) in cases.enumerated() {
            let r = try #require(refusal(body), "case \(i) (\(proto)) accepted")
            thrown[proto, default: []].insert(APICapabilities.normalise(r.param))
        }
        let documented = APICapabilities.protocols.filter { !$0.refused.isEmpty }
        #expect(Set(documented.map(\.id)) == Set(thrown.keys),
                "protocols documenting refusals \(documented.map(\.id)) vs exercised \(thrown.keys.sorted())")
        for p in documented {
            let doc = Set(p.refused.map(\.param))
            let got = thrown[p.id] ?? []
            #expect(doc == got,
                    "\(p.id): documented \(doc.sorted()) vs thrown \(got.sorted())")
        }
    }
}
