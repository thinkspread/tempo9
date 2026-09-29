// Copyright (c) 2026 Jiejing Zhang.
//
// One test per parameter the local API used to accept and then ignore.
//
// Each of these answered 200 with a reply that quietly lacked the guarantee
// the caller asked for: tool_choice=required that the model was free to
// disregard, a stop sequence that never stopped anything, an image that
// simply was not in the prompt.  The rule now is that a parameter the server
// cannot honour is refused with 400 naming the parameter and the value, in
// the protocol's own error shape -- so the assertions here are on the
// refusal, or on the honoured behaviour, never on "it returned".
//
// Hermetic: the parsers run without an engine, a model, or a socket.

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

private let weather: [String: Any] = [
    "type": "function", "name": "get_weather",
    "parameters": ["type": "object",
                   "properties": ["city": ["type": "string"]]],
]

private let chatWeather: [String: Any] = [
    "type": "function",
    "function": ["name": "get_weather",
                 "parameters": ["type": "object"]],
]

@Suite("Local API refuses what it cannot honour")
struct APIRefusalTests {

    // MARK: 1. tool_choice

    @Test("chat: tool_choice other than auto is refused, auto and absent pass")
    func chatToolChoice() throws {
        for choice in ["required", "none"] as [Any] {
            let r = refusal {
                try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                     "tools": [chatWeather],
                                     "tool_choice": choice], speculationK: 0)
            }
            #expect(r?.param == "tool_choice", "tool_choice=\(choice) was accepted")
            #expect(r?.message.contains("\(choice)") == true,
                    "the refusal must quote the value: \(r?.message ?? "nil")")
        }
        let specific = refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "tools": [chatWeather],
                                 "tool_choice": ["type": "function",
                                                 "function": ["name": "get_weather"]]],
                                speculationK: 0)
        }
        #expect(specific?.param == "tool_choice")
        #expect(specific?.message.contains("get_weather") == true)

        #expect(refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "tools": [chatWeather], "tool_choice": "auto"],
                                speculationK: 0)
        } == nil, "auto is what the server does; it must not be refused")
        #expect(refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "tools": [chatWeather]], speculationK: 0)
        } == nil)
    }

    @Test("anthropic: tool_choice any/tool/none are refused, auto passes")
    func anthropicToolChoice() throws {
        let tool: [String: Any] = ["name": "get_weather",
                                   "input_schema": ["type": "object"]]
        for choice in [["type": "any"], ["type": "tool", "name": "get_weather"],
                       ["type": "none"]] as [[String: Any]] {
            let r = refusal {
                try APIRequest.anthropic(
                    ["messages": [["role": "user", "content": "hi"]],
                     "tools": [tool], "tool_choice": choice], speculationK: 0)
            }
            #expect(r?.param == "tool_choice",
                    "tool_choice \(choice) was accepted")
            #expect(r?.message.contains(choice["type"] as! String) == true)
        }
        #expect(refusal {
            try APIRequest.anthropic(
                ["messages": [["role": "user", "content": "hi"]],
                 "tools": [tool], "tool_choice": ["type": "auto"]],
                speculationK: 0)
        } == nil)
    }

    @Test("responses: tool_choice other than auto is refused")
    func responsesToolChoice() throws {
        let r = refusal {
            try APIRequest.responses(
                ["input": "hi", "tools": [weather],
                 "tool_choice": ["type": "function", "name": "get_weather"]],
                speculationK: 0)
        }
        #expect(r?.param == "tool_choice")
        #expect(r?.message.contains("get_weather") == true)
        #expect(refusal {
            try APIRequest.responses(["input": "hi", "tools": [weather],
                                      "tool_choice": "required"], speculationK: 0)
        }?.param == "tool_choice")
        #expect(refusal {
            try APIRequest.responses(["input": "hi", "tools": [weather],
                                      "tool_choice": "auto"], speculationK: 0)
        } == nil)
    }

    // MARK: 2. stop / stop_sequences

    @Test("stop sequences are refused, not run past")
    func stopSequences() throws {
        // The engine stops on token ids only; a string the server cannot
        // stop on must not be accepted and then decoded through.
        let chat = refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "stop": ["\n\n", "END"]], speculationK: 0)
        }
        #expect(chat?.param == "stop", "chat `stop` was accepted and ignored")
        #expect(chat?.message.contains("END") == true)
        #expect(refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "stop": "STOP"], speculationK: 0)
        }?.param == "stop", "the single-string form is the same parameter")

        let claude = refusal {
            try APIRequest.anthropic(
                ["messages": [["role": "user", "content": "hi"]],
                 "stop_sequences": ["Human:"]], speculationK: 0)
        }
        #expect(claude?.param == "stop_sequences",
                "anthropic `stop_sequences` was accepted and ignored")
        #expect(claude?.message.contains("Human:") == true)

        // Empty and null mean "no stop sequences", which is honoured.
        #expect(refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "stop": [] as [String]], speculationK: 0)
        } == nil)
        #expect(refusal {
            try APIRequest.chat(["messages": [["role": "user", "content": "hi"]],
                                 "stop": NSNull()], speculationK: 0)
        } == nil)
        #expect(refusal {
            try APIRequest.anthropic(
                ["messages": [["role": "user", "content": "hi"]],
                 "stop_sequences": [] as [String]], speculationK: 0)
        } == nil)
    }

    // MARK: 3. anthropic tool_result with non-text content

    @Test("anthropic: a tool_result carrying an image is refused, not blanked")
    func toolResultNonText() throws {
        let body: [String: Any] = ["messages": [
            ["role": "assistant", "content": [
                ["type": "tool_use", "id": "toolu_1", "name": "screenshot",
                 "input": [:] as [String: Any]]]],
            ["role": "user", "content": [
                ["type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [["type": "image",
                              "source": ["type": "base64",
                                         "media_type": "image/png",
                                         "data": "iVBORw0KGgo="]]]]]],
        ]]
        let r = refusal { try APIRequest.anthropic(body, speculationK: 0) }
        #expect(r != nil, "the image result became an empty tool message")
        #expect(r?.param == "messages[1].content[0].content[0].type",
                "got \(r?.param ?? "nil")")
        #expect(r?.message.contains("image") == true)

        // Text results, string or blocks, still round-trip.
        let ok = try APIRequest.anthropic(["messages": [
            ["role": "assistant", "content": [
                ["type": "tool_use", "id": "toolu_1", "name": "ls",
                 "input": [:] as [String: Any]]]],
            ["role": "user", "content": [
                ["type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [["type": "text", "text": "a.txt"]]]]],
        ]], speculationK: 0)
        #expect(ok.messages.last?["role"] as? String == "tool")
        #expect(ok.messages.last?["content"] as? String == "a.txt")
    }

    // MARK: 4. responses image input

    @Test("responses: an input_image part is refused, not dropped")
    func responsesImageInput() throws {
        let body: [String: Any] = ["input": [
            ["type": "message", "role": "user", "content": [
                ["type": "input_text", "text": "what is this"],
                ["type": "input_image",
                 "image_url": "data:image/png;base64,iVBORw0KGgo="]]],
        ]]
        let r = refusal { try APIRequest.responses(body, speculationK: 0) }
        #expect(r != nil, "the image was silently left out of the prompt")
        #expect(r?.param == "input[0].content[1].type",
                "got \(r?.param ?? "nil")")
        #expect(r?.message.contains("input_image") == true)
    }

    // MARK: 5. responses text.format

    @Test("responses: text.format reaches guided decoding, or is refused")
    func responsesTextFormat() throws {
        let schema: [String: Any] = ["type": "object",
                                     "properties": ["ok": ["type": "boolean"]]]
        let strict = try APIRequest.responses(
            ["input": "hi", "text": ["format": ["type": "json_schema",
                                               "name": "answer",
                                               "schema": schema]]],
            speculationK: 0)
        #expect(strict.config.responseFormat == "json_schema",
                "text.format json_schema was accepted and ignored")
        #expect(strict.config.responseSchema?.contains("\"ok\"") == true)

        let loose = try APIRequest.responses(
            ["input": "hi", "text": ["format": ["type": "json_object"]]],
            speculationK: 0)
        #expect(loose.config.responseFormat == "json_object")

        let plain = try APIRequest.responses(
            ["input": "hi", "text": ["format": ["type": "text"]]],
            speculationK: 0)
        #expect(plain.config.responseFormat == nil)

        // Same refusal shape as chat's response_format: a schema request
        // with no schema is not downgraded to free-form JSON.
        let missing = refusal {
            try APIRequest.responses(
                ["input": "hi", "text": ["format": ["type": "json_schema",
                                                   "name": "answer"]]],
                speculationK: 0)
        }
        #expect(missing?.param == "text.format.schema")
        #expect(missing?.message ==
                "text.format.json_schema requires a 'schema' object")

        let unknown = refusal {
            try APIRequest.responses(
                ["input": "hi", "text": ["format": ["type": "yaml"]]],
                speculationK: 0)
        }
        #expect(unknown?.param == "text.format.type")
        #expect(unknown?.message.contains("yaml") == true)
    }

    // MARK: 6. count_tokens

    @Test("count_tokens answers with the tokenizer, not a character estimate")
    func countTokensUsesTokenizer() async throws {
        // 3500 characters of text: the old estimate answered ~1000.  The
        // tokenizer stand-in answers something the estimate cannot produce,
        // and records what it was asked to tokenize.
        let text = String(repeating: "abcdefghij", count: 350)
        let body: [String: Any] = [
            "system": "be brief",
            "messages": [["role": "user", "content": text]],
            "tools": [["name": "get_weather",
                       "input_schema": ["type": "object"]]],
        ]
        final class Seen: @unchecked Sendable {
            var messages: [[String: Any]] = []
            var tools: [[String: Any]] = []
        }
        let seen = Seen()
        let n = try await APIRequest.countTokens(body) { messages, tools, _ in
            seen.messages = messages
            seen.tools = tools
            return 7
        }
        #expect(n == 7, "input_tokens=\(n) did not come from the tokenizer")
        // What it tokenizes is the prompt /v1/messages would prefill: the
        // same conversion, system folded into a leading system message and
        // the tools in template shape.
        let asServed = try APIRequest.anthropic(body, speculationK: 0)
        #expect(seen.messages.count == asServed.messages.count)
        #expect(seen.messages.first?["role"] as? String == "system")
        #expect(seen.messages.first?["content"] as? String == "be brief")
        #expect(seen.tools.count == 1)
        #expect((seen.tools[0]["function"] as? [String: Any])?["name"]
                as? String == "get_weather")
    }

    // MARK: 7. non-function tools

    @Test("responses: a hosted tool is refused, not dropped from the list")
    func responsesHostedToolRefused() throws {
        let r = refusal {
            try ToolCallBridge.fromResponses([
                weather,
                ["type": "web_search"],
            ])
        }
        #expect(r != nil, "web_search was dropped and the request went on")
        #expect(r?.param == "tools[1].type", "got \(r?.param ?? "nil")")
        #expect(r?.message.contains("web_search") == true)

        let r2 = refusal {
            try APIRequest.responses(
                ["input": "hi", "tools": [["type": "file_search",
                                           "vector_store_ids": ["vs_1"]]]],
                speculationK: 0)
        }
        #expect(r2?.param == "tools[0].type")

        let ok = try ToolCallBridge.fromResponses([weather]).tools
        #expect(ok.count == 1)
        #expect((ok[0]["function"] as? [String: Any])?["name"] as? String
                == "get_weather")
    }
}
