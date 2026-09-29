// Copyright (c) 2026 Jiejing Zhang.
//
// What Claude Code 2.1.239 actually sends to /v1/messages, and the three
// ways this server got it wrong -- each found by running the real client
// against the server, not by reading the spec:
//
//   A. `role: "system"` INSIDE `messages` (the agent-types preamble) was
//      handed to the chat template verbatim; Qwen's template throws
//      "System message must be at the beginning" and every turn was a 500.
//   B. With stream:true that throw happened after `HTTP/1.1 200` and
//      `message_start` were on the wire: 339 bytes, then the connection
//      closed.  The client saw a stream that stopped, not an error.
//   C. A prompt over the engine's max length surfaced as
//      "Tempo9Error error 0" -- the status and the numbers stayed in the
//      server log -- and Claude Code's constant max_tokens=32000 turned
//      every prompt over 768 tokens into that error on a 32768 engine.
//
// Hermetic: the request shape is copied from the captured request; the
// 29 real tool schemas are not.  The Anthropic turn runs against closures
// (`serveAnthropicTurn`) and a byte buffer, never a socket or an engine.

import Testing
import Foundation
import Tempo9Engine
@testable import Tempo9

/// Everything the turn driver wrote, in order, with each write's close flag.
private final class Captured: @unchecked Sendable {
    var chunks: [(data: Data, close: Bool)] = []
    var out: HTTPOut { HTTPOut { [self] d, c in chunks.append((d, c)) } }
    var text: String {
        String(decoding: chunks.map(\.data).reduce(Data(), +), as: UTF8.self)
    }
    /// SSE event names, in wire order.
    var events: [String] {
        text.components(separatedBy: "\n")
            .filter { $0.hasPrefix("event: ") }
            .map { String($0.dropFirst("event: ".count)) }
    }
    /// The `data:` JSON of the first event with this name.
    func data(of event: String) -> [String: Any]? {
        let lines = text.components(separatedBy: "\n")
        guard let i = lines.firstIndex(of: "event: \(event)"),
              i + 1 < lines.count, lines[i + 1].hasPrefix("data: ") else {
            return nil
        }
        let json = String(lines[i + 1].dropFirst("data: ".count))
        return try? JSONSerialization.jsonObject(with: Data(json.utf8))
            as? [String: Any]
    }
    /// The JSON body after the HTTP head, for a non-streaming reply.
    var body: [String: Any]? {
        guard let r = text.range(of: "\r\n\r\n") else { return nil }
        return try? JSONSerialization.jsonObject(
            with: Data(text[r.upperBound...].utf8)) as? [String: Any]
    }
}

private struct Boom: Error {}

/// Does `a` occur before `b` in `s`?  False when either is missing, so a
/// failing test reports an expectation rather than trapping on an unwrap.
private func precedes(_ a: String, _ b: String, in s: String) -> Bool {
    guard let ra = s.range(of: a), let rb = s.range(of: b) else { return false }
    return ra.lowerBound < rb.lowerBound
}

private func reply(text: String = "ok", completion: Int = 1,
                   maxTokens: Int = 1024) -> LocalReply {
    LocalReply(text: text, reasoning: "", promptTokens: 10,
               completionTokens: completion, seconds: 0, engine: nil,
               maxTokens: maxTokens)
}

/// The shape of Claude Code 2.1.239's first request, minus the 29 tool
/// schemas: system as three cache_control'd text blocks, a user turn of two
/// text blocks, then a `system` role message carrying the agent-types
/// preamble, adaptive thinking, and the constant max_tokens.
private let claudeCodeBody: [String: Any] = [
    "model": "claude-sonnet-4-5",
    "max_tokens": 32000,
    "system": [
        ["type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude.",
         "cache_control": ["type": "ephemeral"]],
        ["type": "text", "text": "You are an interactive CLI tool that helps users."],
        ["type": "text", "text": String(repeating: "# Tone and style\n", count: 40),
         "cache_control": ["type": "ephemeral"]],
    ],
    "messages": [
        ["role": "user", "content": [
            ["type": "text", "text": "<system-reminder>\nAs you answer the user's questions...</system-reminder>"],
            ["type": "text", "text": "what is in this directory"],
        ]],
        ["role": "system",
         "content": "Available agent types for the Agent tool:\n- claude: Catch-all for any task\n- Explore: Read-only search agent"],
    ],
    "thinking": ["type": "adaptive", "display": "omitted"],
    "output_config": ["effort": "high"],
    "metadata": ["user_id": "{\"device_id\":\"x\",\"session_id\":\"y\"}"],
    "stream": true,
]

@Suite("Claude Code's request shape")
struct ClaudeCodeRequestTests {

    // MARK: A. role:"system" inside messages

    @Test("anthropic: a system message inside messages is folded into the leading system prompt")
    func anthropicMidConversationSystemFolded() throws {
        let parsed = try APIRequest.anthropic(claudeCodeBody, speculationK: 0)
        let roles = parsed.messages.map { $0["role"] as? String ?? "?" }
        #expect(roles == ["system", "user"],
                "the template must see one system message, at index 0: \(roles)")
        let system = parsed.messages.first?["content"] as? String ?? ""
        #expect(system.contains("You are Claude Code"))
        #expect(system.contains("# Tone and style"))
        #expect(system.contains("Available agent types for the Agent tool"),
                "the mid-conversation system text must be hoisted, not dropped")
        // In order: the top-level system first, the hoisted message after.
        #expect(precedes("You are Claude Code", "Available agent types", in: system))
        #expect(!roles.dropFirst().contains("system"),
                "no system role may survive past index 0")
        // The user turn is intact, both text blocks joined.
        let user = parsed.messages.last?["content"] as? String ?? ""
        #expect(user.contains("system-reminder") && user.contains("what is in this directory"))
        #expect(parsed.stream)
    }

    @Test("anthropic: [user, system, user] with no top-level system still leads with one system")
    func anthropicSystemOnlyMidConversation() throws {
        let parsed = try APIRequest.anthropic([
            "messages": [
                ["role": "user", "content": "first"],
                ["role": "system", "content": [["type": "text", "text": "rules"]]],
                ["role": "user", "content": "second"],
            ]], speculationK: 0)
        let roles = parsed.messages.map { $0["role"] as? String ?? "?" }
        #expect(roles == ["system", "user", "user"], "\(roles)")
        #expect(parsed.messages[0]["content"] as? String == "rules")
        #expect(parsed.messages[1]["content"] as? String == "first")
        #expect(parsed.messages[2]["content"] as? String == "second")
    }

    @Test("chat: system and developer messages mid-conversation are hoisted, other turns pass through")
    func chatMidConversationSystemFolded() throws {
        let image: [String: Any] = ["type": "image_url",
                                    "image_url": ["url": "data:image/png;base64,AAAA"]]
        let parsed = try APIRequest.chat([
            "messages": [
                ["role": "system", "content": "be brief"],
                ["role": "user", "content": "hi"],
                ["role": "developer", "content": "answer in French"],
                ["role": "user", "content": [["type": "text", "text": "look"], image]],
                ["role": "system", "content": [["type": "text", "text": "and be kind"]]],
            ]], speculationK: 0)
        let roles = parsed.messages.map { $0["role"] as? String ?? "?" }
        #expect(roles == ["system", "user", "user"], "\(roles)")
        let system = parsed.messages[0]["content"] as? String ?? ""
        #expect(system.contains("be brief") && system.contains("answer in French")
                && system.contains("and be kind"))
        #expect(precedes("be brief", "answer in French", in: system))
        #expect(precedes("answer in French", "and be kind", in: system))
        // The user turn with an image part is the same dictionary it was:
        // extractImage runs on these messages after parsing.
        let parts = parsed.messages[2]["content"] as? [[String: Any]]
        #expect(parts?.count == 2)
        #expect(parts?[1]["type"] as? String == "image_url")
    }

    // MARK: C. context length

    @Test("context budget: fits, clamps max_tokens to what is left, refuses only when the prompt alone overflows")
    func contextBudget() throws {
        #expect(ContextBudget.apply(promptTokens: 100, maxTokens: 50,
                                    maxLength: 200) == .fits)
        // Claude Code sends max_tokens 32000 every time.  A prompt that
        // fits must run, with max_tokens cut to the room that is left.
        let c = ContextBudget.apply(promptTokens: 31652, maxTokens: 32000,
                                    maxLength: 32768)
        guard case let .clamped(maxTokens, warning) = c else {
            Issue.record("31652 + 32000 on a 32768 engine must clamp, got \(c)")
            return
        }
        #expect(maxTokens == 1116)
        #expect(warning.contains("32000") && warning.contains("1116")
                && warning.contains("31652") && warning.contains("32768"),
                "the warning must carry all four numbers: \(warning)")
        // One token of room is still room.
        #expect(ContextBudget.apply(promptTokens: 32767, maxTokens: 48,
                                    maxLength: 32768)
                == .clamped(maxTokens: 1, warning: ContextBudget.clampWarning(
                    promptTokens: 32767, maxTokens: 48, clampedTo: 1,
                    maxLength: 32768)))
        // The prompt alone fills the engine: nothing to clamp to.
        #expect(ContextBudget.apply(promptTokens: 32768, maxTokens: 1,
                                    maxLength: 32768)
                == .refused(promptTokens: 32768, maxTokens: 1, maxLength: 32768))
        #expect(ContextBudget.apply(promptTokens: 33652, maxTokens: 48,
                                    maxLength: 32768)
                == .refused(promptTokens: 33652, maxTokens: 48, maxLength: 32768))
        // An engine that reported no limit imposes none here.
        #expect(ContextBudget.apply(promptTokens: 33652, maxTokens: 48,
                                    maxLength: 0) == .fits)
        // The refusal names every number the log line had, and both ways out.
        let msg = LocalSessionError.promptTooLong(
            promptTokens: 33652, maxTokens: 48, maxLength: 32768).localizedDescription
        for needle in ["33652", "48", "33700", "32768", "--max-length", "shorten"] {
            #expect(msg.contains(needle), "\(needle) missing from: \(msg)")
        }
    }

    @Test("an engine error keeps its status and detail on the way to the client, and a too-long prompt is a 400")
    func engineErrorIsNotOpaque() {
        let e = Tempo9Error.engine(status: 5,
                                   detail: "request_start: ALLSPARK_EXCEED_LIMIT_ERROR")
        let d = e.localizedDescription
        #expect(d.contains("5") && d.contains("EXCEED_LIMIT"),
                "status and detail must survive localizedDescription: \(d)")
        #expect(!d.contains("error 0"), "\(d)")
        #expect(OpenAIServer.httpStatus(for: e).hasPrefix("500"))
        let tooLong = LocalSessionError.promptTooLong(
            promptTokens: 33652, maxTokens: 48, maxLength: 32768)
        #expect(OpenAIServer.httpStatus(for: tooLong).hasPrefix("400"),
                "\(OpenAIServer.httpStatus(for: tooLong))")
        #expect(OpenAIServer.anthropicErrorType(for: tooLong) == "invalid_request_error")
        #expect(OpenAIServer.anthropicErrorType(for: e) == "api_error")
    }

    // MARK: B. streaming errors

    @Test("stream: a failure before generation is a status line, never a 200 that stops")
    func streamRenderFailureIsAStatus() async throws {
        let parsed = try APIRequest.anthropic(
            ["messages": [["role": "user", "content": "hi"]], "stream": true],
            speculationK: 0)
        let tooLong = LocalSessionError.promptTooLong(
            promptTokens: 33652, maxTokens: 48, maxLength: 32768)
        let cap = Captured()
        await OpenAIServer.serveAnthropicTurn(
            parsed, id: "msg_t", modelName: "m", out: cap.out,
            prepare: { () throws -> (Void, [String]) in throw tooLong },
            generate: { _, _ in
                Issue.record("generate ran after prepare failed")
                return reply()
            })
        let text = cap.text
        #expect(text.hasPrefix("HTTP/1.1 400"),
                "expected a 400 status line, got: \(text.prefix(120))")
        #expect(!text.contains("text/event-stream"))
        #expect(!text.contains("message_start"))
        #expect(cap.body?["type"] as? String == "error")
        let err = cap.body?["error"] as? [String: Any]
        #expect(err?["type"] as? String == "invalid_request_error")
        #expect((err?["message"] as? String)?.contains("32768") == true,
                "the body must carry the engine's numbers: \(String(describing: err))")
        #expect(cap.chunks.last?.close == true)

        // Anything else that fails before generation is a 500 with a body.
        let cap2 = Captured()
        await OpenAIServer.serveAnthropicTurn(
            parsed, id: "msg_t", modelName: "m", out: cap2.out,
            prepare: { () throws -> (Void, [String]) in throw Boom() },
            generate: { _, _ in reply() })
        #expect(cap2.text.hasPrefix("HTTP/1.1 500"), "\(cap2.text.prefix(120))")
        #expect(!cap2.text.contains("message_start"))
        #expect((cap2.body?["error"] as? [String: Any])?["type"] as? String
                == "api_error")
    }

    @Test("stream: a failure after message_start ends with an error event, and message_start carries the clamp")
    func streamPostStartFailureEndsWithErrorEvent() async throws {
        let parsed = try APIRequest.anthropic(
            ["messages": [["role": "user", "content": "hi"]],
             "max_tokens": 32000, "stream": true],
            speculationK: 0)
        let clamp = "max_tokens 32000 was clamped to 1116"
        let cap = Captured()
        await OpenAIServer.serveAnthropicTurn(
            parsed, id: "msg_t", modelName: "m", out: cap.out,
            prepare: { ((), [clamp]) },
            generate: { _, onDelta in
                onDelta(LocalDelta(content: "hel", reasoning: "", tokens: 1))
                throw Tempo9Error.engine(status: 5, detail: "wait: GPU step failed")
            })
        let text = cap.text
        #expect(text.hasPrefix("HTTP/1.1 200"), "\(text.prefix(120))")
        #expect(cap.events.first == "message_start", "\(cap.events)")
        #expect(cap.events.contains("content_block_delta"))
        #expect(cap.events.last == "error",
                "the last event must be an error, got: \(cap.events)")
        let errEvent = cap.data(of: "error")
        #expect(errEvent?["type"] as? String == "error")
        let err = errEvent?["error"] as? [String: Any]
        #expect(err?["type"] as? String == "api_error")
        #expect((err?["message"] as? String)?.contains("GPU step failed") == true,
                "\(String(describing: err))")
        #expect(cap.chunks.last?.close == true)
        // The clamp decided in prepare is reported where a streaming client
        // can see it: on message_start.
        let start = cap.data(of: "message_start")
        #expect(start?["warnings"] as? [String] == [clamp],
                "\(String(describing: start))")
    }

    @Test("non-stream: the clamp is reported in the reply, and stop_reason is judged against the clamped max_tokens")
    func clampReportedInReply() async throws {
        let parsed = try APIRequest.anthropic(
            ["messages": [["role": "user", "content": "hi"]],
             "max_tokens": 32000],
            speculationK: 0)
        #expect(parsed.config.maxTokens == 32000)
        let clamp = "max_tokens 32000 was clamped to 1116"
        let cap = Captured()
        await OpenAIServer.serveAnthropicTurn(
            parsed, id: "msg_t", modelName: "m", out: cap.out,
            prepare: { ((), [clamp]) },
            generate: { _, _ in
                reply(text: "long answer", completion: 1116, maxTokens: 1116)
            })
        #expect(cap.text.hasPrefix("HTTP/1.1 200"))
        #expect(cap.body?["warnings"] as? [String] == [clamp],
                "\(String(describing: cap.body))")
        // 1116 tokens against the CLAMPED 1116 is max_tokens; against the
        // caller's 32000 it would read as end_turn, and the client would
        // never learn its reply was cut.
        #expect(cap.body?["stop_reason"] as? String == "max_tokens",
                "\(String(describing: cap.body?["stop_reason"]))")
    }
}
