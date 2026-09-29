// Copyright (c) 2026 Jiejing Zhang.
//
// The Ollama routes (/api/chat, /api/generate) answer a request that does
// not fit the engine's max length the way the other three protocols do,
// in Ollama's own shape.  Before this suite they called the engine
// through one catch: a prompt that alone overflowed was a 500 with a
// readable message (or, when streaming, a 200 followed by an error line),
// and a clamped num_predict went unreported.  Ollama's own shape for a
// request it rejects is 400 with `{"error": "<string>"}` -- a string, not
// an object -- and that is what a too-long prompt gets here.
//
// Hermetic: the turn runs against closures (`serveOllamaTurn`) and a byte
// buffer, never a socket or an engine.  The clamp itself is decided in
// LocalSession.prepare through ContextBudget; what is pinned here is that
// num_predict feeds that decision and that the outcome reaches the client.

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
    /// The bytes after the HTTP head.
    var payload: String {
        guard let r = text.range(of: "\r\n\r\n") else { return "" }
        return String(text[r.upperBound...])
    }
    /// The JSON body of a non-streaming reply.
    var body: [String: Any]? {
        try? JSONSerialization.jsonObject(with: Data(payload.utf8))
            as? [String: Any]
    }
    /// Each NDJSON line as an object, in wire order.
    var lines: [[String: Any]] {
        payload.split(separator: "\n").compactMap {
            try? JSONSerialization.jsonObject(with: Data($0.utf8))
                as? [String: Any]
        }
    }
}

private struct Boom: Error {}

private func reply(text: String = "ok", completion: Int = 1,
                   maxTokens: Int = 1024) -> LocalReply {
    LocalReply(text: text, reasoning: "", promptTokens: 10,
               completionTokens: completion, seconds: 0, engine: nil,
               maxTokens: maxTokens)
}

private func request(stream: Bool, numPredict: Int? = nil)
    -> OllamaAPI.ChatRequest {
    var body: [String: Any] = [
        "model": "m", "stream": stream,
        "messages": [["role": "user", "content": "hi"]],
    ]
    if let n = numPredict { body["options"] = ["num_predict": n] }
    return OllamaAPI.chatRequest(from: body)
}

private let tooLong = LocalSessionError.promptTooLong(
    promptTokens: 33652, maxTokens: 48, maxLength: 32768)

@Suite("Ollama: a prompt that does not fit is answered in Ollama's shape")
struct OllamaContextBudgetTests {

    @Test("a prompt that alone overflows is a 400 with {\"error\": string}, streaming or not")
    func promptTooLongIs400() async throws {
        for stream in [true, false] {
            let cap = Captured()
            await OpenAIServer.serveOllamaTurn(
                request(stream: stream), modelName: "m", generate: false,
                tools: [], out: cap.out,
                prepare: { () throws -> (Void, [String]) in throw tooLong },
                run: { _, _ in
                    Issue.record("run ran after prepare failed (stream=\(stream))")
                    return reply()
                })
            let text = cap.text
            #expect(text.hasPrefix("HTTP/1.1 400"),
                    "stream=\(stream): expected a 400 status line, got: \(text.prefix(160))")
            #expect(!text.contains("x-ndjson"),
                    "stream=\(stream): no stream head before the refusal")
            let err = cap.body?["error"]
            #expect(err is String,
                    "stream=\(stream): Ollama's error is a string, got \(String(describing: err))")
            let msg = err as? String ?? ""
            for needle in ["33652", "32768", "--max-length"] {
                #expect(msg.contains(needle),
                        "stream=\(stream): the body must carry \(needle): \(msg)")
            }
            #expect(cap.chunks.last?.close == true)
        }

        // /api/generate takes the same path.
        let gen = Captured()
        await OpenAIServer.serveOllamaTurn(
            request(stream: true), modelName: "m", generate: true,
            tools: [], out: gen.out,
            prepare: { () throws -> (Void, [String]) in throw tooLong },
            run: { _, _ in reply() })
        #expect(gen.text.hasPrefix("HTTP/1.1 400"), "\(gen.text.prefix(160))")

        // Anything else that fails before generation is still the server's
        // fault: 500, same shape.
        let boom = Captured()
        await OpenAIServer.serveOllamaTurn(
            request(stream: false), modelName: "m", generate: false,
            tools: [], out: boom.out,
            prepare: { () throws -> (Void, [String]) in throw Boom() },
            run: { _, _ in reply() })
        #expect(boom.text.hasPrefix("HTTP/1.1 500"), "\(boom.text.prefix(160))")
        #expect(boom.body?["error"] is String)
    }

    @Test("num_predict is the max_tokens the budget clamps; the clamp is reported on the final object and as done_reason length")
    func clampReportedInReply() async throws {
        // num_predict lands where ContextBudget reads max_tokens, so the
        // same decision the other protocols get applies here.
        let cfg = OpenAIServer.ollamaSamplingConfig(
            request(stream: true, numPredict: 32000), speculationK: 0)
        #expect(cfg.maxTokens == 32000)
        let outcome = ContextBudget.apply(promptTokens: 31652,
                                          maxTokens: cfg.maxTokens,
                                          maxLength: 32768)
        guard case let .clamped(maxTokens, clamp) = outcome else {
            Issue.record("expected a clamp, got \(outcome)")
            return
        }
        #expect(maxTokens == 1116)
        #expect(clamp.contains("num_predict") || clamp.contains("max_tokens"))

        // Streaming: the warning rides on the terminal object (the one with
        // done:true), not on every content line, and a reply that ran into
        // the clamped budget says so as done_reason "length" -- judged
        // against the CLAMPED value, so the client learns its reply was cut.
        let cap = Captured()
        await OpenAIServer.serveOllamaTurn(
            request(stream: true, numPredict: 32000), modelName: "m",
            generate: false, tools: [], out: cap.out,
            prepare: { ((), [clamp]) },
            run: { _, onDelta in
                onDelta(LocalDelta(content: "hel", reasoning: "", tokens: 1))
                return reply(text: "hel", completion: 1116, maxTokens: 1116)
            })
        #expect(cap.text.hasPrefix("HTTP/1.1 200"), "\(cap.text.prefix(160))")
        #expect(cap.text.contains("application/x-ndjson"))
        let lines = cap.lines
        let last = try #require(lines.last)
        #expect(last["done"] as? Bool == true, "\(last)")
        #expect(last["warnings"] as? [String] == [clamp], "\(last)")
        #expect(last["done_reason"] as? String == "length", "\(last)")
        for line in lines.dropLast() {
            #expect(line["warnings"] == nil, "only the final object: \(line)")
        }
        #expect(cap.chunks.last?.close == true)

        // Non-streaming: same object, whole reply.
        let single = Captured()
        await OpenAIServer.serveOllamaTurn(
            request(stream: false, numPredict: 32000), modelName: "m",
            generate: true, tools: [], out: single.out,
            prepare: { ((), [clamp]) },
            run: { _, _ in
                reply(text: "long answer", completion: 1116, maxTokens: 1116)
            })
        #expect(single.text.hasPrefix("HTTP/1.1 200"))
        #expect(single.body?["warnings"] as? [String] == [clamp],
                "\(String(describing: single.body))")
        #expect(single.body?["done_reason"] as? String == "length")
        #expect(single.body?["response"] as? String == "long answer")

        // Nothing clamped, nothing reported; a reply that stopped short of
        // its budget is "stop".
        let clean = Captured()
        await OpenAIServer.serveOllamaTurn(
            request(stream: false), modelName: "m", generate: false,
            tools: [], out: clean.out,
            prepare: { ((), []) },
            run: { _, _ in reply(text: "ok", completion: 2, maxTokens: 1116) })
        #expect(clean.body?["warnings"] == nil, "\(String(describing: clean.body))")
        #expect(clean.body?["done_reason"] as? String == "stop")
    }

    @Test("stream: a failure after the head is an error line, not a stream that stops")
    func postHeadFailureIsAnErrorLine() async throws {
        let cap = Captured()
        await OpenAIServer.serveOllamaTurn(
            request(stream: true), modelName: "m", generate: false,
            tools: [], out: cap.out,
            prepare: { ((), []) },
            run: { _, onDelta in
                onDelta(LocalDelta(content: "hel", reasoning: "", tokens: 1))
                throw Tempo9Error.engine(status: 5, detail: "wait: GPU step failed")
            })
        #expect(cap.text.hasPrefix("HTTP/1.1 200"), "\(cap.text.prefix(160))")
        let lines = cap.lines
        #expect(lines.count == 2, "\(lines)")
        #expect((lines.last?["error"] as? String)?.contains("GPU step failed")
                == true, "\(String(describing: lines.last))")
        #expect(cap.chunks.last?.close == true)
    }
}
