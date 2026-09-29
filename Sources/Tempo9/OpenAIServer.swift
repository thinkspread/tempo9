// Copyright (c) 2026 Jiejing Zhang.
//
// A local OpenAI-compatible endpoint over the in-process engine.
//
// It hands Cursor, Continue, aider, Claude Code, Codex or a curl one-liner a
// `http://127.0.0.1:<port>/v1` to talk to, with zero extra installs:
//
//   POST /v1/chat/completions   (stream and non-stream; SSE for Cursor)
//   POST /v1/messages           (Anthropic dialect; Claude Code)
//   POST /v1/responses          (OpenAI Responses; Codex)
//   GET  /v1/models             (reports the loaded gguf)
//   GET  /v1/capabilities       (what is served, refused and stripped)
//
// Localhost only.  No LAN exposure, no embeddings.
//
// Concurrency is the engine's, not this file's: requests are NOT serialized
// here (see `gate` below), and the engine batches whatever arrives, up to the
// max_batch the session was created with.

import CoreGraphics
import ImageIO
import Tempo9Engine
import Foundation
import Network

/// Why the listener never came up.  Both cases used to be invisible.
public enum ListenError: Error, CustomStringConvertible {
    case timedOut(port: UInt16)
    case bindFailed(port: UInt16, underlying: Error)

    public var description: String {
        switch self {
        case .timedOut(let p):
            return "listener on 127.0.0.1:\(p) did not become ready within 10s"
        case .bindFailed(let p, let e):
            return "could not bind 127.0.0.1:\(p): \(e) "
                 + "(a previous server on this port may still be closing)"
        }
    }
}

/// One-shot, lock-guarded: stateUpdateHandler runs on the listener queue and
/// may fire more than once; only the first outcome decides.
private final class ListenOutcome: @unchecked Sendable {
    private let lock = NSLock()
    private var settled = false
    private var stored: Error?
    func finish(_ e: Error?) {
        lock.lock(); defer { lock.unlock() }
        if settled { return }
        settled = true
        stored = e
    }
    var error: Error? {
        lock.lock(); defer { lock.unlock() }
        return stored
    }
}

public final class OpenAIServer: @unchecked Sendable {
    private let session: LocalSession
    private let modelName: String
    private let token: String?
    public let port: UInt16
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "openai-server")
    /// NOT a mutex, whatever the name says. SerialGate is an actor, and
    /// actors are reentrant: a request holds it only until its first
    /// `await`, then the next one enters. Two closures that each sleep
    /// 0.5 s through this exact actor both start at t=0 (checked
    /// 2026-09-28). Concurrent generations therefore reach the engine, which
    /// is what lets it batch them. If one-at-a-time is ever wanted, this is
    /// not the mechanism that provides it.
    private let gate = SerialGate()

    /// Turns a decoded picture into what the engine needs, or nil when this
    /// build has no vision front end loaded.
    ///
    /// Injected rather than built here: the tower is 116 MB (Gemma) or 1.7 GB
    /// (Qwen) and there is exactly one of it in the process, so the server
    /// borrows the one the app already loaded rather than opening a second.
    /// It also keeps this file ignorant of towers, which is why it compiled
    /// for a year without one.
    public var encodeImage: ((CGImage) async throws -> ImagePlacement?)?

    /// MTP speculation depth applied to every request this server serves
    /// (0 = off).  The engine's adaptive verification varies the effective
    /// k downward from here per request.
    public var defaultSpeculationK: Int32 = 0

    /// What the host knows about the loaded weights that the engine does not
    /// (size, quantization, family).  Ollama's discovery calls report it;
    /// anything left nil is reported as unknown rather than invented.
    public var modelInfo: Tempo9ModelInfo?

    /// What GET /api/version answers.
    ///
    /// Deliberately not a real Ollama version.  Some clients gate features on
    /// a minimum, so the number is high enough to clear those gates and
    /// semver-parseable for the ones that parse it -- and the prerelease tag
    /// says plainly that this is not Ollama.  Claiming to BE a version we do
    /// not implement would turn a visible rejection into a silent feature
    /// mismatch, which is the worse failure.
    public var ollamaVersion = "0.12.0-tempo9"

    public init(session: LocalSession, modelName: String,
                port: UInt16 = 11435, token: String? = nil,
                encodeImage: ((CGImage) async throws -> ImagePlacement?)?
                    = nil) {
        self.session = session
        self.modelName = modelName
        self.port = port
        self.token = token
        self.encodeImage = encodeImage
    }


    /// A message a caller can act on.
    ///
    /// `localizedDescription` on a plain Swift error is the type name, so the
    /// engine's status code and detail -- the only part that says WHY -- never
    /// reached the client. This keeps them.
    static func describe(_ error: Error) -> String {
        if let e = error as? Tempo9Error { return "\(e)" }
        let d = error.localizedDescription
        return d.isEmpty ? "\(error)" : d
    }

    /// TEMPO9_STRIP_UNSUPPORTED_TOOLS=1: the parser stripped a
    /// server-executed tool instead of refusing.  One line per tool in the
    /// log, and the same lines under a top-level `warnings` in the reply
    /// (the `warnings:` argument on send/sendSSE/sendEvent) -- an env var
    /// set on purpose is not a silent downgrade, but it must not be a
    /// quiet one either.
    private func logWarnings(_ warnings: [String]) {
        for w in warnings {
            FileHandle.standardError.write(
                Data("[tempo9] warning: \(w)\n".utf8))
        }
    }

    private static func withWarnings(_ obj: [String: Any],
                                     _ warnings: [String]) -> [String: Any] {
        guard !warnings.isEmpty else { return obj }
        var o = obj
        o["warnings"] = (o["warnings"] as? [String] ?? []) + warnings
        return o
    }

    public func start() throws {
        let params = NWParameters.tcp
        // SO_REUSEADDR.  Without it, restarting on the same port inside the
        // TIME_WAIT window fails to bind -- which is every restart during a
        // benchmark sweep, and every "stop the server and start it again"
        // a user does.  Server sockets have wanted this since Berkeley.
        params.allowLocalEndpointReuse = true
        // Loopback only.  Binding the wildcard and hoping nobody routes to
        // it is not a privacy story; requiring the interface is.
        params.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!)
        let l = try NWListener(using: params)
        l.newConnectionHandler = { [weak self] conn in
            self?.serve(conn)
        }

        // NWListener.start(queue:) is ASYNCHRONOUS.  It returns before the
        // socket is bound, and a bind failure is delivered only to
        // stateUpdateHandler.  With no handler installed, an EADDRINUSE --
        // the ordinary case when the previous server on this port is still
        // in TIME_WAIT -- was swallowed whole: start() returned, the caller
        // printed "http://127.0.0.1:<port>/v1", and the process then sat
        // there forever answering nothing, with the engine loop happily
        // logging an idle stat line every five seconds.
        //
        // It reads exactly like a hung engine and it is not one; nothing is
        // listening.  Three benchmark runs died on it before anyone thought
        // to ask lsof whether the port was open.  So: wait for .ready, and
        // make a failure loud and fatal rather than silent and confusing.
        let sem = DispatchSemaphore(value: 0)
        let box = ListenOutcome()
        l.stateUpdateHandler = { state in
            switch state {
            case .ready:
                box.finish(nil); sem.signal()
            case .failed(let e):
                box.finish(e); sem.signal()
            case .waiting(let e):
                // .waiting is Network.framework's "retryable", but for a
                // fixed loopback port it means the address is taken and
                // will stay taken.  Waiting silently is the bug.
                box.finish(e); sem.signal()
            default:
                break
            }
        }
        l.start(queue: queue)
        if sem.wait(timeout: .now() + 10) == .timedOut {
            l.cancel()
            throw ListenError.timedOut(port: port)
        }
        if let e = box.error {
            l.cancel()
            throw ListenError.bindFailed(port: port, underlying: e)
        }
        listener = l
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - per-connection

    private func serve(_ conn: NWConnection) {
        conn.start(queue: queue)
        readRequest(conn, buffer: Data())
    }

    private func readRequest(_ conn: NWConnection, buffer: Data) {
        conn.receive(minimumIncompleteLength: 1,
                     maximumLength: 1 << 20) { [weak self] data, _, done, err in
            guard let self, err == nil else { conn.cancel(); return }
            var buf = buffer
            if let data { buf.append(data) }
            if let request = HTTPRequest(complete: buf) {
                let work = Task { await self.route(request, conn) }
                self.watchDisconnect(conn, cancelling: work)
            } else if done {
                conn.cancel()
            } else {
                self.readRequest(conn, buffer: buf)
            }
        }
    }

    private func route(_ req: HTTPRequest, _ conn: NWConnection) async {
        if let token, req.headers["authorization"] != "Bearer \(token)" {
            send(conn, status: "401 Unauthorized",
                 json: ["error": ["message": "invalid api key"]])
            return
        }
        switch (req.method, req.path) {
        case ("GET", "/v1/models"):
            // `tempo9` mirrors the tool-type list and points at
            // /v1/capabilities, for clients that only ever read this.
            send(conn, json: [
                "object": "list",
                "data": [["id": modelName, "object": "model",
                          "owned_by": "tempo9",
                          "tempo9": APICapabilities.modelHint]],
            ])
        case ("GET", "/v1/capabilities"):
            // Discovery: per protocol, the tool types this server accepts,
            // the parameters it refuses today and why, and what it does
            // implement.  A static document kept beside the refusal code,
            // with a test that it matches what the parsers throw.
            send(conn, json: APICapabilities.document(
                model: modelName,
                stripUnsupportedTools: APIRequest.stripUnsupportedToolsEnv))
        case ("POST", "/v1/chat/completions"):
            await chat(req, conn)
        case ("POST", "/v1/messages"):
            await anthropicMessages(req, conn)
        case ("POST", "/v1/responses"):
            // OpenAI's Responses API — what Codex CLI speaks (0.149 removed
            // chat-completions support outright).  Third protocol, same
            // translation core.
            await responses(req, conn)
        case ("POST", "/v1/messages/count_tokens"):
            // Claude Code calls this on startup (model validation) and per
            // turn (context management); a 404 here reads as "model does
            // not exist" in its UI.  Answered by the real tokenizer over
            // the prompt /v1/messages would prefill -- Claude Code budgets
            // its context window from this number, and the chars/3.5
            // estimate this used to be was off by 2x on CJK and on code.
            await countTokens(req, conn)
        case ("GET", let path) where path.hasPrefix("/v1/models/"):
            // Anthropic's single-model lookup, the other model-validation
            // probe.  Whatever id was asked for, this server serves exactly
            // one model; report it.
            send(conn, json: ["type": "model", "id": modelName,
                              "display_name": modelName,
                              "created_at": "2026-01-01T00:00:00Z",
                              "tempo9": APICapabilities.modelHint])
        // --- Ollama native API ---------------------------------------
        // Apps that hardcode Ollama reach for these rather than /v1.  The
        // wire differences that bite are documented in OllamaAPI.swift.
        case ("GET", "/api/tags"):
            send(conn, json: OllamaAPI.tags(model: modelName,
                                            info: modelInfo))
        case ("GET", "/api/version"):
            send(conn, json: OllamaAPI.version(ollamaVersion))
        case ("GET", "/api/ps"):
            send(conn, json: OllamaAPI.ps(model: modelName, info: modelInfo))
        case ("POST", "/api/show"):
            // `capabilities` decides whether the client offers tool calling
            // and image attachment at all, so it reports what this build can
            // do -- vision only when a tower is actually loaded.
            send(conn, json: OllamaAPI.show(model: modelName,
                                            info: modelInfo,
                                            vision: encodeImage != nil))
        case ("POST", "/api/chat"):
            await ollamaChat(req, conn, generate: false)
        case ("POST", "/api/generate"):
            await ollamaChat(req, conn, generate: true)
        case ("POST", "/api/embed"), ("POST", "/api/embeddings"):
            // Refuse in the caller's own dialect rather than 404ing: a client
            // that asked for embeddings should learn this server has none,
            // not that the route is absent and perhaps the URL is wrong.
            send(conn, status: "501 Not Implemented",
                 json: ["error": "this server does not serve embeddings"])
        default:
            send(conn, status: "404 Not Found",
                 json: ["error": ["message": "unknown route \(req.path)"]])
        }
    }

    // MARK: - image parts

    struct ImagePartError: Error { let message: String }

    /// The one image in the request, decoded and encoded, or nil.
    ///
    /// Deliberately narrow, and each refusal says why rather than saying
    /// "unsupported":
    ///
    /// - `http(s)` URLs are refused. Fetching one would make the offline
    ///   assistant reach the network to answer a question, and leak to the
    ///   host that this machine is running and what it was asked to look at.
    ///   The same argument that keeps CloudPrice hard-coded applies here,
    ///   with a much bigger payload.
    /// - More than one image is refused. The engine carries a media chain
    ///   now, but the placement API is still one picture, and quietly using
    ///   only the first would answer confidently about the wrong one.
    private func extractImage(_ messages: [[String: Any]]) async throws
        -> ImagePlacement? {
        var urls: [String] = []
        for m in messages {
            guard let parts = m["content"] as? [[String: Any]] else { continue }
            for p in parts where (p["type"] as? String) == "image_url" {
                guard let u = (p["image_url"] as? [String: Any])?["url"]
                        as? String ?? p["image_url"] as? String else {
                    throw ImagePartError(
                        message: "image_url part has no url")
                }
                urls.append(u)
            }
        }
        guard !urls.isEmpty else { return nil }
        guard urls.count == 1 else {
            throw ImagePartError(
                message: "one image per request; this had \(urls.count)")
        }
        guard let encodeImage else {
            throw ImagePartError(
                message: "this model has no vision front end loaded")
        }
        let url = urls[0]
        guard url.hasPrefix("data:") else {
            throw ImagePartError(
                message: "only data: URIs are accepted — fetching a remote "
                       + "image would put this machine on the network to "
                       + "answer a local question. Inline the bytes.")
        }
        guard let comma = url.firstIndex(of: ","),
              url[..<comma].hasSuffix(";base64"),
              let raw = Data(base64Encoded:
                                String(url[url.index(after: comma)...]),
                             options: .ignoreUnknownCharacters) else {
            throw ImagePartError(
                message: "data: URI must be base64 (data:image/png;base64,...)")
        }
        guard let src = CGImageSourceCreateWithData(raw as CFData, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw ImagePartError(message: "could not decode the image bytes")
        }
        return try await encodeImage(img)
    }

    // MARK: - chat

    private func chat(_ req: HTTPRequest, _ conn: NWConnection) async {
        guard let body = try? JSONSerialization.jsonObject(with: req.body)
                as? [String: Any] else {
            send(conn, status: "400 Bad Request",
                 json: ["error": ["message": "messages[] required"]])
            return
        }
        let parsed: APIRequest.Chat
        do {
            parsed = try APIRequest.chat(body, speculationK: defaultSpeculationK)
        } catch let r as RequestRefusal {
            send(conn, status: "400 Bad Request", json: r.openAI)
            return
        } catch {
            send(conn, status: "400 Bad Request",
                 json: ["error": ["message": Self.describe(error)]])
            return
        }
        let rawMessages = parsed.messages
        let warnings = parsed.warnings
        logWarnings(warnings)
        // OpenAI allows content as a string OR an array of parts. Image
        // parts are accepted when a vision front end is loaded, and refused
        // with the actual reason when one is not -- "not supported" was true
        // while this served a text model and became a lie the day it started
        // serving a VLM.
        var placement: ImagePlacement?
        do {
            switch try await extractImage(rawMessages) {
            case .none: break
            case .some(let img): placement = img
            }
        } catch let e as ImagePartError {
            send(conn, status: "400 Bad Request",
                 json: ["error": ["message": e.message]])
            return
        } catch {
            send(conn, status: "500 Internal Server Error",
                 json: ["error": ["message": "image encode failed: \(error)"]])
            return
        }
        let cfg = parsed.config
        let thinking = parsed.thinking
        let stream = parsed.stream
        let tmplTools = parsed.tools
        let id = "chatcmpl-" + UUID().uuidString.prefix(12)
        let created = Int(Date().timeIntervalSince1970)

        func toolCallsPayload(_ calls: [ToolCallBridge.Call])
            -> [[String: Any]] {
            calls.enumerated().map { i, c in
                ["index": i, "id": "call_\(id)_\(i)", "type": "function",
                 "function": ["name": c.name,
                              "arguments": c.argumentsJSON]]
            }
        }

        await gate.run { [self] in
            // Render and tokenize before a byte is on the wire, so a
            // template that throws is a status line, not a 200 that stops
            // (see serveAnthropicTurn).
            let turn: PreparedTurn
            do {
                turn = try await session.prepare(
                    messages: rawMessages, config: cfg,
                    enableThinking: thinking, image: placement,
                    tools: tmplTools.isEmpty ? nil : tmplTools)
            } catch {
                send(conn, status: Self.httpStatus(for: error),
                     json: ["error": ["message": Self.describe(error),
                                      "type": Self.openAIErrorType(for: error)]])
                return
            }
            let allWarnings = warnings + turn.warnings
            do {
                if stream {
                    sendStreamHead(conn)
                    // role delta first — Cursor's parser wants it.
                    sendSSE(conn, chunk(id: String(id), created: created,
                                        delta: ["role": "assistant"],
                                        finish: nil))
                    let filter = ToolCallBridge.StreamFilter()
                    // Incremental tool_calls: announce a call the moment its
                    // name is certain, send its arguments the moment its
                    // closer arrives.  Clients CONCATENATE argument deltas,
                    // so each fragment is sent exactly once -- the two
                    // counters are the contract, and the end-of-stream pass
                    // only tops up from where they stop.
                    let ts = ToolStreamProgress()
                    func announce(_ index: Int, _ name: String) {
                        sendSSE(conn, chunk(
                            id: String(id), created: created,
                            delta: ["tool_calls": [[
                                "index": index, "id": "call_\(id)_\(index)",
                                "type": "function",
                                "function": ["name": name,
                                             "arguments": ""]]]],
                            finish: nil))
                    }
                    func sendArgs(_ index: Int, _ json: String) {
                        sendSSE(conn, chunk(
                            id: String(id), created: created,
                            delta: ["tool_calls": [[
                                "index": index,
                                "function": ["arguments": json]]]],
                            finish: nil))
                    }
                    let reply = try await session.run(turn, onDelta: { d in
                            let out = filter.feed(d.content)
                            if !out.isEmpty {
                                self.sendSSE(conn, self.chunk(
                                    id: String(id), created: created,
                                    delta: ["content": out],
                                    finish: nil))
                            }
                            for name in filter.newCallNames() {
                                announce(ts.announced, name)
                                ts.announced += 1
                            }
                            let done = filter.completedCalls(tools: tmplTools)
                            while ts.argsSent < done.count {
                                let c = done[ts.argsSent]
                                if ts.announced <= ts.argsSent {
                                    announce(ts.argsSent, c.name)
                                    ts.announced = ts.argsSent + 1
                                }
                                sendArgs(ts.argsSent, c.argumentsJSON)
                                ts.argsSent += 1
                            }
                        })
                    let tail = filter.flush()
                    if !tail.isEmpty {
                        sendSSE(conn, chunk(id: String(id), created: created,
                                            delta: ["content": tail],
                                            finish: nil))
                    }
                    let parsed = ToolCallBridge.parse(reply.text, tools: tmplTools)
                    if parsed.calls.count < ts.argsSent {
                        // Streamed more than the final parse can see: a
                        // dialect disagreement.  Nothing can be unsent --
                        // log it loudly rather than double-send.
                        let msg = "[tempo9] tool-call stream/final mismatch: "
                            + "sent \(ts.argsSent), final \(parsed.calls.count)\n"
                        FileHandle.standardError.write(Data(msg.utf8))
                    }
                    for i in ts.argsSent..<max(ts.argsSent, parsed.calls.count) {
                        if ts.announced <= i {
                            announce(i, parsed.calls[i].name)
                            ts.announced = i + 1
                        }
                        sendArgs(i, parsed.calls[i].argumentsJSON)
                        ts.argsSent += 1
                    }
                    // "length" against the max_tokens the turn RAN with:
                    // a reply cut at a clamped 1116 must not read as the
                    // model choosing to stop.
                    sendSSE(conn, chunk(id: String(id), created: created,
                                        delta: [:],
                                        finish: (parsed.calls.isEmpty
                                                 && ts.announced == 0)
                                            ? (reply.completionTokens
                                               >= reply.maxTokens
                                               ? "length" : "stop")
                                            : "tool_calls"))
                    // OpenAI stream_options.include_usage shape: one final
                    // chunk with empty choices and the usage totals.  SSE
                    // chunk counting under-reports whenever a consumer lags
                    // (one chunk can carry several tokens), so a client that
                    // wants token numbers must have them from the server.
                    sendSSE(conn, warnings: allWarnings, [
                        "id": String(id), "object": "chat.completion.chunk",
                        "created": created, "model": modelName,
                        "choices": [] as [Any],
                        "usage": [
                            "prompt_tokens": reply.promptTokens,
                            "completion_tokens": reply.completionTokens,
                            "total_tokens": reply.promptTokens
                                + reply.completionTokens,
                        ],
                    ])
                    sendRaw(conn, Data("data: [DONE]\n\n".utf8), close: true)
                } else {
                    let reply = try await session.run(turn, onDelta: { _ in })
                    let parsed = ToolCallBridge.parse(reply.text, tools: tmplTools)
                    var message: [String: Any] = ["role": "assistant",
                                                  "content": parsed.text]
                    if !parsed.calls.isEmpty {
                        message["tool_calls"] = toolCallsPayload(parsed.calls)
                    }
                    send(conn, warnings: allWarnings, json: [
                        "id": String(id), "object": "chat.completion",
                        "created": created, "model": modelName,
                        "choices": [[
                            "index": 0,
                            "message": message,
                            "finish_reason": parsed.calls.isEmpty
                                ? (reply.completionTokens >= reply.maxTokens
                                   ? "length" : "stop")
                                : "tool_calls",
                        ]],
                        "usage": [
                            "prompt_tokens": reply.promptTokens,
                            "completion_tokens": reply.completionTokens,
                            "total_tokens": reply.promptTokens
                                + reply.completionTokens,
                        ],
                    ])
                }
            } catch {
                if stream {
                    // Say what went wrong. Closing a stream with a bare
                    // [DONE] produces a 200 carrying no content, which every
                    // client reads as "the model had nothing to say" -- an
                    // agent framework then retries the same doomed request
                    // and reports "couldn't generate a response".
                    //
                    // The engine's own diagnosis was precise and discarded
                    // here: `genconfig.max_length (33189) > engine_max_length_
                    // (32768)`. A caller that can see that can shorten the
                    // prompt; a caller that sees an empty stream cannot.
                    let payload: [String: Any] = ["error": [
                        "message": Self.describe(error),
                        "type": Self.openAIErrorType(for: error),
                    ]]
                    if let data = try? JSONSerialization.data(
                        withJSONObject: payload) {
                        sendRaw(conn, Data("data: ".utf8) + data
                                + Data("\n\n".utf8), close: false)
                    }
                    sendRaw(conn, Data("data: [DONE]\n\n".utf8), close: true)
                } else {
                    send(conn, status: Self.httpStatus(for: error),
                         json: ["error":
                            ["message": Self.describe(error),
                             "type": Self.openAIErrorType(for: error)]])
                }
            }
        }
    }

    // MARK: - Anthropic Messages (/v1/messages)
    //
    // The other half of the joke: ANTHROPIC_BASE_URL=http://127.0.0.1:11435
    // points Claude Code (or any Anthropic-SDK client) at this server.  Same
    // engine, same gate; only the wire shape differs — system is top-level,
    // content is blocks, streaming is typed SSE events instead of chunks.

    private func anthropicMessages(_ req: HTTPRequest,
                                   _ conn: NWConnection) async {
        guard let body = try? JSONSerialization.jsonObject(with: req.body)
                as? [String: Any] else {
            send(conn, status: "400 Bad Request", json: [
                "type": "error",
                "error": ["type": "invalid_request_error",
                          "message": "messages[] required"]])
            return
        }
        let parsed: APIRequest.Anthropic
        do {
            parsed = try APIRequest.anthropic(body,
                                              speculationK: defaultSpeculationK)
        } catch let r as RequestRefusal {
            send(conn, status: "400 Bad Request", json: r.anthropic)
            return
        } catch {
            send(conn, status: "400 Bad Request", json: [
                "type": "error",
                "error": ["type": "invalid_request_error",
                          "message": Self.describe(error)]])
            return
        }
        let messages = parsed.messages
        let tmplTools = parsed.tools
        let cfg = parsed.config
        let thinking = parsed.thinking
        let stream = parsed.stream
        let warnings = parsed.warnings
        logWarnings(warnings)
        let id = "msg_" + UUID().uuidString.prefix(12)

        await gate.run { [self] in
            await Self.serveAnthropicTurn(
                parsed, id: String(id), modelName: modelName, out: out(conn),
                prepare: {
                    let turn = try await session.prepare(
                        messages: messages, config: cfg,
                        enableThinking: thinking,
                        tools: tmplTools.isEmpty ? nil : tmplTools)
                    return (turn, turn.warnings)
                },
                generate: { turn, onDelta in
                    try await session.run(turn, onDelta: onDelta)
                })
        }
    }

    // MARK: - OpenAI Responses (/v1/responses)

    private func responses(_ req: HTTPRequest, _ conn: NWConnection) async {
        guard let body = try? JSONSerialization.jsonObject(with: req.body)
                as? [String: Any] else {
            send(conn, status: "400 Bad Request",
                 json: ["error": ["message": "bad json"]])
            return
        }
        let parsed: APIRequest.Responses
        do {
            parsed = try APIRequest.responses(body,
                                              speculationK: defaultSpeculationK)
        } catch let r as RequestRefusal {
            send(conn, status: "400 Bad Request", json: r.openAI)
            return
        } catch {
            send(conn, status: "400 Bad Request",
                 json: ["error": ["message": Self.describe(error)]])
            return
        }
        let messages = parsed.messages
        let tmplTools = parsed.tools
        let cfg = parsed.config
        let stream = parsed.stream
        let warnings = parsed.warnings
        logWarnings(warnings)
        let id = "resp_" + UUID().uuidString.prefix(12)
        let itemId = "msg_" + UUID().uuidString.prefix(12)

        func responseObject(_ status: String, text: String?,
                            calls: [ToolCallBridge.Call] = [],
                            usage: (Int, Int)?) -> [String: Any] {
            var out: [[String: Any]] = []
            if let text, !text.isEmpty {
                out = [["type": "message", "id": String(itemId),
                        "status": "completed", "role": "assistant",
                        "content": [["type": "output_text", "text": text,
                                     "annotations": []]]]]
            }
            for (i, c) in calls.enumerated() {
                out.append(["type": "function_call",
                            "id": "fc_\(itemId)_\(i)",
                            "call_id": "call_\(itemId)_\(i)",
                            "name": c.name,
                            "arguments": c.argumentsJSON,
                            "status": "completed"])
            }
            var r: [String: Any] = [
                "id": String(id), "object": "response", "status": status,
                "model": modelName, "output": out,
                "created_at": Int(Date().timeIntervalSince1970),
            ]
            if let usage {
                r["usage"] = ["input_tokens": usage.0,
                              "output_tokens": usage.1,
                              "total_tokens": usage.0 + usage.1]
            }
            return Self.withWarnings(r, warnings)
        }

        await gate.run { [self] in
            // Render and tokenize before a byte is on the wire (see
            // serveAnthropicTurn).
            let turn: PreparedTurn
            do {
                turn = try await session.prepare(
                    messages: messages, config: cfg, enableThinking: false,
                    tools: tmplTools.isEmpty ? nil : tmplTools)
            } catch {
                send(conn, status: Self.httpStatus(for: error),
                     json: ["error": ["message": Self.describe(error),
                                      "type": Self.openAIErrorType(for: error)]])
                return
            }
            do {
                if stream {
                    sendStreamHead(conn)
                    sendEvent(conn, "response.created", [
                        "type": "response.created",
                        "response": responseObject("in_progress", text: nil,
                                                   usage: nil)])
                    sendEvent(conn, "response.output_item.added", [
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": ["type": "message", "id": String(itemId),
                                 "status": "in_progress",
                                 "role": "assistant", "content": []]])
                    sendEvent(conn, "response.content_part.added", [
                        "type": "response.content_part.added",
                        "item_id": String(itemId), "output_index": 0,
                        "content_index": 0,
                        "part": ["type": "output_text", "text": "",
                                 "annotations": []]])
                    let filter = ToolCallBridge.StreamFilter()
                    // Incremental function_call items: the canonical ladder
                    // (added empty/in_progress -> arguments.delta ->
                    // arguments.done -> item done), unchanged in shape --
                    // codex's state machine went catatonic on a shortcut
                    // once -- but each rung now fires as soon as the stream
                    // certifies it instead of after the whole reply.
                    let driver = ToolCallBridge.BlockToolDriver(filter: filter)
                    let st = BlockStreamState()
                    var names: [Int: String] = [:]
                    func closeTextIfOpen() {
                        guard st.textOpen else { return }
                        st.textOpen = false
                        let text = st.textAcc
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        sendEvent(conn, "response.output_text.done", [
                            "type": "response.output_text.done",
                            "item_id": String(itemId), "output_index": 0,
                            "content_index": 0, "text": text])
                        sendEvent(conn, "response.output_item.done", [
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": ["type": "message", "id": String(itemId),
                                     "status": "completed",
                                     "role": "assistant",
                                     "content": [["type": "output_text",
                                                  "text": text,
                                                  "annotations": []]]]])
                    }
                    func openItem(_ i: Int, _ name: String) {
                        closeTextIfOpen()
                        names[i] = name
                        sendEvent(conn, "response.output_item.added", [
                            "type": "response.output_item.added",
                            "output_index": 1 + i,
                            "item": ["type": "function_call",
                                     "id": "fc_\(itemId)_\(i)",
                                     "call_id": "call_\(itemId)_\(i)",
                                     "name": name, "arguments": "",
                                     "status": "in_progress"]])
                    }
                    func argsItem(_ i: Int, _ json: String) {
                        let fcId = "fc_\(itemId)_\(i)"
                        sendEvent(conn, "response.function_call_arguments.delta", [
                            "type": "response.function_call_arguments.delta",
                            "item_id": fcId, "output_index": 1 + i,
                            "delta": json])
                        sendEvent(conn, "response.function_call_arguments.done", [
                            "type": "response.function_call_arguments.done",
                            "item_id": fcId, "output_index": 1 + i,
                            "arguments": json])
                        sendEvent(conn, "response.output_item.done", [
                            "type": "response.output_item.done",
                            "output_index": 1 + i,
                            "item": ["type": "function_call", "id": fcId,
                                     "call_id": "call_\(itemId)_\(i)",
                                     "name": names[i] ?? "",
                                     "arguments": json,
                                     "status": "completed"]])
                    }
                    let reply = try await session.run(turn, onDelta: { d in
                            let out = filter.feed(d.content)
                            if !out.isEmpty {
                                st.textAcc += out
                                self.sendEvent(conn, "response.output_text.delta", [
                                    "type": "response.output_text.delta",
                                    "item_id": String(itemId),
                                    "output_index": 0, "content_index": 0,
                                    "delta": out])
                            }
                            driver.pump(tools: tmplTools,
                                        open: openItem, args: argsItem)
                        })
                    let tail = filter.flush()
                    if !tail.isEmpty, st.textOpen {
                        st.textAcc += tail
                        sendEvent(conn, "response.output_text.delta", [
                            "type": "response.output_text.delta",
                            "item_id": String(itemId),
                            "output_index": 0, "content_index": 0,
                            "delta": tail])
                    }
                    let parsed = ToolCallBridge.parse(reply.text, tools: tmplTools)
                    let counts = driver.finish(parsed: parsed.calls,
                                               open: openItem,
                                               args: argsItem)
                    if counts.sent > counts.final {
                        let msg = "[tempo9] responses tool stream/final "
                            + "mismatch: sent \(counts.sent), final \(counts.final)\n"
                        FileHandle.standardError.write(Data(msg.utf8))
                    }
                    closeTextIfOpen()
                    sendEvent(conn, "response.completed", [
                        "type": "response.completed",
                        "response": Self.withWarnings(responseObject(
                            "completed", text: parsed.text,
                            calls: parsed.calls,
                            usage: (reply.promptTokens,
                                    reply.completionTokens)),
                            turn.warnings)])
                    sendRaw(conn, Data(), close: true)
                } else {
                    let reply = try await session.run(turn, onDelta: { _ in })
                    let parsed = ToolCallBridge.parse(reply.text, tools: tmplTools)
                    send(conn, json: Self.withWarnings(responseObject(
                        "completed", text: parsed.text, calls: parsed.calls,
                        usage: (reply.promptTokens, reply.completionTokens)),
                        turn.warnings))
                }
            } catch {
                if stream {
                    // Headers are out: the Responses stream's own `error`
                    // event, then close -- not a bare close.
                    sendEvent(conn, "error", [
                        "type": "error",
                        "code": Self.openAIErrorType(for: error),
                        "message": Self.describe(error)])
                    sendRaw(conn, Data(), close: true)
                } else {
                    send(conn, status: Self.httpStatus(for: error),
                         json: ["error":
                            ["message": Self.describe(error),
                             "type": Self.openAIErrorType(for: error)]])
                }
            }
        }
    }

    private func countTokens(_ req: HTTPRequest, _ conn: NWConnection) async {
        guard let body = try? JSONSerialization.jsonObject(with: req.body)
                as? [String: Any] else {
            send(conn, status: "400 Bad Request", json: [
                "type": "error",
                "error": ["type": "invalid_request_error",
                          "message": "bad json"]])
            return
        }
        do {
            let n = try await APIRequest.countTokens(body) {
                messages, tools, thinking in
                try await self.session.countTokens(messages: messages,
                                                   tools: tools,
                                                   enableThinking: thinking)
            }
            send(conn, json: ["input_tokens": n])
        } catch let r as RequestRefusal {
            send(conn, status: "400 Bad Request", json: r.anthropic)
        } catch {
            send(conn, status: "500 Internal Server Error", json: [
                "type": "error",
                "error": ["type": "api_error",
                          "message": Self.describe(error)]])
        }
    }

    // MARK: - the Anthropic turn, as a function of its outputs

    /// HTTP status for an error that reached a route after parsing.
    ///
    /// A prompt that does not fit is the caller's to fix (shorten it, or
    /// start the server with a larger --max-length), so it is a 400 like
    /// any other refusal; everything else is the server's.
    static func httpStatus(for error: Error) -> String {
        if error is RequestRefusal { return "400 Bad Request" }
        if case LocalSessionError.promptTooLong = error {
            return "400 Bad Request"
        }
        return "500 Internal Server Error"
    }

    /// Anthropic's error `type` for the same error.
    static func anthropicErrorType(for error: Error) -> String {
        httpStatus(for: error).hasPrefix("4") ? "invalid_request_error"
                                              : "api_error"
    }

    /// OpenAI's error `type` for the same error.
    static func openAIErrorType(for error: Error) -> String {
        httpStatus(for: error).hasPrefix("4") ? "invalid_request_error"
                                              : "server_error"
    }

    /// One /v1/messages turn, from a parsed request to bytes on the wire.
    ///
    /// Static and closure-driven so it can run without an engine: `prepare`
    /// is the host-side half of the turn (render, tokenize -- everything
    /// that can fail before generation), `generate` the engine half.  The
    /// route passes LocalSession's two halves; a test passes closures that
    /// throw where it wants and reads `out`.
    static func serveAnthropicTurn<Turn>(
        _ parsed: APIRequest.Anthropic, id: String, modelName: String,
        out: HTTPOut,
        prepare: () async throws -> (Turn, [String]),
        generate: (Turn, @escaping (LocalDelta) -> Void) async throws
            -> LocalReply) async {
        let tmplTools = parsed.tools
        let stream = parsed.stream
        // Everything that can fail before generation -- the template
        // render above all -- happens BEFORE a byte is on the wire, so a
        // failure is an ordinary status line with a body.  It used to
        // happen after `200` and `message_start`: Claude Code got 339 bytes
        // and a closed socket for a template that threw on its second
        // message, and nothing to show why.
        let turn: Turn
        let turnWarnings: [String]
        do {
            (turn, turnWarnings) = try await prepare()
        } catch {
            out.json(status: httpStatus(for: error), [
                "type": "error",
                "error": ["type": anthropicErrorType(for: error),
                          "message": describe(error)]])
            return
        }
        let warnings = parsed.warnings + turnWarnings
        var started = false
        do {
            if stream {
                out.streamHead()
                started = true
                out.event("message_start", warnings: warnings, [
                    "type": "message_start",
                    "message": [
                        "id": id, "type": "message",
                        "role": "assistant", "model": modelName,
                        "content": [], "stop_reason": NSNull(),
                        "usage": ["input_tokens": 0, "output_tokens": 0],
                    ]])
                out.event("content_block_start", [
                    "type": "content_block_start", "index": 0,
                    "content_block": ["type": "text", "text": ""]])
                let filter = ToolCallBridge.StreamFilter()
                // Incremental tool_use blocks: same ladder, earlier in
                // time.  The text block (index 0) closes the moment the
                // first call opens -- once the filter is suppressing, no
                // text can follow.  BlockToolDriver enforces the block
                // protocol's ordering (k closes before k+1 opens).
                let driver = ToolCallBridge.BlockToolDriver(filter: filter)
                let st = BlockStreamState()
                func closeTextIfOpen() {
                    guard st.textOpen else { return }
                    st.textOpen = false
                    out.event("content_block_stop",
                              ["type": "content_block_stop", "index": 0])
                }
                func openBlock(_ i: Int, _ name: String) {
                    closeTextIfOpen()
                    out.event("content_block_start", [
                        "type": "content_block_start", "index": 1 + i,
                        "content_block": ["type": "tool_use",
                                          "id": "toolu_\(id)_\(i)",
                                          "name": name, "input": [:]]])
                }
                func argsBlock(_ i: Int, _ json: String) {
                    out.event("content_block_delta", [
                        "type": "content_block_delta", "index": 1 + i,
                        "delta": ["type": "input_json_delta",
                                  "partial_json": json]])
                    out.event("content_block_stop", [
                        "type": "content_block_stop", "index": 1 + i])
                }
                let reply = try await generate(turn) { d in
                    let text = filter.feed(d.content)
                    if !text.isEmpty {
                        out.event("content_block_delta", [
                            "type": "content_block_delta", "index": 0,
                            "delta": ["type": "text_delta", "text": text]])
                    }
                    driver.pump(tools: tmplTools,
                                open: openBlock, args: argsBlock)
                }
                let tail = filter.flush()
                if !tail.isEmpty, st.textOpen {
                    out.event("content_block_delta", [
                        "type": "content_block_delta", "index": 0,
                        "delta": ["type": "text_delta", "text": tail]])
                }
                let parsedReply = ToolCallBridge.parse(reply.text,
                                                       tools: tmplTools)
                let counts = driver.finish(parsed: parsedReply.calls,
                                           open: openBlock, args: argsBlock)
                if counts.sent > counts.final {
                    let msg = "[tempo9] anthropic tool stream/final "
                        + "mismatch: sent \(counts.sent), final \(counts.final)\n"
                    FileHandle.standardError.write(Data(msg.utf8))
                }
                closeTextIfOpen()
                let stop = !parsedReply.calls.isEmpty ? "tool_use"
                    : reply.completionTokens >= reply.maxTokens
                        ? "max_tokens" : "end_turn"
                out.event("message_delta", [
                    "type": "message_delta",
                    "delta": ["stop_reason": stop],
                    "usage": ["output_tokens": reply.completionTokens]])
                out.event("message_stop", ["type": "message_stop"])
                out.raw(Data(), close: true)
            } else {
                let reply = try await generate(turn) { _ in }
                let parsedReply = ToolCallBridge.parse(reply.text,
                                                       tools: tmplTools)
                var content: [[String: Any]] = []
                if !reply.reasoning.isEmpty {
                    content.append(["type": "thinking",
                                    "thinking": reply.reasoning])
                }
                if !parsedReply.text.isEmpty {
                    content.append(["type": "text", "text": parsedReply.text])
                }
                for (i, c) in parsedReply.calls.enumerated() {
                    let input = (try? JSONSerialization.jsonObject(
                        with: Data(c.argumentsJSON.utf8))) ?? [:]
                    content.append(["type": "tool_use",
                                    "id": "toolu_\(id)_\(i)",
                                    "name": c.name, "input": input])
                }
                let stop = !parsedReply.calls.isEmpty ? "tool_use"
                    : reply.completionTokens >= reply.maxTokens
                        ? "max_tokens" : "end_turn"
                out.json(warnings: warnings, [
                    "id": id, "type": "message",
                    "role": "assistant", "model": modelName,
                    "content": content,
                    "stop_reason": stop,
                    "usage": [
                        "input_tokens": reply.promptTokens,
                        "output_tokens": reply.completionTokens,
                    ],
                ])
            }
        } catch {
            if started {
                // Headers are out; the only channel left is the stream
                // itself.  Anthropic's SSE has an `error` event for exactly
                // this, and the SDKs raise on it -- a closed socket with no
                // event reads as "the model had nothing to say".
                out.event("error", [
                    "type": "error",
                    "error": ["type": anthropicErrorType(for: error),
                              "message": describe(error)]])
                out.raw(Data(), close: true)
            } else {
                out.json(status: httpStatus(for: error), [
                    "type": "error",
                    "error": ["type": anthropicErrorType(for: error),
                              "message": describe(error)]])
            }
        }
    }

    /// The socket as an HTTPOut.
    private func out(_ conn: NWConnection) -> HTTPOut {
        HTTPOut { [self] data, close in sendRaw(conn, data, close: close) }
    }

    private func sendEvent(_ conn: NWConnection, _ event: String,
                           warnings: [String] = [],
                           _ obj: [String: Any]) {
        let obj = Self.withWarnings(obj, warnings)
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let line = String(data: data, encoding: .utf8) else { return }
        sendRaw(conn, Data("event: \(event)\ndata: \(line)\n\n".utf8),
                close: false)
    }

    private func chunk(id: String, created: Int, delta: [String: Any],
                       finish: String?) -> [String: Any] {
        [
            "id": id, "object": "chat.completion.chunk",
            "created": created, "model": modelName,
            "choices": [[
                "index": 0, "delta": delta,
                "finish_reason": finish as Any,
            ]],
        ]
    }

    // MARK: - Ollama native API

    /// POST /api/chat and POST /api/generate.
    ///
    /// The same engine call `chat` makes, in different clothes: NDJSON lines
    /// instead of SSE, `stream` defaulting to true, and tool arguments as
    /// objects.  The translation is in OllamaAPI so this stays plumbing.
    private func ollamaChat(_ req: HTTPRequest, _ conn: NWConnection,
                            generate: Bool) async {
        guard let body = try? JSONSerialization.jsonObject(with: req.body)
                as? [String: Any] else {
            send(conn, status: "400 Bad Request",
                 json: ["error": "invalid JSON body"])
            return
        }
        let r = generate ? OllamaAPI.generateRequest(from: body)
                         : OllamaAPI.chatRequest(from: body)
        guard !r.messages.isEmpty else {
            send(conn, status: "400 Bad Request",
                 json: ["error": generate ? "prompt required"
                                          : "messages[] required"])
            return
        }

        // chatRequest rewrote Ollama's bare-base64 `images` into the typed
        // parts extractImage understands, so vision reaches the same code
        // path and the same refusals explain themselves the same way.
        var placement: ImagePlacement?
        do {
            placement = try await extractImage(r.messages)
        } catch let e as ImagePartError {
            send(conn, status: "400 Bad Request", json: ["error": e.message])
            return
        } catch {
            send(conn, status: "500 Internal Server Error",
                 json: ["error": "image encode failed: \(error)"])
            return
        }

        let cfg = Self.ollamaSamplingConfig(r, speculationK: defaultSpeculationK)
        let tmplTools: [[String: Any]]
        do {
            tmplTools = try ToolCallBridge.fromChat(r.tools).tools
        } catch let e as RequestRefusal {
            send(conn, status: "400 Bad Request", json: ["error": e.message])
            return
        } catch {
            send(conn, status: "400 Bad Request",
                 json: ["error": Self.describe(error)])
            return
        }
        let name = modelName
        let sink = out(conn)

        await gate.run { [self] in
            await Self.serveOllamaTurn(
                r, modelName: name, generate: generate, tools: tmplTools,
                out: sink,
                prepare: {
                    let turn = try await session.prepare(
                        messages: r.messages, config: cfg,
                        enableThinking: r.think, image: placement,
                        tools: tmplTools.isEmpty ? nil : tmplTools)
                    return (turn, turn.warnings)
                },
                run: { turn, onDelta in
                    try await session.run(turn, onDelta: onDelta)
                })
        }
    }

    /// Ollama's `options` onto the engine's sampling config.  `num_predict`
    /// is Ollama's max_tokens and lands in `maxTokens`, so the same context
    /// budget that clamps max_tokens on the other protocols sees it.
    static func ollamaSamplingConfig(_ r: OllamaAPI.ChatRequest,
                                     speculationK: Int32) -> SamplingConfig {
        var cfg = SamplingConfig()
        cfg.speculationK = speculationK
        cfg.maxTokens = r.maxTokens ?? 2048
        if let t = r.temperature {
            cfg.temperature = Float(t)
            cfg.doSample = t > 0
            if t == 0 { cfg.topK = 1 }
        }
        if let p = r.topP { cfg.topP = Float(p) }
        if let k = r.topK { cfg.topK = numericCast(k) }
        return cfg
    }

    /// One /api/chat or /api/generate turn, from a parsed request to bytes
    /// on the wire -- the Ollama twin of `serveAnthropicTurn`, driven by
    /// the same two closures so it runs without an engine.
    ///
    /// Ollama's shape for a request it rejects is a status code and
    /// `{"error": "<string>"}` (server/routes.go: `c.JSON(http.StatusBadRequest,
    /// gin.H{"error": err.Error()})`); mid-stream, the same object as one
    /// more NDJSON line.  So a prompt that alone overflows the engine is a
    /// 400 with that body before any byte of a stream is out -- Ollama
    /// itself would silently truncate the prompt (`truncate` defaults to
    /// true there); this server refuses instead, with the numbers.  A
    /// clamped num_predict rides on the terminal object as a top-level
    /// `warnings` array (Ollama's response has no field for it, and
    /// its clients ignore fields they do not know) and as
    /// `done_reason: "length"` when the reply ran into the clamp.
    static func serveOllamaTurn<Turn>(
        _ r: OllamaAPI.ChatRequest, modelName name: String, generate: Bool,
        tools tmplTools: [[String: Any]], out: HTTPOut,
        prepare: () async throws -> (Turn, [String]),
        run: (Turn, @escaping (LocalDelta) -> Void) async throws
            -> LocalReply) async {
        // Everything that can fail before generation happens BEFORE a byte
        // is on the wire (see serveAnthropicTurn): the too-long prompt is
        // an ordinary status line, streaming or not.
        let turn: Turn
        let warnings: [String]
        do {
            (turn, warnings) = try await prepare()
        } catch {
            out.json(status: httpStatus(for: error),
                     ["error": describe(error)])
            return
        }
        func doneReason(_ reply: LocalReply) -> String {
            // Judged against the max_tokens the turn RAN with (the clamped
            // one), as chat's finish_reason and Anthropic's stop_reason are.
            reply.completionTokens >= reply.maxTokens ? "length" : "stop"
        }
        var started = false
        do {
            if r.stream {
                out.ndjsonHead()
                started = true
                let filter = ToolCallBridge.StreamFilter()
                let reply = try await run(turn) { d in
                    let text = filter.feed(d.content)
                    if !text.isEmpty {
                        out.ndjson(OllamaAPI.chunk(
                            model: name, content: text, generate: generate))
                    }
                }
                let tail = filter.flush()
                if !tail.isEmpty {
                    out.ndjson(OllamaAPI.chunk(
                        model: name, content: tail, generate: generate))
                }
                let parsed = ToolCallBridge.parse(reply.text,
                                                  tools: tmplTools)
                let calls = OllamaAPI.toolCalls(
                    parsed.calls.map { ($0.name, $0.argumentsJSON) })
                // Content is empty here on purpose: it already went out
                // as chunks, and repeating it in the terminal object
                // makes clients that append render the reply twice.
                out.ndjson(OllamaAPI.done(
                    model: name, content: "", calls: calls,
                    promptTokens: reply.promptTokens,
                    completionTokens: reply.completionTokens,
                    seconds: reply.seconds, generate: generate,
                    reason: doneReason(reply)),
                    warnings: warnings, close: true)
            } else {
                let reply = try await run(turn) { _ in }
                let parsed = ToolCallBridge.parse(reply.text,
                                                  tools: tmplTools)
                let calls = OllamaAPI.toolCalls(
                    parsed.calls.map { ($0.name, $0.argumentsJSON) })
                out.json(warnings: warnings, OllamaAPI.single(
                    model: name, content: parsed.text, calls: calls,
                    promptTokens: reply.promptTokens,
                    completionTokens: reply.completionTokens,
                    seconds: reply.seconds, generate: generate,
                    reason: doneReason(reply)))
            }
        } catch {
            // Same reasoning as the SSE path: a stream that just stops
            // reads as "the model had nothing to say", and the caller
            // retries the same doomed request instead of shortening it.
            let msg = describe(error)
            if started {
                out.ndjson(["error": msg], close: true)
            } else {
                out.json(status: httpStatus(for: error), ["error": msg])
            }
        }
    }

    // MARK: - plumbing

    private func send(_ conn: NWConnection, status: String = "200 OK",
                      warnings: [String] = [],
                      json: [String: Any]) {
        let json = Self.withWarnings(json, warnings)
        let body = (try? JSONSerialization.data(withJSONObject: json)) ?? Data()
        var head = "HTTP/1.1 \(status)\r\n"
        head += "Content-Type: application/json\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(body)
        sendRaw(conn, out, close: true)
    }

    private func sendStreamHead(_ conn: NWConnection) {
        let head = "HTTP/1.1 200 OK\r\n"
            + "Content-Type: text/event-stream\r\n"
            + "Cache-Control: no-cache\r\n"
            + "Connection: close\r\n\r\n"
        sendRaw(conn, Data(head.utf8), close: false)
    }

    private func sendSSE(_ conn: NWConnection, warnings: [String] = [],
                         _ obj: [String: Any]) {
        let obj = Self.withWarnings(obj, warnings)
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let line = String(data: data, encoding: .utf8) else { return }
        sendRaw(conn, Data("data: \(line)\n\n".utf8), close: false)
    }

    /// A client that walks away must take its generation with it.
    ///
    /// Once the request is fully read nothing legitimate arrives on this
    /// connection again (one request per connection, Connection: close), so
    /// a further receive completing — EOF or error — means the client is
    /// gone: cancel the handler task.  The cancellation unwinds through
    /// LocalSession.stream's for-await into Engine.generate's onTermination
    /// and reaches as_request_stop within one 100 ms wait tick.
    ///
    /// Without this, an aborted stream decoded to max_tokens against a dead
    /// socket (sendRaw discarded the send errors), and — the gate being
    /// serial — every request behind it queued until the ghost finished.
    /// An agent client that times out and retries queues its retry behind
    /// the very request it abandoned.
    private func watchDisconnect(_ conn: NWConnection,
                                 cancelling work: Task<Void, Never>) {
        conn.receive(minimumIncompleteLength: 1,
                     maximumLength: 4096) { [weak self] _, _, done, err in
            if err != nil || done {
                work.cancel()
                conn.cancel()
            } else {
                // Stray bytes from a nonconforming client: ignore, rearm.
                self?.watchDisconnect(conn, cancelling: work)
            }
        }
    }

    private func sendRaw(_ conn: NWConnection, _ data: Data, close: Bool) {
        conn.send(content: data, completion: .contentProcessed { err in
            // A send error means the peer is gone; cancelling the connection
            // fires the disconnect watcher above, which stops the request's
            // generation.  Swallowing the error here was how an aborted SSE
            // kept the engine decoding for nobody.
            if err != nil || close { conn.cancel() }
        })
    }
}

/// Where a response goes: the socket in production, a byte buffer in a
/// test.  One write primitive, the three HTTP shapes the routes emit built
/// on it -- the same bytes `send`/`sendEvent`/`sendStreamHead` produce.
struct HTTPOut {
    let write: (Data, _ close: Bool) -> Void

    func raw(_ data: Data, close: Bool) { write(data, close) }

    func json(status: String = "200 OK", warnings: [String] = [],
              _ obj: [String: Any]) {
        var o = obj
        if !warnings.isEmpty { o["warnings"] = warnings }
        let body = (try? JSONSerialization.data(withJSONObject: o)) ?? Data()
        var head = "HTTP/1.1 \(status)\r\n"
        head += "Content-Type: application/json\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        write(Data(head.utf8) + body, true)
    }

    func streamHead() {
        write(Data(("HTTP/1.1 200 OK\r\n"
                    + "Content-Type: text/event-stream\r\n"
                    + "Cache-Control: no-cache\r\n"
                    + "Connection: close\r\n\r\n").utf8), false)
    }

    func event(_ name: String, warnings: [String] = [],
               _ obj: [String: Any]) {
        var o = obj
        if !warnings.isEmpty { o["warnings"] = warnings }
        guard let data = try? JSONSerialization.data(withJSONObject: o),
              let line = String(data: data, encoding: .utf8) else { return }
        write(Data("event: \(name)\ndata: \(line)\n\n".utf8), false)
    }

    /// Ollama streams newline-delimited JSON, not SSE: no "data: " prefix,
    /// no [DONE] sentinel, and a content type clients do check.
    func ndjsonHead() {
        write(Data(("HTTP/1.1 200 OK\r\n"
                    + "Content-Type: application/x-ndjson\r\n"
                    + "Cache-Control: no-cache\r\n"
                    + "Connection: close\r\n\r\n").utf8), false)
    }

    func ndjson(_ obj: [String: Any], warnings: [String] = [],
                close: Bool = false) {
        var o = obj
        if !warnings.isEmpty { o["warnings"] = warnings }
        guard let data = try? JSONSerialization.data(withJSONObject: o),
              let line = String(data: data, encoding: .utf8) else { return }
        write(Data("\(line)\n".utf8), close)
    }
}

/// Mutable progress a streaming handler shares with its onDelta closure.
private final class ToolStreamProgress: @unchecked Sendable {
    var announced = 0
    var argsSent = 0
}

/// Block-protocol streaming state (Anthropic / Responses): is the text
/// block/item still open.
private final class BlockStreamState: @unchecked Sendable {
    var textOpen = true
    var textAcc = ""
}

/// One-at-a-time async gate; FIFO by actor mailbox order.
private actor SerialGate {
    func run(_ op: @Sendable () async -> Void) async { await op() }
}

/// The two request shapes this server accepts, parsed just far enough.
private struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    /// nil until the buffer holds a complete request (headers + full body).
    init?(complete buf: Data) {
        guard let split = buf.range(of: Data("\r\n\r\n".utf8)) else {
            return nil
        }
        let headData = buf[..<split.lowerBound]
        guard let head = String(data: headData, encoding: .utf8) else {
            return nil
        }
        var lines = head.components(separatedBy: "\r\n")
        let start = lines.removeFirst().components(separatedBy: " ")
        guard start.count >= 2 else { return nil }
        var hdrs: [String: String] = [:]
        for line in lines {
            guard let c = line.firstIndex(of: ":") else { continue }
            hdrs[line[..<c].lowercased()] = line[line.index(after: c)...]
                .trimmingCharacters(in: .whitespaces)
        }
        let want = Int(hdrs["content-length"] ?? "0") ?? 0
        let have = buf.distance(from: split.upperBound, to: buf.endIndex)
        guard have >= want else { return nil }
        method = start[0]
        // Strip the query string: the Anthropic TS SDK requests
        // /v1/messages?beta=true, and an exact-match router that has never
        // seen a query 404s it — which a client UI reports as "model does
        // not exist".
        path = String(start[1].prefix(while: { $0 != "?" }))
        headers = hdrs
        body = Data(buf[split.upperBound...].prefix(want))
    }
}
