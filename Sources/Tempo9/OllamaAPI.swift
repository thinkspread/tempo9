// Copyright (c) 2026 Jiejing Zhang.
//
// Ollama's native REST API, translated onto the handlers we already have.
//
// WHY this exists: apps integrate local inference one of two ways.  Most
// offer a "custom OpenAI-compatible endpoint" field, and those already work
// against /v1 with no code from us.  The rest hardcode Ollama's own routes.
// This file buys that second group, and it buys them without asking anyone
// to write a line of code against us -- which is more leverage than shipping
// a client library they would have to adopt.
//
// Everything here is a pure value transform: JSON in, JSON out, no server
// state.  The plumbing lives in OpenAIServer.swift.  Keeping the split means
// the wire format is testable without a socket or a loaded model.
//
// Three differences from OpenAI that are easy to get wrong, and each one
// silently breaks a client rather than erroring:
//
//   1. Streaming is NEWLINE-DELIMITED JSON, not SSE.  No "data: " prefix, no
//      [DONE] sentinel; the last object carries done:true.
//   2. `stream` defaults to TRUE here.  OpenAI defaults to false.  Reading
//      the flag with `?? false` yields a single JSON body where the client
//      is parsing lines, and it hangs waiting for a stream that ended.
//   3. Tool-call arguments are a JSON OBJECT, not a JSON string.  OpenAI
//      sends "{\"a\":1}"; Ollama sends {"a":1}.  A client decoding into a
//      typed struct fails outright on the wrong one.

import Foundation
import CryptoKit

/// What the host knows about the loaded model that the engine does not.
///
/// Injected rather than discovered, for the same reason `encodeImage` is:
/// the server has no idea where its weights came from, and inventing
/// plausible values would be worse than admitting the gap.  Anything left
/// nil is reported as unknown rather than guessed.
public struct Tempo9ModelInfo: Sendable {
    public var sizeBytes: Int64?
    public var quantization: String?
    public var family: String?
    public var parameterSize: String?

    public init(sizeBytes: Int64? = nil, quantization: String? = nil,
                family: String? = nil, parameterSize: String? = nil) {
        self.sizeBytes = sizeBytes
        self.quantization = quantization
        self.family = family
        self.parameterSize = parameterSize
    }
}

public enum OllamaAPI {

    // MARK: - time and identity

    /// RFC3339 with nanoseconds, which is what Ollama emits and what a few
    /// clients parse strictly enough to care.
    public static func timestamp(_ date: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        let ns = Int((date.timeIntervalSince1970.truncatingRemainder(
            dividingBy: 1)) * 1_000_000_000)
        return f.string(from: date) + String(format: ".%09dZ", ns)
    }

    /// A stable opaque identifier for a model entry.
    ///
    /// NOT a content hash.  Ollama's digest is the sha256 of the GGUF blob;
    /// we do not hash gigabytes to answer a listing call.  Clients use this
    /// field as a cache key, and an empty string breaks the ones that do, so
    /// this is a deterministic id over (name, size) instead -- stable across
    /// restarts, different whenever the file changes size.  Do not present
    /// it to a user as a checksum.
    public static func syntheticDigest(name: String, size: Int64) -> String {
        let seed = Data("\(name):\(size)".utf8)
        let hex = SHA256.hash(data: seed).map { String(format: "%02x", $0) }
        return "sha256:" + hex.joined()
    }

    // MARK: - discovery

    public static func details(_ name: String,
                               _ info: Tempo9ModelInfo?) -> [String: Any] {
        let family = info?.family ?? ""
        return [
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": family.isEmpty ? [] : [family],
            "parameter_size": info?.parameterSize ?? "",
            "quantization_level": info?.quantization ?? "",
        ]
    }

    /// GET /api/tags -- the call an app makes to discover what is loaded.
    ///
    /// We serve exactly one model, so the list has one entry; that is the
    /// same shape /v1/models reports and it is not a limitation worth hiding.
    public static func tags(model name: String,
                            info: Tempo9ModelInfo?,
                            modified: Date = Date()) -> [String: Any] {
        let size = info?.sizeBytes ?? 0
        return ["models": [[
            "name": name,
            "model": name,
            "modified_at": timestamp(modified),
            "size": size,
            "digest": syntheticDigest(name: name, size: size),
            "details": details(name, info),
        ]]]
    }

    /// GET /api/ps -- what is resident.  One model, always loaded.
    public static func ps(model name: String,
                          info: Tempo9ModelInfo?) -> [String: Any] {
        let size = info?.sizeBytes ?? 0
        return ["models": [[
            "name": name,
            "model": name,
            "size": size,
            "digest": syntheticDigest(name: name, size: size),
            "details": details(name, info),
            // Ollama unloads on a timer; we do not, and saying "never"
            // beats inventing a plausible expiry a client would count down.
            "expires_at": "9999-12-31T23:59:59.000000000Z",
            "size_vram": size,
        ]]]
    }

    /// POST /api/show -- model detail.
    ///
    /// `capabilities` is the load-bearing field: clients read it to decide
    /// whether to offer tool calling or attach an image, so it must reflect
    /// what this build can actually do rather than advertising everything.
    public static func show(model name: String,
                            info: Tempo9ModelInfo?,
                            vision: Bool) -> [String: Any] {
        var caps = ["completion", "tools"]
        if vision { caps.append("vision") }
        return [
            // The template lives in the GGUF and is applied inside the
            // engine; we do not have its text to hand back, and echoing a
            // wrong one would be worse than an empty string.
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": details(name, info),
            "model_info": [:] as [String: Any],
            "capabilities": caps,
        ]
    }

    public static func version(_ v: String) -> [String: Any] {
        ["version": v]
    }

    // MARK: - chat request translation

    public struct ChatRequest {
        public var messages: [[String: Any]] = []
        public var stream = true
        public var temperature: Double?
        public var topP: Double?
        public var topK: Int?
        public var maxTokens: Int?
        public var tools: [[String: Any]] = []
        public var think = false
    }

    /// Ollama chat body -> the OpenAI-shaped messages our stack speaks.
    ///
    /// The one real translation is images: Ollama puts bare base64 strings in
    /// `images` alongside the text, where OpenAI uses typed content parts
    /// with a data: URI.  Rewriting them here is what lets a vision model be
    /// reached through this API at all.
    public static func chatRequest(from body: [String: Any]) -> ChatRequest {
        var r = ChatRequest()
        // Default TRUE -- see the header note.
        r.stream = body["stream"] as? Bool ?? true
        r.think = body["think"] as? Bool ?? false
        r.tools = body["tools"] as? [[String: Any]] ?? []

        if let o = body["options"] as? [String: Any] {
            r.temperature = o["temperature"] as? Double
            r.topP = o["top_p"] as? Double
            r.topK = o["top_k"] as? Int
            r.maxTokens = o["num_predict"] as? Int
        }

        for m in body["messages"] as? [[String: Any]] ?? [] {
            var out: [String: Any] = ["role": m["role"] as? String ?? "user"]
            let text = m["content"] as? String ?? ""
            let images = m["images"] as? [String] ?? []
            if images.isEmpty {
                out["content"] = text
            } else {
                var parts: [[String: Any]] = []
                if !text.isEmpty {
                    parts.append(["type": "text", "text": text])
                }
                for b64 in images {
                    parts.append([
                        "type": "image_url",
                        "image_url": ["url": "data:image/png;base64,\(b64)"],
                    ])
                }
                out["content"] = parts
            }
            // Pass tool plumbing straight through; both dialects use the
            // same role names and our template consumes them unchanged.
            if let tc = m["tool_calls"] { out["tool_calls"] = tc }
            if let n = m["tool_name"] { out["name"] = n }
            r.messages.append(out)
        }
        return r
    }

    /// POST /api/generate carries a bare prompt instead of messages.
    public static func generateRequest(from body: [String: Any])
        -> ChatRequest {
        var b = body
        b["messages"] = [["role": "user",
                          "content": body["prompt"] as? String ?? ""]]
        var r = chatRequest(from: b)
        if let images = body["images"] as? [String], !images.isEmpty {
            var m = r.messages.first ?? ["role": "user"]
            var parts: [[String: Any]] = []
            if let t = m["content"] as? String, !t.isEmpty {
                parts.append(["type": "text", "text": t])
            }
            for b64 in images {
                parts.append(["type": "image_url",
                              "image_url": ["url":
                                  "data:image/png;base64,\(b64)"]])
            }
            m["content"] = parts
            r.messages = [m]
        }
        return r
    }

    // MARK: - chat response

    /// Ollama tool calls carry arguments as an OBJECT.  Our bridge produces
    /// the OpenAI string form, so parse it back; a call whose arguments will
    /// not parse is dropped rather than sent as a string, because a client
    /// decoding into a typed struct fails hard on the wrong shape and a
    /// missing call at least degrades to plain text.
    public static func toolCalls(_ pairs: [(name: String, json: String)])
        -> [[String: Any]] {
        pairs.compactMap { call in
            guard let d = call.json.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: d)
                      as? [String: Any] else { return nil }
            return ["function": ["name": call.name, "arguments": obj]]
        }
    }

    /// One streaming line: a content delta, not yet done.
    public static func chunk(model: String, content: String,
                             generate: Bool) -> [String: Any] {
        var o: [String: Any] = ["model": model,
                                "created_at": timestamp(),
                                "done": false]
        if generate {
            o["response"] = content
        } else {
            o["message"] = ["role": "assistant", "content": content]
        }
        return o
    }

    /// The terminal object.  Durations are NANOSECONDS; clients divide by
    /// eval_count to display tok/s, so a wrong unit shows up as a plausible
    /// but 1000x-off speed rather than as an error.
    public static func done(model: String, content: String,
                            calls: [[String: Any]],
                            promptTokens: Int, completionTokens: Int,
                            seconds: Double, generate: Bool,
                            reason: String = "stop") -> [String: Any] {
        let total = Int64(max(0, seconds) * 1_000_000_000)
        var o: [String: Any] = [
            "model": model,
            "created_at": timestamp(),
            "done": true,
            "done_reason": calls.isEmpty ? reason : "stop",
            "total_duration": total,
            "load_duration": 0,
            "prompt_eval_count": promptTokens,
            "prompt_eval_duration": 0,
            "eval_count": completionTokens,
            "eval_duration": total,
        ]
        if generate {
            o["response"] = content
        } else {
            var msg: [String: Any] = ["role": "assistant", "content": content]
            if !calls.isEmpty { msg["tool_calls"] = calls }
            o["message"] = msg
        }
        return o
    }

    /// A whole non-streaming reply is the same object as the terminal one:
    /// full content plus done:true.  Deliberately the same builder, so the
    /// two paths cannot drift.
    public static func single(model: String, content: String,
                              calls: [[String: Any]],
                              promptTokens: Int, completionTokens: Int,
                              seconds: Double,
                              generate: Bool,
                              reason: String = "stop") -> [String: Any] {
        done(model: model, content: content, calls: calls,
             promptTokens: promptTokens, completionTokens: completionTokens,
             seconds: seconds, generate: generate, reason: reason)
    }
}
