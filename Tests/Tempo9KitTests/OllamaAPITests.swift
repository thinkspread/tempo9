// Copyright (c) 2026 Jiejing Zhang.
//
// One test per way the Ollama dialect differs from OpenAI.  Each of these
// differences fails SILENTLY against a real client -- the server answers 200
// and the app shows nothing -- so they are exactly the ones worth pinning.

import Testing
import Foundation
@testable import Tempo9

@Suite("Ollama wire format")
struct OllamaAPITests {

    @Test("stream defaults to TRUE, unlike OpenAI")
    func streamDefaultsTrue() {
        // Reading this flag with `?? false` yields one JSON body while the
        // client parses lines, and it hangs on a stream that never came.
        let r = OllamaAPI.chatRequest(from: [
            "messages": [["role": "user", "content": "hi"]],
        ])
        #expect(r.stream == true,
                "absent `stream` must mean streaming; OpenAI's default is the opposite")

        let off = OllamaAPI.chatRequest(from: [
            "messages": [["role": "user", "content": "hi"]], "stream": false,
        ])
        #expect(off.stream == false, "explicit stream:false must still win")
    }

    @Test("tool arguments are a JSON object, not a JSON string")
    func toolArgumentsAreObjects() {
        // OpenAI sends "{\"city\":\"Paris\"}"; Ollama sends {"city":"Paris"}.
        // A client decoding into a typed struct fails outright on the wrong
        // one, so the bridge's string form must be parsed back.
        let calls = OllamaAPI.toolCalls([("get_weather", "{\"city\":\"Paris\"}")])
        #expect(calls.count == 1)
        let args = (calls[0]["function"] as? [String: Any])?["arguments"]
        #expect(args is [String: Any],
                "arguments must decode as an object, got \(type(of: args))")
        #expect((args as? [String: Any])?["city"] as? String == "Paris")
    }

    @Test("a tool call with unparseable arguments is dropped, not stringified")
    func unparseableToolCallDropped() {
        // Degrading to plain text beats handing a client a shape it will
        // hard-fail on.
        let calls = OllamaAPI.toolCalls([("broken", "not json at all")])
        #expect(calls.isEmpty)
    }

    @Test("terminal object carries done:true and nanosecond durations")
    func doneShape() {
        let d = OllamaAPI.done(model: "m", content: "", calls: [],
                               promptTokens: 7, completionTokens: 3,
                               seconds: 1.5, generate: false)
        #expect(d["done"] as? Bool == true)
        #expect(d["eval_count"] as? Int == 3)
        #expect(d["prompt_eval_count"] as? Int == 7)
        // Clients divide duration by eval_count to show tok/s: a seconds
        // value here displays a plausible, 1000x-wrong speed rather than
        // erroring, which is why the unit is pinned.
        #expect(d["total_duration"] as? Int64 == 1_500_000_000,
                "durations are NANOSECONDS")
    }

    @Test("generate uses `response`, chat uses `message`")
    func generateFieldName() {
        let chat = OllamaAPI.chunk(model: "m", content: "x", generate: false)
        #expect(chat["message"] != nil && chat["response"] == nil)
        let gen = OllamaAPI.chunk(model: "m", content: "x", generate: true)
        #expect(gen["response"] as? String == "x" && gen["message"] == nil)
    }

    @Test("bare base64 images become typed content parts")
    func imageTranslation() {
        // Ollama puts base64 in `images` beside the text; our vision path
        // reads OpenAI-style parts with a data: URI.  Without this rewrite a
        // vision model is unreachable through this API.
        let r = OllamaAPI.chatRequest(from: [
            "messages": [["role": "user", "content": "what is this",
                          "images": ["QUJD"]]],
        ])
        let parts = r.messages.first?["content"] as? [[String: Any]]
        #expect(parts?.count == 2, "expected one text part and one image part")
        #expect(parts?[0]["type"] as? String == "text")
        let url = (parts?[1]["image_url"] as? [String: Any])?["url"] as? String
        #expect(url == "data:image/png;base64,QUJD")
    }

    @Test("capabilities reflect the build, not a wish list")
    func capabilitiesAreHonest() {
        // Clients read this to decide whether to offer tool calling or let
        // the user attach an image at all.
        let novision = OllamaAPI.show(model: "m", info: nil, vision: false)
        let caps = novision["capabilities"] as? [String] ?? []
        #expect(caps.contains("tools"))
        #expect(!caps.contains("vision"),
                "vision must not be advertised without a loaded tower")

        let vision = OllamaAPI.show(model: "m", info: nil, vision: true)
        #expect((vision["capabilities"] as? [String] ?? []).contains("vision"))
    }

    @Test("digest is stable across calls and moves when size changes")
    func digestStability() {
        // Clients use it as a cache key; an empty string breaks those, and a
        // value that changes per call defeats the caching.
        let a = OllamaAPI.syntheticDigest(name: "m", size: 100)
        let b = OllamaAPI.syntheticDigest(name: "m", size: 100)
        let c = OllamaAPI.syntheticDigest(name: "m", size: 101)
        #expect(a == b)
        #expect(a != c)
        #expect(a.hasPrefix("sha256:"))
    }

    @Test("unknown model facts are reported blank, never invented")
    func unknownsStayUnknown() {
        let t = OllamaAPI.tags(model: "m", info: nil)
        let m = (t["models"] as? [[String: Any]])?.first
        #expect(m?["size"] as? Int64 == 0)
        let d = m?["details"] as? [String: Any]
        #expect(d?["quantization_level"] as? String == "")
        #expect(d?["format"] as? String == "gguf")
    }
}
