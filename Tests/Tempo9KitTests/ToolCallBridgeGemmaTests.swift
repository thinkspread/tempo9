// Copyright (c) 2026 Jiejing Zhang.
//
// Gemma 4 tool calls through the bridge.  ToolCallParser read Gemma's
// `<|tool_call>call:NAME{k:v}<tool_call|>` while the bridge's visible-text
// cut and stream filter still knew only Qwen's markers, so a declared-tools
// Gemma reply came back with the call parsed into tool_calls AND its raw
// text left in content -- streamed or not.

import Testing
import Foundation
import Tempo9Engine
@testable import Tempo9

private let reply = "Let me check.<|tool_call>call:get_weather{city:Paris,days:3}<tool_call|>"
private let tools: [[String: Any]] = [[
    "name": "get_weather",
    "parameters": ["type": "object",
                   "properties": ["city": ["type": "string"],
                                  "days": ["type": "integer"]]]]]

/// Worst case for the stream filter: one character per delta.
private func stream(_ f: ToolCallBridge.StreamFilter, _ s: String) -> String {
    var out = ""
    for ch in s { out += f.feed(String(ch)) }
    return out + f.flush()
}

@Suite struct ToolCallBridgeGemmaTests {
    @Test func declaredToolsLeaveNoCallTextInContent() {
        let p = ToolCallBridge.parse(reply, tools: tools)
        #expect(p.calls.count == 1)
        #expect(p.calls.first?.name == "get_weather")
        #expect(p.calls.first?.argumentsJSON.contains("\"city\":\"Paris\"") == true)
        #expect(p.calls.first?.argumentsJSON.contains("\"days\":3") == true)
        #expect(p.text == "Let me check.")
    }

    @Test func callOnlyReplyHasEmptyContent() {
        let only = "<|tool_call>call:get_weather{city:Paris}<tool_call|>"
        let p = ToolCallBridge.parse(only, tools: tools)
        #expect(p.calls.count == 1)
        #expect(p.text.isEmpty)
    }

    @Test func streamHoldsBackGemmaCall() {
        let f = ToolCallBridge.StreamFilter(tools: tools)
        #expect(stream(f, reply) == "Let me check.")
        #expect(f.newCallNames() == ["get_weather"])
        #expect(f.completedCalls(tools: tools).count == 1)
    }

    @Test func streamAnnouncesNameOnlyOnceItsBraceArrives() {
        let f = ToolCallBridge.StreamFilter(tools: tools)
        _ = f.feed("<|tool_call>call:get_wea")
        #expect(f.newCallNames().isEmpty)
        _ = f.feed("ther{city:")
        #expect(f.newCallNames() == ["get_weather"])
        #expect(f.completedCalls(tools: tools).isEmpty)   // no closer yet
        _ = f.feed("Paris}<tool_call|>")
        #expect(f.completedCalls(tools: tools).count == 1)
    }

    @Test func noToolsStillPassesGemmaTextThrough() {
        // The no-tools contract (ToolCallBridgeNoToolsTests) holds for this
        // dialect too: the reply is text, exactly as the model wrote it.
        let p = ToolCallBridge.parse(reply, tools: [])
        #expect(p.calls.isEmpty)
        #expect(p.text == reply)
        let f = ToolCallBridge.StreamFilter(tools: [])
        #expect(stream(f, reply) == reply)
    }

    @Test func plainGemmaProseUnchanged() {
        // A lone '<' or '|' in prose must not be held back forever.
        let prose = "Use a < b | c, then call me."
        let f = ToolCallBridge.StreamFilter(tools: tools)
        #expect(stream(f, prose) == prose)
        #expect(ToolCallBridge.parse(prose, tools: tools).text == prose)
    }
}
