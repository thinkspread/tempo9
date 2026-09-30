// Copyright (c) 2026 Jiejing Zhang.
//
// A request that declares no tools must get no tool calls back.  Agent
// conversations replayed WITHOUT their tool list (SiliconBench's agent
// split) came back as finish_reason "tool_calls" with empty content --
// 30 of 100 on Qwen3-0.6B -- because the server parsed <tool_call> text
// regardless of what the request declared.

import Testing
import Foundation
import Tempo9Engine
@testable import Tempo9

private let reply = "Let me check.\n<tool_call>\n<function=get_weather>\n"
    + "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
private let tools: [[String: Any]] = [[
    "name": "get_weather",
    "parameters": ["type": "object",
                   "properties": ["city": ["type": "string"]]]]]

/// Worst case for the stream filter: one character per delta.
private func stream(_ f: ToolCallBridge.StreamFilter, _ s: String) -> String {
    var out = ""
    for ch in s { out += f.feed(String(ch)) }
    return out + f.flush()
}

@Suite struct ToolCallBridgeNoToolsTests {
    @Test func noToolsMeansNoCalls() {
        let p = ToolCallBridge.parse(reply, tools: [])
        #expect(p.calls.isEmpty)
        #expect(p.text == reply.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    @Test func declaredToolsAreParsed() {
        let p = ToolCallBridge.parse(reply, tools: tools)
        #expect(p.calls.count == 1)
        #expect(p.calls.first?.name == "get_weather")
        #expect(p.calls.first?.argumentsJSON.contains("Paris") == true)
        #expect(p.text == "Let me check.")
    }

    @Test func streamPassesThroughWithoutTools() {
        let f = ToolCallBridge.StreamFilter(tools: [])
        #expect(stream(f, reply) == reply)
        #expect(f.newCallNames().isEmpty)
        #expect(f.completedCalls(tools: []).isEmpty)
    }

    @Test func streamHoldsBackCallsWithTools() {
        let f = ToolCallBridge.StreamFilter(tools: tools)
        #expect(stream(f, reply) == "Let me check.\n")
        #expect(f.newCallNames() == ["get_weather"])
        #expect(f.completedCalls(tools: tools).count == 1)
    }

    @Test func plainTextUnchangedEitherWay() {
        let plain = "Paris is the capital of France."
        #expect(ToolCallBridge.parse(plain, tools: []).text == plain)
        #expect(ToolCallBridge.parse(plain, tools: tools).text == plain)
        #expect(stream(ToolCallBridge.StreamFilter(tools: []), plain) == plain)
        #expect(stream(ToolCallBridge.StreamFilter(tools: tools), plain) == plain)
    }
}
