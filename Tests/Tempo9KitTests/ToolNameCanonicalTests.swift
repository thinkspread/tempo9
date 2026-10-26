// Copyright (c) 2026 Jiejing Zhang.
//
// A call's name must be one the client declared.  Claude Code in --bare mode
// declares Bash/Edit/Read; Qwen3.5 called `bash`, Claude Code answered "No
// such tool available: bash", and the model called `bash` again -- 700+
// times, no end.  Hermetic: parser only.

import Testing
import Foundation
import Tempo9Engine
@testable import Tempo9

@Suite("tool call names map onto the declared tools")
struct ToolNameCanonicalTests {
    private func tool(_ name: String) -> [String: Any] {
        ["type": "function", "function": [
            "name": name, "description": "",
            "parameters": ["type": "object",
                           "properties": ["command": ["type": "string"],
                                          "limit": ["type": "integer"]]]]]
    }
    private let bashCall = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>"

    @Test("a case-only mismatch takes the declared name, and its schema")
    func caseMismatch() {
        let r = ToolCallBridge.parse(bashCall,
                                     tools: [tool("Bash"), tool("Read")])
        #expect(r.calls.map(\.name) == ["Bash"])
        // The schema lookup follows the declared name: `limit` is coerced
        // to an integer, which it was not while the name stayed `bash`.
        #expect(r.calls.first?.argumentsJSON.contains("\"limit\":3") == true,
                "\(r.calls.first?.argumentsJSON ?? "")")
    }

    @Test("exact, unknown and ambiguous names are left alone")
    func leftAlone() {
        #expect(ToolCallBridge.canonicalName("bash", tools: [tool("bash"), tool("Bash")]) == "bash")
        #expect(ToolCallBridge.canonicalName("grep", tools: [tool("Bash")]) == "grep")
        #expect(ToolCallBridge.canonicalName("READ", tools: [tool("Read"), tool("read")]) == "READ")
        #expect(ToolCallBridge.canonicalName("web_fetch", tools: [tool("web.fetch")]) == "web.fetch")
    }

    @Test("the stream announces the declared name too")
    func streamed() {
        let tools = [tool("Bash")]
        let f = ToolCallBridge.StreamFilter(tools: tools)
        _ = f.feed(bashCall)
        #expect(f.newCallNames() == ["Bash"])
        #expect(f.completedCalls(tools: tools).map(\.name) == ["Bash"])
    }
}
