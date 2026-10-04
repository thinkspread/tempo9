// Copyright (c) 2026 Jiejing Zhang.
//
// Anthropic cache_control -> engine cache points.
//
// The parser half: where each breakpoint lands in the CONVERTED conversation
// (system folded in at 0, tool_result turned into a tool message), and the
// tools-level boundary every breakpoint implies.  The engine half's contract
// (strictly increasing offsets, TTL never growing along the prompt) is
// checked on PromptCachePoint.normalized.  Hermetic: no engine, no model.

import Testing
import Foundation
import Tempo9Engine
@testable import Tempo9

@Suite("cache_control")
struct CacheControlTests {
    private let oneHour: [String: Any] = ["type": "ephemeral", "ttl": "1h"]
    private let tool: [String: Any] = [
        "name": "Glob", "description": "find files",
        "input_schema": ["type": "object",
                         "properties": ["pattern": ["type": "string"]]]]

    /// Claude Code's shape: system blocks and a system-role message carry
    /// 1h breakpoints, the last tool_result carries one, tools carry none.
    @Test("Claude Code's breakpoints map onto converted messages")
    func claudeCodeShape() throws {
        let body: [String: Any] = [
            "tools": [tool],
            "system": [["type": "text", "text": "billing header"],
                       ["type": "text", "text": "You are an agent.",
                        "cache_control": oneHour]],
            "messages": [
                ["role": "user", "content": [
                    ["type": "text", "text": "How many .txt files?"]]],
                ["role": "system", "content": [
                    ["type": "text", "text": "# Environment",
                     "cache_control": oneHour]]],
                ["role": "assistant", "content": [
                    ["type": "tool_use", "id": "toolu_1", "name": "Glob",
                     "input": ["pattern": "*.txt"]]]],
                ["role": "user", "content": [
                    ["type": "tool_result", "tool_use_id": "toolu_1",
                     "content": "a.txt", "cache_control": oneHour]]],
            ]]
        let r = try APIRequest.anthropic(body, speculationK: 0)
        // [system, user, assistant, tool]
        #expect(r.messages.map { $0["role"] as? String } ==
                ["system", "user", "assistant", "tool"])
        #expect(r.cacheBreakpoints == [
            .tools(ttl: 3600),
            .message(index: 0, ttl: 3600),
            .message(index: 3, ttl: 3600),
        ])
    }

    @Test("no cache_control, no breakpoints -- not even the tools level")
    func none() throws {
        let r = try APIRequest.anthropic([
            "tools": [tool],
            "messages": [["role": "user", "content": "hi"]],
        ], speculationK: 0)
        #expect(r.cacheBreakpoints.isEmpty)
    }

    @Test("a 5-minute breakpoint without tools has no tools level")
    func fiveMinutesNoTools() throws {
        let r = try APIRequest.anthropic([
            "messages": [["role": "user", "content": [
                ["type": "text", "text": "hi",
                 "cache_control": ["type": "ephemeral"]]]]],
        ], speculationK: 0)
        #expect(r.cacheBreakpoints == [.message(index: 0, ttl: 300)])
    }

    @Test("normalized: sorted, deduped, inside the prompt, TTL non-growing")
    func normalized() {
        let p = PromptCachePoint.normalized([
            .init(tokenOffset: 50, ttlSeconds: 300),
            .init(tokenOffset: 10, ttlSeconds: 3600),
            .init(tokenOffset: 50, ttlSeconds: 3600),
            .init(tokenOffset: 0, ttlSeconds: 300),
            .init(tokenOffset: 200, ttlSeconds: 300),
        ], promptTokens: 100)
        #expect(p == [.init(tokenOffset: 10, ttlSeconds: 3600),
                      .init(tokenOffset: 50, ttlSeconds: 3600)])
        // A later point may not outlive an earlier one (the engine refuses
        // the request otherwise): clamp it, never fail the request.
        #expect(PromptCachePoint.normalized([
            .init(tokenOffset: 10, ttlSeconds: 300),
            .init(tokenOffset: 20, ttlSeconds: 3600),
        ], promptTokens: 100) == [.init(tokenOffset: 10, ttlSeconds: 300),
                                  .init(tokenOffset: 20, ttlSeconds: 300)])
    }
}

@Suite("tool-boundary ladder")
struct ToolLadderTests {
    /// Claude Code's tail, in tokens (bytesPerToken 1 keeps it readable):
    /// ... TaskStop 201, WaitForMcpServers 214, WebFetch 240, WebSearch 214,
    /// Workflow 1332, Write 161.  The 2k rung must land in FRONT of
    /// WaitForMcpServers (2161 from the end), which is the race it exists for.
    @Test("rungs land on the first tool boundary at or beyond each distance")
    func claudeCodeTail() {
        let sizes: [Double] = [5000, 3000, 1155, 1198, 449, 389, 201,
                               214, 240, 214, 1332, 161]
        let rungs = LocalSession.toolLadderRungs(
            sizes: sizes, bytesPerToken: 1, distances: [1024, 2048, 4096, 8192])
        // 1024 -> before Workflow (1493 from the end); 2048 -> before
        // WaitForMcpServers (2161); 4096 -> before the 1198 tool (4398);
        // 8192 -> before the 3000 tool (8553).
        #expect(rungs == [1, 3, 7, 10])
    }

    @Test("a distance longer than the list gets no rung, and k never hits 0")
    func shortList() {
        #expect(LocalSession.toolLadderRungs(
            sizes: [100, 100, 100], bytesPerToken: 1,
            distances: [150, 1000]) == [1])
    }
}
