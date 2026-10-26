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

@Suite("Anthropic usage reports the cache")
struct AnthropicCacheUsageTests {
    private func reply(prompt: Int, hit: Int64?, lastPoint: Int) -> LocalReply {
        LocalReply(text: "3", reasoning: "", promptTokens: prompt,
                   completionTokens: 2, seconds: 0,
                   engine: hit.map {
                       RequestStats(promptTokens: Int64(prompt),
                                    generatedTokens: 2,
                                    prefixCacheHitTokens: $0,
                                    prefillMs: 0, decodeMs: 0)
                   },
                   maxTokens: 64, lastCachePoint: lastPoint)
    }

    @Test("read + creation + input add up to the prompt")
    func split() {
        // Claude Code turn 2: 43479 tokens, 32000 served from the cache,
        // last breakpoint (the tool_result) at 43461.
        let u = OpenAIServer.anthropicUsage(
            reply(prompt: 43479, hit: 32000, lastPoint: 43461))
        #expect(u["cache_read_input_tokens"] as? Int == 32000)
        #expect(u["cache_creation_input_tokens"] as? Int == 11461)
        #expect(u["input_tokens"] as? Int == 18)
        #expect(u["output_tokens"] as? Int == 2)
    }

    @Test("no breakpoints: nothing is created; no engine stats: nothing read")
    func degenerate() {
        let a = OpenAIServer.anthropicUsage(
            reply(prompt: 100, hit: 64, lastPoint: 0))
        #expect(a["cache_read_input_tokens"] as? Int == 64)
        #expect(a["cache_creation_input_tokens"] as? Int == 0)
        #expect(a["input_tokens"] as? Int == 36)
        let b = OpenAIServer.anthropicUsage(
            reply(prompt: 100, hit: nil, lastPoint: 90))
        #expect(b["cache_read_input_tokens"] as? Int == 0)
        #expect(b["cache_creation_input_tokens"] as? Int == 90)
        #expect(b["input_tokens"] as? Int == 10)
        // A hit past the last breakpoint (the engine resumes from its own
        // frontier) creates nothing and never goes negative.
        let c = OpenAIServer.anthropicUsage(
            reply(prompt: 100, hit: 96, lastPoint: 90))
        #expect(c["cache_creation_input_tokens"] as? Int == 0)
        #expect(c["input_tokens"] as? Int == 4)
    }
}

@Suite("cache point offsets")
struct CachePointOffsetTests {
    /// cumulative bytes of tokens with lengths 3, 1, 4, 2: [0, 3, 4, 8, 10]
    private let cumulative = [0, 3, 4, 8, 10]

    @Test("a token straddling the boundary is excluded")
    func straddle() {
        #expect(LocalSession.tokenOffset(atByte: 0, cumulative: cumulative) == 0)
        #expect(LocalSession.tokenOffset(atByte: 3, cumulative: cumulative) == 1)
        #expect(LocalSession.tokenOffset(atByte: 7, cumulative: cumulative) == 2)
        #expect(LocalSession.tokenOffset(atByte: 8, cumulative: cumulative) == 3)
        #expect(LocalSession.tokenOffset(atByte: 99, cumulative: cumulative) == 4)
    }

    @Test("byte matching is bounds-checked")
    func matches() {
        let h = Array("<tools>a</tools>".utf8)
        #expect(LocalSession.matches(h, at: 8, Array("</tools>".utf8)))
        #expect(!LocalSession.matches(h, at: 9, Array("</tools>".utf8)))
        #expect(!LocalSession.matches(h, at: 10, Array("</tools>".utf8)))
        #expect(LocalSession.matches(h, at: 16, []))
        #expect(!LocalSession.matches(h, at: -1, Array("<".utf8)))
    }
}
