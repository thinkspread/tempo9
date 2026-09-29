// Copyright (c) 2026 Jiejing Zhang.
//
// One test per bug that reached a user. Each names the symptom it prevents,
// because a regression test whose failure message is "expected true" tells
// the next person nothing about what broke.

import Testing
import Foundation
@testable import GGUFKit
@testable import Tempo9Engine
@testable import Tempo9

// MARK: - Tokenizer recursion (crashed on turn two of an agent session)

@Suite("Tokenizer depth")
struct TokenizerDepthTests {

    /// A prompt carrying tool schemas renders a template with special-token
    /// markers by the hundred. The tokenizer used to recurse once per marker,
    /// and requests are served on Swift's cooperative pool, whose stacks are
    /// a fraction of the main thread's -- so it took SIGBUS at about 2000
    /// markers. Measured: 2000 ok, 2100 dead.
    ///
    /// The assertion is that it RETURNS. Ten thousand markers is five times
    /// the old ceiling, and a recursive implementation cannot reach it.
    @Test("ten thousand markers do not exhaust the stack")
    func deepSpecialTokens() async throws {
        let tok = try makeToySpecialTokenizer()
        let text = String(repeating: "<|im_start|>abc ", count: 10_000)

        // On the cooperative pool ON PURPOSE: the main thread's 8 MB stack
        // hides this bug completely, which is why it shipped.
        let ids = await Task.detached { tok.encode(text) }.value
        #expect(ids.count > 10_000,
                "encode returned but produced too few ids to be real")
    }

    @Test("markers are still split correctly")
    func specialTokenSplitting() throws {
        let tok = try makeToySpecialTokenizer()
        let ids = tok.encode("<|im_start|>a<|im_end|>")
        // The two markers must appear as their own ids, in order.
        #expect(ids.contains(kImStart))
        #expect(ids.contains(kImEnd))
        #expect(ids.firstIndex(of: kImStart)! < ids.firstIndex(of: kImEnd)!)
    }
}

// MARK: - Backwards search (the first, wrong, diagnosis of the same crash)

@Suite("Reasoning tag scan")
struct ReasoningTagTests {

    /// `range(of:options:.backwards)` recurses through Foundation and blew
    /// the same small stack. Replaced with a byte scan; these cases are the
    /// ones checked against the old implementation when it was swapped.
    @Test("last occurrence matches on the cases that mattered",
          arguments: [
            ("<think>", "a <think> b </think> c", 2),
            ("</think>", "a <think> b </think> c", 12),
            ("<think>", "x<think>y<think>z", 9),
            ("<think>", "no marker here", -1),
            ("<think>", "", -1),
          ])
    func lastByteIndex(needle: String, haystack: String, expected: Int) {
        let got = LocalSession.lastByteIndex(of: needle, in: haystack) ?? -1
        #expect(got == expected,
                "lastByteIndex(\(needle)) = \(got), expected \(expected)")
    }

    @Test("a 40 KB haystack does not recurse")
    func largeHaystack() async {
        let hay = String(repeating: "pad ", count: 10_000) + "<think>tail"
        let got = await Task.detached {
            LocalSession.lastByteIndex(of: "<think>", in: hay)
        }.value
        #expect(got == 40_000)
    }
}

// MARK: - helpers

let kImStart = 3
let kImEnd = 4

/// A tiny byte-level tokenizer with two special tokens. Enough to exercise
/// the segment splitter without a multi-gigabyte model on disk.
private func makeToySpecialTokenizer() throws -> BPETokenizer {
    let tokensBase = ["<unk>", "a", "b", "<|im_start|>", "<|im_end|>", "c", " "]
    let typesBase: [Double] = [2, 1, 1, 3, 3, 1, 1]     // 3 == special
    // Byte fallback for everything else the text can contain.
    var tokens = tokensBase, types = typesBase
    for b in 0..<256 {
        tokens.append(String(format: "<0x%02X>", b)); types.append(6)
    }
    return try BPETokenizer(tokens: tokens, merges: [], tokenTypes: types,
                            dialect: .gpt2)
}

// MARK: - Error reporting (a failed request that looked like success)

@Suite("Error surfacing")
struct ErrorSurfacingTests {

    /// The engine refuses an over-long request with a precise status, and the
    /// stream used to close with a bare [DONE] -- a 200 carrying nothing,
    /// which every client reads as "the model had nothing to say". OpenClaw
    /// retried three times and reported "Agent couldn't generate a response".
    ///
    /// `describe(_:)` is the piece that keeps the reason: on a plain Swift
    /// error `localizedDescription` is the type name, so the engine's status
    /// and detail -- the only part that says WHY -- were dropped even on the
    /// 500 path.
    @Test("an engine error keeps its status and detail")
    func engineErrorIsDescribed() {
        let e = Tempo9Error.engine(status: 5,
                                      detail: "request_start: ALLSPARK_EXCEED_LIMIT_ERROR")
        let s = OpenAIServer.describe(e)
        #expect(s.contains("5"), "status dropped from: \(s)")
        #expect(s.contains("EXCEED_LIMIT"), "detail dropped from: \(s)")
        #expect(s != "The operation couldn’t be completed.",
                "fell back to the useless default")
    }

    /// A non-engine error still has to say something.
    @Test("other errors are not blanked")
    func otherErrorsSurvive() {
        struct Boom: Error {}
        #expect(!OpenAIServer.describe(Boom()).isEmpty)
    }
}
