// Copyright (c) 2026 Jiejing Zhang.

import XCTest
@testable import Tempo9
@testable import Tempo9Engine

final class ThinkSplitterTests: XCTestCase {
    private func drain(_ chunks: [String],
                       splitter: ThinkSplitter = ThinkSplitter())
        -> (String, String) {
        var c = "", r = ""
        for ch in chunks {
            let s = splitter.feed(ch)
            c += s.content; r += s.reasoning
        }
        let f = splitter.flush()
        return (c + f.content, r + f.reasoning)
    }

    func testWholeTagsInOneChunk() {
        let (c, r) = drain(["<think>weighing it</think>the answer"])
        XCTAssertEqual(r, "weighing it")
        XCTAssertEqual(c, "the answer")
    }

    /// The case that matters: tokens do not respect tag boundaries. Feeding
    /// one character at a time must give the same split, and must never leak
    /// a fragment like "<thi" into the visible content.
    func testTagSplitAcrossEveryCharacter() {
        let text = "<think>abc</think>xyz"
        let (c, r) = drain(text.map(String.init))
        XCTAssertEqual(r, "abc")
        XCTAssertEqual(c, "xyz")
        XCTAssertFalse(c.contains("<"), "leaked part of a tag into content")
    }

    func testNoThinkingAtAll() {
        let (c, r) = drain(["just an answer"])
        XCTAssertEqual(c, "just an answer")
        XCTAssertTrue(r.isEmpty)
    }

    /// A lone "<" that never becomes a tag is real text and must survive.
    func testDanglingAngleBracketIsNotEaten() {
        let (c, r) = drain(["5 < 7"])
        XCTAssertEqual(c, "5 < 7")
        XCTAssertTrue(r.isEmpty)
    }

    func testTrailingPartialTagIsFlushed() {
        let (c, _) = drain(["answer <thi"])
        XCTAssertEqual(c, "answer <thi")
    }

    func testSecondThinkBlock() {
        let (c, r) = drain(["<think>a</think>one<think>b</think>two"])
        XCTAssertEqual(r, "ab")
        XCTAssertEqual(c, "onetwo")
    }

    func testForceReasoningStartsInThinking() {
        let s = ThinkSplitter(forceReasoning: true)
        let (c, r) = drain(["reasoning first</think>then answer"], splitter: s)
        XCTAssertEqual(r, "reasoning first")
        XCTAssertEqual(c, "then answer")
    }

    /// Content before an open tag is content, not reasoning.
    func testPreambleBeforeOpenTag() {
        let (c, r) = drain(["hi <think>hmm</think> done"])
        XCTAssertEqual(c, "hi  done")
        XCTAssertEqual(r, "hmm")
    }

    // MARK: Gemma 4 channel tags
    //
    // Moved from the Eyes On app's `tooltest`, which printed pass or fail and
    // always exited 0.

    private func gemma() -> ThinkSplitter {
        ThinkSplitter(openTag: "<|channel>", closeTag: "<channel|>")
    }

    /// The observed shape: with thinking off, the template appends an empty,
    /// already-closed thinking block, the model writes into it anyway, and
    /// the reply arrives led by a lone close tag.
    func testGemmaCloseTagWithoutOpenTag() {
        let (c, r) = drain(["The user wants a summary.<channel|>Here it is."],
                           splitter: gemma())
        XCTAssertEqual(r, "The user wants a summary.")
        XCTAssertEqual(c, "Here it is.")
    }

    /// The channel name is ordinary text after the tag, not part of it.
    func testGemmaChannelNameIsReasoning() {
        let (c, r) = drain(["<|channel>thought\nplanning<channel|>answer"],
                           splitter: gemma())
        XCTAssertEqual(r, "thought\nplanning")
        XCTAssertEqual(c, "answer")
    }

    /// A repeated close tag mid-answer is a control token, not text.
    func testGemmaRepeatedCloseTagIsDropped() {
        let (c, r) = drain(["<|channel>t<channel|>one<channel|>two"],
                           splitter: gemma())
        XCTAssertEqual(r, "t")
        XCTAssertEqual(c, "onetwo")
    }

    func testGemmaTagSplitAcrossChunks() {
        let (c, r) = drain(["<|channel>thinking<chan", "nel|>done"],
                           splitter: gemma())
        XCTAssertEqual(r, "thinking")
        XCTAssertEqual(c, "done")
    }
}
