// Copyright (c) 2026 Jiejing Zhang.
//
// The literal failure this splitter exists to prevent:
// "analysisThe user asks...assistantfinalParis" -- channel names and role
// headers concatenated into the answer once the markers were stripped.

import XCTest
@testable import Tempo9
@testable import Tempo9Engine

final class HarmonySplitterTests: XCTestCase {

    private func run(_ pieces: [String]) -> (content: String, reasoning: String) {
        let s = HarmonySplitter()
        var content = "", reasoning = ""
        for p in pieces {
            let out = s.feed(p)
            content += out.content
            reasoning += out.reasoning
        }
        let tail = s.flush()
        return (content + tail.content, reasoning + tail.reasoning)
    }

    func testTwoBlockTurn() {
        let turn = "<|channel|>analysis<|message|>The user asks. Easy."
            + "<|end|><|start|>assistant<|channel|>final<|message|>Paris"
        let out = run([turn])
        XCTAssertEqual(out.content, "Paris")
        XCTAssertTrue(out.reasoning.contains("The user asks. Easy."))
        // The headers must never reach the answer.
        XCTAssertFalse(out.content.contains("final"))
        XCTAssertFalse(out.content.contains("analysis"))
        XCTAssertFalse(out.content.contains("assistant"))
    }

    func testCharacterByCharacterStreaming() {
        // Markers arrive as atomic tokens in production, but nothing about
        // the splitter is allowed to depend on that.
        let turn = "<|channel|>analysis<|message|>Think.<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Answer!"
        let out = run(turn.map(String.init))
        XCTAssertEqual(out.content, "Answer!")
        XCTAssertTrue(out.reasoning.contains("Think."))
    }

    func testRawContinuationWithoutMarkers() {
        // Template-less continuation: the model emits plain text and the
        // whole thing is the answer.
        let out = run([" Paris."])
        XCTAssertEqual(out.content, " Paris.")
        XCTAssertEqual(out.reasoning, "")
    }

    func testPartialMarkerTailIsHeldThenFlushedAsText() {
        let s = HarmonySplitter()
        let first = s.feed("abc<|chan")
        XCTAssertEqual(first.content, "abc")        // tail held back
        let tail = s.flush()
        XCTAssertEqual(tail.content, "<|chan")      // never became a marker
    }

    func testPartialMarkerCompletesAcrossFeeds() {
        let s = HarmonySplitter()
        var content = "", reasoning = ""
        for p in ["abc<|chan", "nel|>analysis<|mess", "age|>T<|end|>",
                  "<|channel|>final<|message|>A"] {
            let out = s.feed(p)
            content += out.content; reasoning += out.reasoning
        }
        XCTAssertEqual(content, "abcA")
        XCTAssertTrue(reasoning.contains("T"))
    }

    func testToolCallBlockGoesToReasoning() {
        // Until tool-call parsing lands for harmony, a commentary/tool block
        // must not leak into the answer.
        let turn = "<|channel|>commentary to=functions.get_weather"
            + "<|message|>{\"city\":\"Paris\"}<|call|>"
        let out = run([turn])
        XCTAssertEqual(out.content, "")
        XCTAssertTrue(out.reasoning.contains("{\"city\":\"Paris\"}"))
    }
}
