// Copyright (c) 2026 Jiejing Zhang.

import XCTest
@testable import GGUFKit

final class IncrementalDecoderTests: XCTestCase {
    /// The boundary logic on its own, without needing a tokenizer: feed the
    /// bytes of a Chinese string one at a time and require that the pieces
    /// concatenate back to the original with no U+FFFD anywhere. A decoder
    /// that emits per-token would fail this on byte 1 of every character.
    func testSplitMultibyteNeverEmitsReplacementChar() {
        let text = "北京，古称燕京。Hello 🌏 end"
        let bytes = Array(text.utf8)
        var out = ""
        var buf: [UInt8] = []
        for b in bytes {
            buf.append(b)
            let cut = IncrementalDecoder.completePrefixLength(buf)
            if cut > 0 {
                out += String(decoding: buf[0..<cut], as: UTF8.self)
                buf.removeFirst(cut)
            }
        }
        out += buf.isEmpty ? "" : String(decoding: buf, as: UTF8.self)
        XCTAssertEqual(out, text)
        XCTAssertFalse(out.contains("\u{FFFD}"), "decoder invented a U+FFFD")
    }

    func testHoldsBackPartialSequence() {
        // First two bytes of 北 (E5 8C 97): nothing may be emitted yet.
        XCTAssertEqual(IncrementalDecoder.completePrefixLength([0xE5, 0x8C]), 0)
        // All three: emit all three.
        XCTAssertEqual(IncrementalDecoder.completePrefixLength([0xE5, 0x8C, 0x97]), 3)
        // ASCII then a partial: emit only the ASCII.
        XCTAssertEqual(IncrementalDecoder.completePrefixLength([0x41, 0xE5, 0x8C]), 1)
    }

    func testFourByteEmoji() {
        let e = Array("🌏".utf8)   // F0 9F 8C 8F
        XCTAssertEqual(IncrementalDecoder.completePrefixLength(Array(e[0..<3])), 0)
        XCTAssertEqual(IncrementalDecoder.completePrefixLength(e), 4)
    }

    /// A malformed byte must not wedge the stream forever.
    func testStrayContinuationByteIsNotHeldForever() {
        XCTAssertEqual(IncrementalDecoder.completePrefixLength([0x80]), 1)
    }
}
