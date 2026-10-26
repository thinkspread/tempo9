// Copyright (c) 2026 Jiejing Zhang.
//
// Flint's pre-tokenizer is three Isolated Splits in sequence, not one regex.
// Both ways of getting it wrong still produce valid ids, so each test uses a
// merge that only fires when the stage boundaries are wrong.

import XCTest
@testable import GGUFKit

final class StagedPretokenizerTests: XCTestCase {
    private let enc: [UInt8: Character] = {
        var m = [UInt8: Character]()
        for (b, s) in BPETokenizer.byteToUnicode() { m[b] = Character(s) }
        return m
    }()
    private func mapped(_ s: String) -> String {
        String(s.utf8.map { enc[$0]! })
    }

    /// Every byte as a token, plus "42" and "<space>+first byte of 你".
    private func makeFlint() throws -> BPETokenizer {
        var tokens = BPETokenizer.byteToUnicode().map { String(Character($0.1)) }
        let sp = mapped(" "), ni = String(mapped("你").first!)
        tokens += ["42", sp + ni]
        let types = [Double](repeating: 1, count: tokens.count)
        return try BPETokenizer(
            tokens: tokens, merges: ["4 2", "\(sp) \(ni)"], tokenTypes: types,
            dialect: .gpt2,
            pattern: TokenizerConfig.pattern(forPretokenizer: "flint"),
            preSplits: TokenizerConfig.preSplits(forPretokenizer: "flint"),
            normalization: .forPretokenizer("flint"))
    }

    /// The last stage has no digit branch.  Keeping only its matches -- what
    /// a single-regex dialect does -- silently drops every number.
    func testDigitsSurviveTheLastStage() throws {
        let t = try makeFlint()
        XCTAssertEqual(t.encode("a42"), [t.id(of: "a")!, t.id(of: "42")!])
    }

    /// CJK is \p{Lo}, so one alternation would hand " 你好" to the word
    /// branch as one piece and merge the space into it.  Stage two isolates
    /// the CJK run first, so the space stays alone.
    func testSpaceBeforeCJKStaysItsOwnPiece() throws {
        let t = try makeFlint()
        let ids = t.encode(" 你好")
        XCTAssertEqual(ids.first, t.id(of: mapped(" ")))
        XCTAssertEqual(ids.count, 1 + Array("你好".utf8).count)
    }

    func testFlintIsNotNormalised() {
        XCTAssertEqual(Normalization.forPretokenizer("flint"), .none)
        XCTAssertTrue(TokenizerConfig.preSplits(forPretokenizer: "qwen2").isEmpty)
    }
}
