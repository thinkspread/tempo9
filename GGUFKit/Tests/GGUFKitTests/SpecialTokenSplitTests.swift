// Copyright (c) 2026 Jiejing Zhang.
//
// The special-token splitter, rewritten as a single byte-level pass. The
// invariant: splitting on specials is the same as encoding each plain run on
// its own and putting the special's id between them -- earliest special
// first, longest special at that position.

import XCTest
@testable import GGUFKit

final class SpecialTokenSplitTests: XCTestCase {
    /// Every byte as its own byte-level token, plus specials. No merges, so a
    /// plain run encodes to one id per UTF-8 byte.
    private func makeTokenizer(specials: [String]) throws -> BPETokenizer {
        var tokens = BPETokenizer.byteToUnicode().map { String(Character($0.1)) }
        var types = [Double](repeating: 1, count: tokens.count)
        for s in specials { tokens.append(s); types.append(3) }
        return try BPETokenizer(tokens: tokens, merges: [], tokenTypes: types,
                                dialect: .gpt2)
    }

    func testSplitEqualsPerSegmentEncoding() throws {
        let t = try makeTokenizer(specials: ["<s>", "<t>"])
        let s = t.id(of: "<s>")!, u = t.id(of: "<t>")!
        XCTAssertEqual(t.encode("ab<s>cd"),
                       t.encode("ab") + [s] + t.encode("cd"))
        XCTAssertEqual(t.encode("<s>"), [s])                       // only
        XCTAssertEqual(t.encode("<s>x"), [s] + t.encode("x"))       // at start
        XCTAssertEqual(t.encode("x<t>"), t.encode("x") + [u])       // at end
        XCTAssertEqual(t.encode("<s><t><s>"), [s, u, s])            // adjacent
        XCTAssertEqual(t.encode("z<t>y<s>w"),
                       t.encode("z") + [u] + t.encode("y") + [s] + t.encode("w"))
    }

    /// Two specials start at the same byte: the longer one wins, wherever the
    /// shorter one sits in the vocabulary.
    func testLongestSpecialWinsAtAPosition() throws {
        for order in [["<s>", "<s>long"], ["<s>long", "<s>"]] {
            let t = try makeTokenizer(specials: order)
            let short = t.id(of: "<s>")!, long = t.id(of: "<s>long")!
            XCTAssertEqual(t.encode("a<s>longb"),
                           t.encode("a") + [long] + t.encode("b"))
            XCTAssertEqual(t.encode("a<s>lonb"),
                           t.encode("a") + [short] + t.encode("lonb"))
        }
    }

    func testMultiByteTextAroundSpecials() throws {
        let t = try makeTokenizer(specials: ["<s>"])
        let s = t.id(of: "<s>")!
        XCTAssertEqual(t.encode("日本<s>語🙂"),
                       t.encode("日本") + [s] + t.encode("語🙂"))
    }

    func testTextWithoutSpecialsIsUntouched() throws {
        let plain = try makeTokenizer(specials: [])
        let withSpecials = try makeTokenizer(specials: ["<s>", "<t>"])
        let text = "no markers here, just text: <s and t> and 日本語"
        XCTAssertEqual(withSpecials.encode(text), plain.encode(text))
        XCTAssertEqual(plain.encode(text).count, Array(text.utf8).count)
    }

    /// An incomplete special is plain text.
    func testPartialSpecialIsNotMatched() throws {
        let t = try makeTokenizer(specials: ["<s>"])
        XCTAssertEqual(t.encode("x<s"), t.encode("x") + t.encode("<s"))
        XCTAssertFalse(t.encode("x<s").contains(t.id(of: "<s>")!))
    }

    /// A special followed by a combining mark still matches: specials are
    /// literal byte strings (as in HF), not grapheme clusters.
    func testSpecialFollowedByCombiningMark() throws {
        let t = try makeTokenizer(specials: ["<s>"])
        let s = t.id(of: "<s>")!
        XCTAssertEqual(t.encode("<s>\u{0301}a"), [s] + t.encode("\u{0301}a"))
    }

    /// Hundreds of markers in one prompt: the shape of an agent request.
    /// Linear now; the old scan re-searched the remainder for every special
    /// after every match.
    func testManyMarkersStayLinear() throws {
        let t = try makeTokenizer(specials: ["<s>", "<t>"])
        let s = t.id(of: "<s>")!
        let chunk = "{\"name\": \"tool\"}<s>"
        let text = String(repeating: chunk, count: 2_000)
        let ids = t.encode(text)
        XCTAssertEqual(ids.filter { $0 == s }.count, 2_000)
        XCTAssertEqual(ids.count, 2_000 * (Array("{\"name\": \"tool\"}".utf8).count + 1))
    }
}
