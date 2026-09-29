// Copyright (c) 2026 Jiejing Zhang.
//
// The two host-side rules SentencePiece needs. Neither is about tokenizing --
// the ids themselves are checked per-token against llama.cpp by the engine's
// own gate -- but both change what the user reads.

import XCTest
@testable import GGUFKit
@testable import ChatTemplateKit

/// A tokenizer that emits fixed bytes, so the decoder's rule can be tested
/// without a vocabulary.
private final class StubVocab: TokenizingVocabulary {
    let dialect: TokenizerDialect = .spm
    let eosTokenID: Int? = nil
    let stopTokenIDs: [Int] = []
    let stripsLeadingSpaceOnFirstToken: Bool
    private let pieces: [Int: String]

    init(strips: Bool, pieces: [Int: String]) {
        self.stripsLeadingSpaceOnFirstToken = strips
        self.pieces = pieces
    }
    func id(of token: String) -> Int? { nil }
    func encode(_ text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decodeBytes(_ ids: [Int], skipSpecialTokens: Bool,
                     preserving: Set<Int>) -> [UInt8] {
        ids.flatMap { Array((pieces[$0] ?? "").utf8) }
    }
}

final class SPMHostRulesTests: XCTestCase {

    // MARK: - the streaming leading space

    /// SPM prepends a space before encoding so a first word tokenizes like a
    /// mid-sentence one. Whole-sequence decoding drops it; a streaming
    /// decoder never sees position 0 as such, so the rule lives here.
    func testFirstSpaceIsDroppedOnce() {
        let t = StubVocab(strips: true, pieces: [1: " Paris", 2: " is", 3: " nice"])
        let d = IncrementalDecoder(tokenizer: t)
        XCTAssertEqual(d.feed([1]), "Paris")
        // Only the first one: the spaces between words are content.
        XCTAssertEqual(d.feed([2]) + d.feed([3]), " is nice")
    }

    func testSecondLeadingSpaceSurvives() {
        // A model that genuinely begins with two spaces keeps the second.
        let t = StubVocab(strips: true, pieces: [1: "  indented"])
        XCTAssertEqual(IncrementalDecoder(tokenizer: t).feed([1]), " indented")
    }

    func testSkippedFirstTokenDoesNotSpendTheChance() {
        // A special that decodes to nothing must not consume the one strip,
        // or every answer after a BOS keeps its leading space.
        let t = StubVocab(strips: true, pieces: [9: "", 1: " Paris"])
        let d = IncrementalDecoder(tokenizer: t)
        XCTAssertEqual(d.feed([9]), "")
        XCTAssertEqual(d.feed([1]), "Paris")
    }

    func testByteLevelDialectIsUntouched() {
        let t = StubVocab(strips: false, pieces: [1: " Paris"])
        XCTAssertEqual(IncrementalDecoder(tokenizer: t).feed([1]), " Paris")
    }

    // MARK: - templates that reject a system role

    /// Mistral v0.3 and the Llama-2 line raise "Conversation roles must
    /// alternate"; every agent front end sends a system prompt, so without
    /// the fold those models are reachable by curl and not by an agent.
    func testSystemFoldsIntoFirstUser() throws {
        let tmpl = try ChatTemplate(
            source: "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}")
        let msgs: [[String: Any]] = [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "U"],
        ]
        // The template above accepts system, so nothing folds.
        XCTAssertEqual(try tmpl.render(messages: msgs, addGenerationPrompt: false),
                       "[system]S[user]U")

        let strict = try ChatTemplate(source: """
            {% for m in messages %}\
            {% if m.role != 'user' and m.role != 'assistant' %}\
            {{ raise_exception('roles must alternate') }}{% endif %}\
            [{{ m.role }}]{{ m.content }}{% endfor %}
            """)
        XCTAssertEqual(try strict.render(messages: msgs, addGenerationPrompt: false),
                       "[user]S\n\nU")
    }

    /// Multimodal part lists have no defined concatenation, so guessing one
    /// would corrupt the prompt rather than fail it.
    func testMultimodalContentIsNotFolded() throws {
        let strict = try ChatTemplate(source: """
            {% for m in messages %}\
            {% if m.role != 'user' %}{{ raise_exception('no system') }}{% endif %}\
            {{ m.content }}{% endfor %}
            """)
        let msgs: [[String: Any]] = [
            ["role": "system", "content": "S"],
            ["role": "user", "content": [["type": "text", "text": "U"]]],
        ]
        XCTAssertThrowsError(
            try strict.render(messages: msgs, addGenerationPrompt: false))
    }
}
