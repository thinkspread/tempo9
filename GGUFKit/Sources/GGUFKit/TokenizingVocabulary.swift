// Copyright (c) 2026 Jiejing Zhang.
//
// The surface a session needs from a tokenizer, so that more than one
// implementation can supply it.
//
// There are two, and they live in different languages on purpose. Byte-level
// BPE (`gpt2`, `gemma4`) is here in Swift because its pre-tokenizer is a
// unicode regex. SentencePiece is inside the engine, because it has no regex,
// because a GGUF served on a platform without a HuggingFace repo beside it has
// no other tokenizer at all, and because the tokenizer is part of the engine's
// text-in/text-out contract -- a host that re-derives it can be subtly wrong,
// and subtly wrong tokenization reads as a bad model, not as an error.

import Foundation

public protocol TokenizingVocabulary: AnyObject {
    var dialect: TokenizerDialect { get }
    var eosTokenID: Int? { get }
    /// Every id that ends a generation, not just the file's single `eos`.
    var stopTokenIDs: [Int] { get }

    func encode(_ text: String, addSpecialTokens: Bool) -> [Int]
    func id(of token: String) -> Int?

    /// Bytes rather than a String: a byte-fallback token can be half of a
    /// UTF-8 sequence, so the caller reassembles across steps.
    func decodeBytes(_ ids: [Int], skipSpecialTokens: Bool,
                     preserving: Set<Int>) -> [UInt8]

    /// SentencePiece prepends a space before encoding so that a first word
    /// tokenizes like a mid-sentence one; that space is an artifact of the
    /// encoder, not content, and the decoder drops it from the first token
    /// of a sequence. Whole-sequence decoding can do this itself -- a
    /// streaming decoder cannot, because it never sees position 0 as such.
    var stripsLeadingSpaceOnFirstToken: Bool { get }

    /// True for a vocabulary carrying GPT-OSS's harmony markers
    /// (<|channel|>/<|message|>/<|end|>); the caller picks the harmony
    /// splitter instead of the tag-pair one.
    var isHarmony: Bool { get }

    /// The special-token texts decoding must preserve, because something
    /// downstream parses them. Defaults to the dialect's list; a harmony
    /// vocabulary substitutes its own markers.
    var preservedTagTexts: [String] { get }
}

public extension TokenizingVocabulary {
    var stripsLeadingSpaceOnFirstToken: Bool { false }
    var isHarmony: Bool { false }
    var preservedTagTexts: [String] { dialect.preservedTagTexts }
}

extension BPETokenizer: TokenizingVocabulary {}
