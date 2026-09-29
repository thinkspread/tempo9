// Copyright (c) 2026 Jiejing Zhang.
//
// The SentencePiece tokenizer that lives inside the engine, wrapped so a
// session can hold it exactly like the Swift byte-level BPE one.
//
// Nothing here implements tokenization. That is deliberate: the ids a model
// was trained on are part of the engine's contract, and a second
// implementation of them in the host is a second chance to be subtly wrong in
// a way that reads as a bad model rather than as an error.

import Foundation
import GGUFKit
import CTempo9Engine

public enum EngineTokenizerError: Error, CustomStringConvertible {
    case open(String)
    case encode(String)

    public var description: String {
        switch self {
        case .open(let m): return "engine tokenizer: \(m)"
        case .encode(let m): return "engine tokenizer encode: \(m)"
        }
    }
}

/// Whether the engine, rather than GGUFKit, should tokenize this file.
public func engineOwnsTokenizer(model: String?) -> Bool {
    // Only SentencePiece so far. `gpt2` and `gemma4` stay in Swift because
    // their pre-tokenizer is a unicode regex, which is the expensive half to
    // port and the half with no correctness pressure to move.
    return model == "llama"
}

public final class EngineTokenizer: TokenizingVocabulary, @unchecked Sendable {
    public let dialect: TokenizerDialect = .spm
    public let stripsLeadingSpaceOnFirstToken = true
    public let eosTokenID: Int?
    public let stopTokenIDs: [Int]
    public let chatTemplate: String

    private let handle: te9_tokenizer_t
    private let vocabulary: [String]
    private let idOf: [String: Int]
    private let specialIDs: Set<Int>

    public init(ggufPath: String, vocabulary: [String],
                specialIDs: Set<Int>) throws {
        var h: te9_tokenizer_t? = nil
        let rc = te9_tokenizer_open(ggufPath, &h)
        guard rc == TE9_OK, let h else {
            throw EngineTokenizerError.open(String(cString: te9_last_error()))
        }
        self.handle = h
        self.vocabulary = vocabulary
        self.specialIDs = specialIDs
        var map = [String: Int](minimumCapacity: vocabulary.count)
        for (i, t) in vocabulary.enumerated() where map[t] == nil { map[t] = i }
        self.idOf = map
        let eos = Int(te9_tokenizer_eos(h))
        self.eosTokenID = eos >= 0 ? eos : nil
        self.stopTokenIDs = eos >= 0 ? [eos] : []
        self.chatTemplate = String(cString: te9_tokenizer_chat_template(h))
    }

    deinit { te9_tokenizer_close(handle) }

    public func id(of token: String) -> Int? { idOf[token] }

    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        var needed = 0
        // Ask for the count first: the engine reports what it needed even
        // when the buffer was too small, so one sized retry always suffices.
        _ = te9_tokenizer_encode(handle, text, addSpecialTokens ? 1 : 0,
                                nil, 0, &needed)
        if needed == 0 { return [] }
        var ids = [Int32](repeating: 0, count: needed)
        var got = 0
        let rc = ids.withUnsafeMutableBufferPointer { buf in
            te9_tokenizer_encode(handle, text, addSpecialTokens ? 1 : 0,
                                buf.baseAddress, needed, &got)
        }
        guard rc == TE9_OK else { return [] }
        return ids.prefix(got).map(Int.init)
    }

    public func decodeBytes(_ ids: [Int], skipSpecialTokens: Bool = true,
                            preserving: Set<Int> = []) -> [UInt8] {
        var out = [UInt8]()
        let skip = specialIDs.subtracting(preserving)
        for id in ids {
            guard id >= 0 else { continue }
            if skipSpecialTokens, skip.contains(id) { continue }
            var needed = 0
            _ = te9_tokenizer_decode_one(handle, Int32(id), nil, 0, &needed)
            if needed == 0 { continue }
            var buf = [CChar](repeating: 0, count: needed)
            var got = 0
            let rc = buf.withUnsafeMutableBufferPointer {
                te9_tokenizer_decode_one(handle, Int32(id), $0.baseAddress,
                                        needed, &got)
            }
            guard rc == TE9_OK else { continue }
            out.append(contentsOf: buf.prefix(got).map { UInt8(bitPattern: $0) })
        }
        return out
    }
}
