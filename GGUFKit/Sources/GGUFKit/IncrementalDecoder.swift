// Copyright (c) 2026 Jiejing Zhang.
//
// Turning a stream of token ids into text you can show as it arrives.
//
// A byte-level BPE token is a slice of BYTES. One Chinese character is three
// bytes and routinely spans two tokens, so decoding each token on its own and
// concatenating the strings produces U+FFFD at every split -- garbled output
// manufactured by the client, which looks exactly like garbled output from the
// engine and would be debugged as such.
//
// So this decodes to bytes, emits only the complete UTF-8 sequences, and holds
// the partial tail until the bytes that finish it arrive.

import Foundation

public final class IncrementalDecoder {
    private let tokenizer: any TokenizingVocabulary
    private let skipSpecialTokens: Bool
    private let preserving: Set<Int>
    private var pending: [UInt8] = []
    private var awaitingFirstByte: Bool

    public init(tokenizer: any TokenizingVocabulary, skipSpecialTokens: Bool = true,
                preserving: Set<Int> = []) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.preserving = preserving
        self.awaitingFirstByte = tokenizer.stripsLeadingSpaceOnFirstToken
    }

    /// Text that is complete as of these ids. May be empty when the ids only
    /// carried the first bytes of a character -- that is normal, not a stall.
    public func feed(_ ids: [Int]) -> String {
        guard !ids.isEmpty else { return "" }
        var bytes = tokenizer.decodeBytes(ids,
                                          skipSpecialTokens: skipSpecialTokens,
                                          preserving: preserving)
        if awaitingFirstByte, let first = bytes.first {
            // Exactly one space, and only the very first one produced by the
            // stream: a model that legitimately begins with two spaces keeps
            // the second, and a stream whose first token decoded to nothing
            // (a skipped special) has not spent its chance yet.
            if first == UInt8(ascii: " ") { bytes.removeFirst() }
            awaitingFirstByte = false
        }
        pending += bytes
        let cut = Self.completePrefixLength(pending)
        guard cut > 0 else { return "" }
        let out = String(decoding: pending[0..<cut], as: UTF8.self)
        pending.removeFirst(cut)
        return out
    }

    /// Whatever is left at end of stream. A well-formed stream ends empty; if
    /// it does not, the trailing bytes are genuinely truncated and rendering
    /// them as U+FFFD is the honest outcome rather than dropping them.
    public func flush() -> String {
        defer { pending.removeAll() }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }

    public func reset() { pending.removeAll() }

    /// Length of the longest prefix that contains only whole UTF-8 sequences.
    ///
    /// Scans back at most three continuation bytes for the lead byte of a
    /// possibly-unfinished sequence. Anything longer than that is malformed
    /// rather than incomplete, and is left in place to be emitted as U+FFFD --
    /// holding it back forever would stall the stream on a bad byte.
    static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        let n = bytes.count
        guard n > 0 else { return 0 }
        var i = n - 1
        var scanned = 0
        while i > 0, bytes[i] & 0b1100_0000 == 0b1000_0000, scanned < 3 {
            i -= 1
            scanned += 1
        }
        let lead = bytes[i]
        let need: Int
        if lead & 0b1000_0000 == 0 { need = 1 }
        else if lead & 0b1110_0000 == 0b1100_0000 { need = 2 }
        else if lead & 0b1111_0000 == 0b1110_0000 { need = 3 }
        else if lead & 0b1111_1000 == 0b1111_0000 { need = 4 }
        else { return n }  // stray continuation byte: not our problem to fix
        return (n - i) < need ? i : n
    }
}
