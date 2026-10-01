// Copyright (c) 2026 Jiejing Zhang.
// The pre-tokenizer regexes are transcribed from their upstream definitions:
// tiktoken (MIT, Copyright (c) 2022 OpenAI) and the Llama 3 / Tekken
// tokenizer configs. See NOTICE.
//
// Byte-level BPE, built from what the .gguf already carries.
//
// This is the piece a host cannot cache its way around. A model graph is
// derived once per file; tokenization happens on every request, so as long as
// it lives in Python the process does too. The vocabulary, the merge list and
// the special-token types are all in the GGUF, so nothing here needs the
// original HuggingFace directory.
//
// Three stages, in the order the HF pipeline applies them:
//   1. split off special tokens literally (they must never be BPE'd)
//   2. pre-tokenize with the model's regex
//   3. byte-level map, then merge by rank
//
// Correctness here is all-or-nothing and silent when wrong: a mismatched
// pre-tokenizer regex still produces *valid* tokens, just different ones, and
// the model answers slightly worse for reasons nothing reports. Hence
// check_gguf_tokenizer_parity.py.

import Foundation

public struct TokenizerConfig {
    /// GPT-2 style pre-tokenizer regex. Qwen2 through Qwen3.5 share this one.
    public static let qwenPattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
        + "|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+"
        + "|\\s+(?!\\S)|\\s+"

    /// Llama 3's regex (tokenizer.ggml.pre "llama-bpe"). One visible
    /// difference from qwenPattern and it is not cosmetic: digits group in
    /// THREES ("2024" -> "202"+"4"), where Qwen splits every digit alone.
    /// Both are valid id sequences over the same vocab, which is exactly why
    /// nothing crashed while every number in a llama-arch prompt tokenized
    /// unlike anything the model was trained on.
    public static let llama3Pattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
        + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+"
        + "|\\s+(?!\\S)|\\s+"

    /// tiktoken o200k_base (tokenizer.ggml.pre "gpt-4o"; GPT-OSS). Verbatim
    /// from tiktoken -- ICU takes the original pattern, so unlike llama.cpp
    /// there is no lookahead re-encoding of the case-aware letter runs.
    /// Digits group in threes, contractions attach to their word (with case
    /// folding), and the punctuation run swallows trailing '/' (an o200k
    /// quirk the others do not have).
    public static let o200kPattern =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
        + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+"
        + "|\\s+(?!\\S)|\\s+"

    /// Mistral's tekken (tokenizer.ggml.pre "tekken"; Ministral 3). o200k's
    /// case-aware letter runs but NO contraction branch ("it's" ->
    /// "it"+"'"+"s") and single-digit numbers like Qwen -- which is why the
    /// digit probe alone did not catch this model on the wrong pattern.
    public static let tekkenPattern =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
        + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
        + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
        + "|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+"
        + "|\\s+(?!\\S)|\\s+"

    /// Pattern for a `tokenizer.ggml.pre` value. Refuses unknown dialects by
    /// name, mirroring TokenizerDialect.from and the engine's
    /// te9_tokenizer_open: a guessed regex still produces valid ids, just
    /// from the wrong distribution, and nothing downstream reports a thing.
    /// (nil is allowed -- gemma4 writes no `pre` and does not regex-split.)
    public static func pattern(forPretokenizer pre: String?) throws -> String {
        switch pre ?? "" {
        case "", "default", "gpt-2", "qwen2", "qwen35":
            return qwenPattern
        case "llama3", "llama-bpe":
            return llama3Pattern
        case "gpt-4o":
            return o200kPattern
        case "tekken":
            return tekkenPattern
        default:
            throw GGUFError.missingKey(
                "unsupported tokenizer.ggml.pre '\(pre ?? "")' -- this build "
                + "implements qwen2/qwen35/gpt-2/default, llama-bpe/llama3, "
                + "gpt-4o, and tekken")
        }
    }
}

/// Unicode normalization applied before anything else.
///
/// The GGUF does **not** carry this -- it is in tokenizer.json's `normalizer`
/// field, which llama.cpp does not import -- so it has to be inferred from
/// `tokenizer.ggml.pre`. Qwen's normalizer is NFC, and skipping it is not a
/// rounding error: decomposed text ("cafe" + U+0301) tokenizes as three
/// tokens instead of two, differently from every training example, and
/// nothing downstream reports a thing.
public enum Normalization: String {
    case none
    case nfc
    case nfkc

    func apply(_ text: String) -> String {
        switch self {
        case .none: return text
        case .nfc: return text.precomposedStringWithCanonicalMapping
        case .nfkc: return text.precomposedStringWithCompatibilityMapping
        }
    }

    /// What a given `tokenizer.ggml.pre` implies. NFC is QWEN's normalizer,
    /// not a universal default: llama3, o200k (gpt-4o), and tekken all ship
    /// `normalizer: null` in their tokenizer.json, and llama.cpp normalizes
    /// nothing for them. Applying NFC anyway merged "e"+U+0301 into one
    /// codepoint and produced ids the reference never produces -- caught by
    /// the combining-mark cases in check_gguf_tokenizer_parity.py, three
    /// dialects at once.
    public static func forPretokenizer(_ pre: String?) -> Normalization {
        switch pre ?? "" {
        case "llama3", "llama-bpe", "gpt-4o", "tekken":
            return .none
        default:
            return .nfc
        }
    }
}

/// Which BPE dialect a .gguf speaks, from `tokenizer.ggml.model`.
///
/// The two that ship differ before BPE even starts, and the difference is
/// invisible afterwards -- both produce valid ids, just different ones:
///
///   gpt2      bytes are remapped into a printable alphabet (0x20 -> "G-dot"),
///             the text is first split by a GPT-2 regex, and each piece is
///             BPE'd on its own.
///   gemma4    a space is rewritten to U+2581 and nothing else is; there is
///             NO regex split, so BPE runs over the whole segment at once,
///             and a character outside the vocabulary falls back to its
///             UTF-8 bytes as <0xNN> tokens.
///
/// Reading a Gemma vocabulary with the gpt2 dialect is not a near miss. Its
/// vocabulary has no "G-dot" word-boundary token, so every space in the
/// prompt resolves to some unrelated id, and the model answers with fluent
/// nonsense -- which is exactly how this was found.
public enum TokenizerDialect: String, Sendable {
    case gpt2
    case gemma4
    /// SentencePiece. Tokenized by the engine, not by this file -- the case
    /// exists so a session can name what it is holding.
    case spm

    /// How this family marks a block of reasoning.
    ///
    /// Qwen3.5 opens `<think>` and closes `</think>`, and its template leaves
    /// the opener OPEN in the generation prompt when thinking is on, so the
    /// output begins inside the block.
    ///
    /// Gemma 4 is the other way round in both halves. The markers are
    /// `<|channel>thought` and `<channel|>`, and thinking-ON is the case
    /// where the template leaves NOTHING open -- the model emits its own
    /// opener. Thinking-OFF is the one that appends a pre-closed empty block.
    /// Reading Qwen's rule onto it leaks the literal word "thought" into the
    /// answer and files the reasoning as content.
    public var reasoningTags: (open: String, close: String) {
        switch self {
        case .gpt2: return ("<think>", "</think>")
        // `<|channel>` alone, not `<|channel>thought`: the channel NAME is
        // ordinary text after the marker token, so a model that puts a
        // newline or a space between them would leave the longer form
        // unmatched — and it does. Requiring the exact concatenation leaked
        // the literal word "thought" into the top of replies. The name goes
        // to reasoning, where it belongs and where nobody reads it.
        case .gemma4: return ("<|channel>", "<channel|>")
        // The SentencePiece-era models predate reasoning channels; Mistral
        // v0.3 and Llama 2 emit no thinking tags at all.
        case .spm: return ("", "")
        }
    }

    /// The special tokens whose TEXT must survive decoding, because
    /// something downstream parses it. Everything else special is stripped.
    public var preservedTagTexts: [String] {
        switch self {
        case .gpt2:
            return ["<think>", "</think>", "<tool_call>", "</tool_call>"]
        // `<|channel>` rather than `<|channel>thought`: the marker is the
        // token, "thought" is ordinary text after it, and asking the
        // tokenizer for an id it does not have would keep neither.
        case .gemma4:
            return ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>"]
        case .spm: return []
        }
    }

    static func from(_ model: String?) throws -> TokenizerDialect {
        switch model ?? "gpt2" {
        // llama.cpp writes "gpt2" for every byte-level BPE vocabulary,
        // whatever the architecture.
        case "gpt2": return .gpt2
        case "gemma4": return .gemma4
        default:
            // Loudly, on purpose. Guessing gpt2 is what produced the fluent
            // nonsense above, and a refusal at load names the file.
            throw GGUFError.missingKey(
                "unsupported tokenizer.ggml.model '\(model ?? "")' -- this "
                + "build implements gpt2 and gemma4")
        }
    }
}

public final class BPETokenizer {
    public let vocabulary: [String]
    public private(set) var eosTokenID: Int?
    public private(set) var bosTokenID: Int?
    public private(set) var addBOS: Bool = false
    /// Every token that ends a generation, not just the .gguf's `eos`.
    ///
    /// The .gguf carries ONE eos id, and for Gemma 4 it is `<eos>`(1) — a
    /// pretraining marker the chat template never emits. A turn ends with
    /// `<turn|>`(106), and a tool call pauses at `<|tool_response>`(50);
    /// generation_config.json lists all three, and generation_config.json is
    /// exactly the file a GGUF-only host does not have. Stopping on the
    /// declared eos alone means never stopping: the model finishes its
    /// answer, opens a new turn and answers again, forever.
    ///
    /// Resolved by NAME from the vocabulary, so a model without them is
    /// unaffected and nothing is hard-coded to an id that another build of
    /// the same family might renumber.
    public private(set) var stopTokenIDs: [Int] = []
    /// GPT-OSS's harmony format: blocks framed by <|channel|>/<|message|>/
    /// <|end|>, where the channel NAME picks reasoning vs answer. Detected
    /// from the vocabulary, not the architecture -- the marker tokens are
    /// the thing the splitter needs to exist.
    public private(set) var isHarmony = false

    /// The tag texts decoding must preserve for THIS vocabulary. The
    /// dialect's list, unless the vocab is harmony -- harmony's markers are
    /// per-format, not per-dialect (GPT-OSS is dialect gpt2, like Qwen, and
    /// Qwen's <think> list would preserve nothing harmony emits).
    public var preservedTagTexts: [String] {
        isHarmony
            ? ["<|channel|>", "<|message|>", "<|start|>", "<|end|>",
               "<|call|>", "<|return|>"]
            : dialect.preservedTagTexts
    }

    private var idOf: [String: Int] = [:]
    private var mergeRank: [Pair: Int] = [:]
    private var specials: [(text: String, id: Int)] = []
    /// The same specials as UTF-8, bucketed by their first byte and longest
    /// first within a bucket -- what encodeSegments actually scans with.
    private var specialsByFirstByte: [[(bytes: [UInt8], id: Int)]] =
        Array(repeating: [], count: 256)
    private let regex: NSRegularExpression
    private let byteEncoder: [UInt8: Character]
    private let byteDecoder: [Character: UInt8]
    private let normalization: Normalization
    public let dialect: TokenizerDialect
    /// `<0x41>` -> 0x41, for gemma4's byte fallback. Empty for gpt2, which
    /// has no unmappable character by construction.
    private var byteTokenOf: [UInt8: Int] = [:]

    struct Pair: Hashable { let left: String; let right: String }

    // MARK: - construction

    public convenience init(gguf: GGUFFile) throws {
        guard let tokens = gguf.kv["tokenizer.ggml.tokens"]?.stringsValue,
              !tokens.isEmpty else {
            throw GGUFError.missingKey("tokenizer.ggml.tokens (re-read the "
                                       + "file with readArrays: true)")
        }
        let merges = gguf.kv["tokenizer.ggml.merges"]?.stringsValue ?? []
        let types = gguf.kv["tokenizer.ggml.token_type"]?.numbersValue ?? []
        let pre = gguf.kv["tokenizer.ggml.pre"]?.stringValue
        let dialect = try TokenizerDialect.from(
            gguf.kv["tokenizer.ggml.model"]?.stringValue)
        // gemma4's only normalizer is the space rewrite, which happens
        // inside encode. Applying NFC on top would be a second, unasked-for
        // transformation -- and tokenizer.json says the sequence is exactly
        // one Replace.
        try self.init(tokens: tokens, merges: merges, tokenTypes: types,
                      dialect: dialect,
                      pattern: dialect == .gemma4
                          ? TokenizerConfig.qwenPattern
                          : TokenizerConfig.pattern(forPretokenizer: pre),
                      normalization: dialect == .gemma4
                          ? .none : .forPretokenizer(pre))

        eosTokenID = gguf.kv["tokenizer.ggml.eos_token_id"]?.intValue
        bosTokenID = gguf.kv["tokenizer.ggml.bos_token_id"]?.intValue
        addBOS = gguf.kv["tokenizer.ggml.add_bos_token"]?.boolValue ?? false
        isHarmony = pre == "gpt-4o" && id(of: "<|channel|>") != nil

        var stops = eosTokenID.map { [$0] } ?? []
        for name in Self.endOfTurnNames(for: dialect) {
            if let id = id(of: name), !stops.contains(id) { stops.append(id) }
        }
        stopTokenIDs = stops
    }

    public init(tokens: [String], merges: [String], tokenTypes: [Double],
                dialect: TokenizerDialect = .gpt2,
                pattern: String = TokenizerConfig.qwenPattern,
                normalization: Normalization = .nfc) throws {
        self.vocabulary = tokens
        self.regex = try NSRegularExpression(pattern: pattern)
        self.normalization = normalization
        self.dialect = dialect

        var encoder = [UInt8: Character]()
        var decoder = [Character: UInt8]()
        for (byte, scalar) in Self.byteToUnicode() {
            let ch = Character(scalar)
            encoder[byte] = ch
            decoder[ch] = byte
        }
        self.byteEncoder = encoder
        self.byteDecoder = decoder

        idOf.reserveCapacity(tokens.count)
        for (index, token) in tokens.enumerated() where idOf[token] == nil {
            idOf[token] = index
        }

        mergeRank.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            // "A B"; the pieces themselves never contain a space, because
            // byte-level mapping moves 0x20 out of ASCII space.
            guard let split = merge.firstIndex(of: " ") else { continue }
            let pair = Pair(left: String(merge[merge.startIndex..<split]),
                            right: String(merge[merge.index(after: split)...]))
            if mergeRank[pair] == nil { mergeRank[pair] = rank }
        }

        // GGUF token types: 1 normal, 2 unknown, 3 control, 4 user-defined,
        // 5 unused, 6 byte. Control and user-defined are matched literally.
        for (index, type) in tokenTypes.enumerated() where index < tokens.count {
            if type == 3 || type == 4 {
                specials.append((tokens[index], index))
            }
        }
        // Longest first, so <|im_start|> wins over any shorter prefix.
        specials.sort { $0.text.count > $1.text.count }
        for special in specials {
            let bytes = Array(special.text.utf8)
            guard let first = bytes.first else { continue }
            specialsByFirstByte[Int(first)].append((bytes, special.id))
        }
        for b in 0..<256 {
            specialsByFirstByte[b].sort { $0.bytes.count > $1.bytes.count }
        }

        if dialect == .gemma4 {
            // Byte-fallback tokens, by their literal "<0xNN>" spelling.
            // Type 6 is GGUF's BYTE class, but the spelling is what has to
            // round-trip, so match on it and let the type disagree.
            for (index, token) in tokens.enumerated()
            where token.count == 6 && token.hasPrefix("<0x")
                    && token.hasSuffix(">") {
                let hex = token.dropFirst(3).dropLast()
                if let b = UInt8(hex, radix: 16), byteTokenOf[b] == nil {
                    byteTokenOf[b] = index
                }
            }
        }
    }

    private static func endOfTurnNames(for dialect: TokenizerDialect) -> [String] {
        switch dialect {
        // Qwen's .gguf already declares <|im_end|> as its eos, so there is
        // nothing to add — listing it anyway would be harmless but would
        // suggest the .gguf could not be trusted, and here it can.
        case .gpt2: return []
        case .gemma4: return ["<turn|>", "<|tool_response>"]
        // </s> is the file's own eos and is already listed.
        case .spm: return []
        }
    }

    // MARK: - encode / decode

    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        var out = [Int]()
        if addSpecialTokens, addBOS, let bos = bosTokenID { out.append(bos) }
        // Normalize once, at the top: doing it per pre-token would let a
        // combining mark that got split across pieces escape composition.
        encodeSegments(normalization.apply(text), into: &out)
        return out
    }

    /// Split on special tokens, BPE everything between them.
    ///
    /// One left-to-right pass over the UTF-8 bytes. At each byte that some
    /// special starts with, the candidates sharing that first byte are tried
    /// longest first; the first position with a match is the earliest special
    /// and the longest one there wins -- the same choice the old loop made.
    ///
    /// The old loop asked Foundation's `range(of:)` for EVERY special after
    /// EVERY match, each call scanning the rest of the text (to the end, when
    /// that special was absent) with grapheme-aware comparison. On a
    /// SiliconBench agent prompt (4.3k tokens, hundreds of markers) that was
    /// 95% of encode and ~55 ms per request -- more than the prefill of a
    /// short prompt. It replaced a recursive version that overflowed the
    /// cooperative thread pool's stack on agent prompts; this one keeps the
    /// fix (no recursion, no per-match copy of the remainder).
    ///
    /// Bytes, not Characters, is also what HF does: a special matches its
    /// literal text even when a combining mark follows it, where a grapheme
    /// comparison would have refused. A first byte can only be a UTF-8 lead
    /// or ASCII byte, so a match never starts inside a multi-byte character.
    private func encodeSegments(_ text: String, into out: inout [Int]) {
        let utf8 = Array(text.utf8)
        let n = utf8.count
        var segmentStart = 0
        var i = 0
        func flush(_ end: Int) {
            if end > segmentStart {
                encodePlain(String(decoding: utf8[segmentStart..<end],
                                   as: UTF8.self), into: &out)
            }
        }
        utf8.withUnsafeBufferPointer { buf in
            while i < n {
                let candidates = specialsByFirstByte[Int(buf[i])]
                var matched: (length: Int, id: Int)?
                if !candidates.isEmpty {
                    for c in candidates where c.bytes.count <= n - i {
                        var equal = true
                        var k = 1
                        while k < c.bytes.count {
                            if buf[i + k] != c.bytes[k] { equal = false; break }
                            k += 1
                        }
                        if equal { matched = (c.bytes.count, c.id); break }
                    }
                }
                if let m = matched {
                    flush(i)
                    out.append(m.id)
                    i += m.length
                    segmentStart = i
                } else {
                    i += 1
                }
            }
        }
        flush(n)
    }

    private func encodePlain(_ text: String, into out: inout [Int]) {
        if dialect == .gemma4 {
            encodeSentencePiece(text, into: &out)
            return
        }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0,
                                                             length: ns.length))
        for match in matches {
            let piece = ns.substring(with: match.range)
            guard !piece.isEmpty else { continue }
            var mapped = ""
            for byte in Array(piece.utf8) {
                mapped.append(byteEncoder[byte] ?? "?")
            }
            for symbol in bpe(mapped) {
                if let id = idOf[symbol] {
                    out.append(id)
                }
                // A symbol missing from the vocabulary cannot happen with a
                // consistent vocab+merges pair (every single mapped byte is a
                // token), so dropping is the honest fallback: emitting an
                // <unk> the model was not trained to see would be worse.
            }
        }
    }

    /// gemma4: rewrite spaces, then BPE the WHOLE segment at once.
    ///
    /// There is no pre-tokenizer split to lean on -- tokenizer.json's Split
    /// on " " runs AFTER the normalizer has already removed every " ", so it
    /// matches nothing and the merge search legitimately spans the entire
    /// text. That is why this uses the heap merge below rather than the
    /// scan-for-best loop: at 515k merge rules and a few thousand
    /// characters, a rescan per merge is quadratic and turns a prompt into
    /// seconds of tokenization.
    private func encodeSentencePiece(_ text: String, into out: inout [Int]) {
        // Split on SCALARS, not Characters. Swift's Character is an extended
        // grapheme cluster, so "e" + U+0301 is one Character and a family
        // emoji is one Character -- and BPE would then look up a symbol the
        // vocabulary has never seen, sending whole emoji through byte
        // fallback and mis-splitting every combining mark. HF splits on
        // scalars, and this is a place the two must agree exactly.
        var symbols = [String]()
        symbols.reserveCapacity(text.unicodeScalars.count)
        for u in text.unicodeScalars {
            symbols.append(u == " " ? "\u{2581}" : String(u))
        }
        for symbol in bpeHeap(symbols) {
            if let id = idOf[symbol] {
                out.append(id)
            } else {
                // byte_fallback: a character the vocabulary does not have
                // becomes its UTF-8 bytes. Dropping it — what the gpt2 path
                // does, where it cannot happen — would silently delete text.
                for b in Array(symbol.utf8) {
                    if let id = byteTokenOf[b] { out.append(id) }
                }
            }
        }
    }

    /// BPE by lowest merge rank, with a heap instead of a rescan.
    ///
    /// Symbols live in a doubly-linked list over an array so a merge is O(1);
    /// candidates are pushed for the neighbours a merge creates. A popped
    /// candidate is checked against the CURRENT texts of its two symbols,
    /// which is what makes stale entries harmless — cheaper than trying to
    /// delete them from the heap.
    private func bpeHeap(_ symbols: [String]) -> [String] {
        var text = symbols
        if text.count < 2 { return text }
        let n = text.count
        var prev = Array(-1..<(n - 1))          // prev[0] = -1
        var next = Array(1...n)                 // next[n-1] = n (= none)
        var alive = [Bool](repeating: true, count: n)

        struct Candidate: Comparable {
            let rank: Int, left: Int, right: Int
            let leftText: String, rightText: String
            static func < (a: Candidate, b: Candidate) -> Bool {
                a.rank != b.rank ? a.rank < b.rank : a.left < b.left
            }
            static func == (a: Candidate, b: Candidate) -> Bool {
                a.rank == b.rank && a.left == b.left
            }
        }
        var heap = [Candidate]()
        heap.reserveCapacity(n)

        func push(_ l: Int, _ r: Int) {
            guard l >= 0, r < n, alive[l], alive[r] else { return }
            guard let rank = mergeRank[Pair(left: text[l], right: text[r])]
            else { return }
            let c = Candidate(rank: rank, left: l, right: r,
                              leftText: text[l], rightText: text[r])
            heap.append(c)
            var i = heap.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                if heap[i] < heap[parent] {
                    heap.swapAt(i, parent); i = parent
                } else { break }
            }
        }
        func pop() -> Candidate? {
            guard !heap.isEmpty else { return nil }
            let top = heap[0]
            heap[0] = heap[heap.count - 1]
            heap.removeLast()
            var i = 0
            while true {
                let l = 2 * i + 1, r = 2 * i + 2
                var best = i
                if l < heap.count, heap[l] < heap[best] { best = l }
                if r < heap.count, heap[r] < heap[best] { best = r }
                if best == i { break }
                heap.swapAt(i, best); i = best
            }
            return top
        }

        for i in 0..<(n - 1) { push(i, i + 1) }

        while let c = pop() {
            // Stale? Either symbol may have been merged away, or grown.
            guard alive[c.left], alive[c.right],
                  next[c.left] == c.right,
                  text[c.left] == c.leftText, text[c.right] == c.rightText
            else { continue }
            text[c.left] = c.leftText + c.rightText
            alive[c.right] = false
            let after = next[c.right]
            next[c.left] = after
            if after < n { prev[after] = c.left }
            push(prev[c.left], c.left)
            push(c.left, after)
        }
        return (0..<n).filter { alive[$0] }.map { text[$0] }
    }

    private func bpe(_ word: String) -> [String] {
        var symbols = word.map(String.init)
        if symbols.count < 2 { return symbols }

        while true {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(symbols.count - 1) {
                let pair = Pair(left: symbols[i], right: symbols[i + 1])
                if let rank = mergeRank[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
            if symbols.count == 1 { break }
        }
        return symbols
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        String(decoding: decodeBytes(ids, skipSpecialTokens: skipSpecialTokens),
               as: UTF8.self)
    }

    /// The raw bytes, before they are interpreted as UTF-8.
    ///
    /// Streaming needs this. A byte-level BPE token is a slice of bytes, not
    /// of characters, so one Chinese character (3 bytes) routinely spans two
    /// tokens. Decoding a token at a time through `decode` would turn every
    /// such split into U+FFFD -- garbled output produced by the client, which
    /// would be indistinguishable from garbled output produced by the engine.
    /// `IncrementalDecoder` holds the partial tail back instead.
    /// `preserving` are special tokens to emit as text even when
    /// `skipSpecialTokens` is on. Streaming needs it: `</think>` IS a special
    /// token, so skipping every special silently removes the only marker that
    /// says where reasoning ends, and the whole answer stays classified as
    /// reasoning forever.
    public func decodeBytes(_ ids: [Int],
                            skipSpecialTokens: Bool = true,
                            preserving: Set<Int> = []) -> [UInt8] {
        var bytes = [UInt8]()
        let specialIDs = Set(specials.map(\.id)).subtracting(preserving)
        for id in ids {
            guard id >= 0, id < vocabulary.count else { continue }
            if skipSpecialTokens, specialIDs.contains(id) { continue }
            let token = vocabulary[id]
            if specialIDs.contains(id) {
                bytes.append(contentsOf: Array(token.utf8))
                continue
            }
            if dialect == .gemma4 {
                // The mirror of encode: U+2581 back to a space, <0xNN> back
                // to the raw byte, everything else through as its own UTF-8.
                // The bytes of a fallback run then reassemble into whatever
                // character they came from -- which is why this returns bytes
                // and lets IncrementalDecoder hold a split tail.
                if token.count == 6, token.hasPrefix("<0x"),
                   token.hasSuffix(">"),
                   let b = UInt8(token.dropFirst(3).dropLast(), radix: 16) {
                    bytes.append(b)
                    continue
                }
                for ch in token {
                    if ch == "\u{2581}" {
                        bytes.append(0x20)
                    } else {
                        bytes.append(contentsOf: Array(String(ch).utf8))
                    }
                }
                continue
            }
            for ch in token {
                if let byte = byteDecoder[ch] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: Array(String(ch).utf8))
                }
            }
        }
        return bytes
    }

    public func id(of token: String) -> Int? { idOf[token] }

    // MARK: - GPT-2 byte <-> unicode

    /// The byte-level trick: map all 256 bytes to printable code points so BPE
    /// operates on text, never on raw bytes.
    static func byteToUnicode() -> [(UInt8, Unicode.Scalar)] {
        var bs = [UInt32]()
        bs.append(contentsOf: UInt32(33)...UInt32(126))
        bs.append(contentsOf: UInt32(161)...UInt32(172))
        bs.append(contentsOf: UInt32(174)...UInt32(255))
        var cs = bs
        var n: UInt32 = 0
        for b in UInt32(0)...UInt32(255) where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        return zip(bs, cs).map { (UInt8($0), Unicode.Scalar($1)!) }
    }
}
