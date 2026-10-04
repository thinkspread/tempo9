// Copyright (c) 2026 Jiejing Zhang.
//
// The whole pipeline, in this process: template -> tokens -> engine ->
// text -> content/reasoning. The HTTP server did all of this; running it
// here is what removes Python from the shipped app.
//
// Deliberately shaped like an HTTP client's stream() so the two are
// interchangeable and can be compared answer-for-answer: a second inference
// path that has not been checked against the first is a way to introduce
// differences and then blame them on the engine.
//
// What lives where, and why none of it is here:
//   GGUFKit          tokenizer and the chat template, both read from the .gguf
//   ChatTemplateKit  Jinja rendering of that template
//   Tempo9     the engine, plus the incremental decoder and think split
// This file is only the wiring.

import Foundation
import GGUFKit
import ChatTemplateKit
import Tempo9Engine
import VisionTowerKit

/// Deltas and the final reply, owned here rather than borrowed from the app's
/// HTTP client, so this target can be exercised without the app.
/// One image, as the engine needs it described.
///
/// Its own type rather than VisionTowerKit's Encoding: a caller should be able
/// to feed embeddings from anywhere, and Encoding has no public initialiser
/// precisely because it is the tower's output, not an input format.
/// One utterance, already embedded by the tower.
///
/// `contentKey` is over the PCM, not the embedding: it keys the prefix cache,
/// so a wrong-but-stable value would serve a cached prefill for different
/// audio, and hashing 11 MB of embedding every request to avoid that would
/// cost more than it saves. The caller has the samples; it hashes those.
public struct AudioPlacement: Sendable {
    public var embedding: [Float]   // [tokens, hidden] fp32
    public var tokens: Int
    public var hidden: Int
    public var contentKey: Int64

    public init(embedding: [Float], tokens: Int, hidden: Int,
                contentKey: Int64) {
        self.embedding = embedding
        self.tokens = tokens
        self.hidden = hidden
        self.contentKey = contentKey
    }
}

public struct ImagePlacement: Sendable {
    public var embedding: [Float]   // [tokens, hidden] fp32
    public var tokens: Int
    public var hidden: Int
    public var gridH: Int
    public var gridW: Int
    public var contentKey: Int64
    /// Qwen3-VL deepstack features, LAYER-major
    /// [numDeepstackLayers, tokens, hidden] fp32; empty for models without
    /// them. Travels beside `embedding` because it covers the SAME token
    /// run -- the engine adds map i into decoder layer i's output at the
    /// image positions.
    public var deepstack: [Float] = []
    public var numDeepstackLayers: Int = 0

    public init(embedding: [Float], tokens: Int, hidden: Int,
                gridH: Int, gridW: Int, contentKey: Int64,
                deepstack: [Float] = [], numDeepstackLayers: Int = 0) {
        self.embedding = embedding
        self.tokens = tokens
        self.hidden = hidden
        self.gridH = gridH
        self.gridW = gridW
        self.contentKey = contentKey
        self.deepstack = deepstack
        self.numDeepstackLayers = numDeepstackLayers
    }
}

public struct LocalDelta: Sendable {
    public var content = ""
    public var reasoning = ""
    /// Tokens generated so far in this reply, thinking included. The UI
    /// cannot derive this from the text: reasoning is split out, and a
    /// character count is not a token count in any language, least of all
    /// Chinese.
    public var tokens = 0
    public var isEmpty: Bool { content.isEmpty && reasoning.isEmpty }
}

public struct LocalReply: Sendable {
    public var text: String
    public var reasoning: String
    public var promptTokens: Int
    public var completionTokens: Int
    public var seconds: Double
    /// The ENGINE's accounting for this request (prefill/decode split and
    /// prefix-cache hit), when it supplied one.  `seconds` above is this
    /// layer's wall clock and includes tokenisation and templating; when
    /// the two disagree, the engine's is the one that answers "how fast is
    /// the model" and this layer's is the one that answers "how fast is the
    /// product".  Report which one a number came from.
    public var engine: RequestStats?
    /// The max_tokens this turn actually ran with.  Equal to the caller's
    /// unless ContextBudget clamped it to what the engine's max length
    /// left after the prompt -- and that clamped value, not the caller's,
    /// is the one a `stop_reason: "max_tokens"` has to be judged against.
    public var maxTokens: Int
}

/// A turn after `LocalSession.prepare` and before `LocalSession.run`: the
/// prompt as ids, the config the engine will get, the media, and the
/// decoder state for the reply.  Opaque outside the module on purpose --
/// the two halves are only ever meant to be called in that order.
public struct PreparedTurn {
    let ids: [Int]
    let config: SamplingConfig
    let embeds: ImageEmbeddings?
    let decoder: IncrementalDecoder
    let splitter: StreamSplitter
    let opensThink: Bool
    let started: Date
    /// Anthropic cache_control breakpoints resolved to prompt-token offsets.
    let cachePoints: [PromptCachePoint]
    /// Prompt tokens exactly as the engine will see them (media woven in).
    public var promptTokens: Int { ids.count }
    /// The max_tokens the engine will run with (see `LocalReply.maxTokens`).
    public var maxTokens: Int { config.maxTokens }
    /// One line per adjustment the caller was not asked about, for the
    /// server to put in the reply's `warnings` and the log.
    public var warnings: [String]
}

public enum LocalSessionError: LocalizedError {
    /// The engine abandoned a GPU step (a Metal command buffer failed --
    /// memory pressure, typically) and interrupted this request. Whatever
    /// was generated is not trustworthy; the caller gets an error, not a
    /// truncated answer.
    case interrupted(String)
    case notLoaded
    case tokenizerMissing(String)
    case towerMismatch(towerHidden: Int, llmHidden: Int)
    /// The prompt alone fills (or overfills) the engine's max length, so
    /// there is no room for a single new token.  Carries the numbers the
    /// engine's log line had and the client never saw.
    case promptTooLong(promptTokens: Int, maxTokens: Int, maxLength: Int)

    public var errorDescription: String? {
        switch self {
        case let .promptTooLong(promptTokens, maxTokens, maxLength):
            return ContextBudget.refusalMessage(
                promptTokens: promptTokens, maxTokens: maxTokens,
                maxLength: maxLength)
        case .interrupted(let why): return "generation interrupted by the engine: \(why)"
        case .notLoaded:
            return "local engine is not loaded yet"
        case .tokenizerMissing(let detail):
            return "the .gguf has no usable tokenizer (\(detail))"
        case .towerMismatch(let towerHidden, let llmHidden):
            return "vision tower emits \(towerHidden)-wide embeddings but "
                + "the LLM expects \(llmHidden) — wrong tower for this model"
        }
    }
}

/// Loads once, serves many requests. Loading a 35B model takes minutes and
/// keeps ~6 GB of pages resident, so this is a long-lived object, never a
/// per-request one.
public final class LocalSession {
    private let engine: Engine
    private let tokenizer: any TokenizingVocabulary
    private let template: ChatTemplate
    private let eosTokenID: Int64?
    /// Every token that ends generation — see BPETokenizer.stopTokenIDs.
    private let stopTokenIDs: [Int64]
    /// Last rendered prompt and its token ids, for suffix-incremental
    /// tokenization.  Conversations are append-only: each turn's prompt is
    /// the previous prompt plus a new "<|im_start|>..." block, and special
    /// markers are hard tokenizer boundaries (they map to single special
    /// ids, so nothing merges across them).  Tokenizing only the suffix
    /// turns a measured 60 ms of BPE on an 8K-token history into ~1 ms —
    /// the growing term of warm-cache TTFT (18 us/token) was exactly this.
    private var tokCache: (prompt: String, ids: [Int])?
    /// How this model splices a picture into a prompt: which token the
    /// template emits, whether it is wrapped, and whether the image run
    /// carries an M-RoPE table. See MediaLayout — the four answers differ
    /// between Qwen3.5 and Gemma 4 and none of them fails loudly.
    public var mediaLayout: MediaLayout?
    /// The placeholder the chat template emits for an image. Kept as a
    /// separate settable for callers that only know the id (the warm-up
    /// path builds a synthetic embedding and needs nothing else); setting
    /// it alone selects the Qwen convention, which is what those callers
    /// were already getting.
    public var imageTokenID: Int? {
        get { mediaLayout?.imageTokenID }
        set {
            guard let newValue else { mediaLayout = nil; return }
            if mediaLayout?.imageTokenID == newValue { return }
            mediaLayout = MediaLayout(imageTokenID: newValue,
                                      openTokenID: nil, closeTokenID: nil,
                                      positions: .mropeImageGrid,
                                      name: "qwen3.5")
        }
    }
    /// LLM hidden size, from the .gguf. Only the image warm-up needs it: to
    /// build a synthetic embedding of the right shape without a tower.
    private let hidden: Int?
    /// The engine's total budget per request, prompt included -- what
    /// `--max-length` set.  Kept here because the engine only reports a
    /// breach as a status code after the fact; the session knows the
    /// prompt's exact token count before the engine is touched.
    public let maxLength: Int

    /// `graphPath` is a prebuilt .asgraph; building one still needs Python,
    /// so the app ships it beside the weights rather than generating it.
    /// The Metal GEMM backend is selected by an ENVIRONMENT VARIABLE, which
    /// is the whole reason this exists.
    ///
    /// ServerLauncher set AS_GEMM_BACKEND=metal on the Python subprocess it
    /// spawned. In-process there is no subprocess and nothing set it, so the
    /// engine silently fell back to CPU GEMM: TTFT 30 s and 13.2 tok/s
    /// against ~60 over HTTP, with no error and no log line saying so. A
    /// backend chosen by env var is invisible when it is wrong — the only
    /// symptom is that everything is slow.
    ///
    /// overwrite=0, so an explicitly set value still wins and the CPU path
    /// stays reachable for A/B without editing code.
    /// Set the engine's environment before ANYTHING touches the engine.
    ///
    /// Public and callable from app startup, because "before the first
    /// session is created" turned out not to be early enough: a status
    /// badge asking which GEMM backend was in use initialised the Metal
    /// context — singleton, probes and compiles on first call — before the
    /// kernel directory had been set. The indicator built to reveal a CPU
    /// fallback caused one.
    public static func prepareEnvironment() { selectBackendOnce() }

    private static func selectBackendOnce() {
        setenv("AS_GEMM_BACKEND", "metal", 0)
        // Prefix-cache economics for API/agent workloads (measured with
        // codex against the 35B): a GDN snapshot is ~25 MB, the default
        // 1024-token interval turns one 15k-token agent turn into a
        // ~400-500 MB cache entry, and the default 2 GB pot then holds
        // four turns — everything evicts before it can be reused
        // (hit_requests=0/20, cache pinned at 2043/2048 MB).  Sparser
        // snapshots and a bigger pot: a 4096 interval still snapshots
        // every span edge that matters for resume, at a quarter of the
        // memory; overridable per run, as always (overwrite=0).
        setenv("AS_CPU_GDN_SNAPSHOT_INTERVAL", "4096", 0)
        setenv("AS_CPU_PREFIX_CACHE_MB", "4096", 0)
        // Where the Metal kernels are.
        //
        // The engine looks for a compiled metallib, then for kernel source
        // in AS_METAL_KERNEL_DIR, and otherwise runs every weight GEMM on
        // the CPU — announcing it in one log line and then behaving like a
        // correct application that is four times slower.
        //
        // Two places, because there are two kinds of process. A shipped app
        // carries the sources in its bundle. A command-line tool built from
        // this package has no bundle at all, which is how localctl spent an
        // evening quietly measuring the CPU path while the app next to it
        // ran on Metal — and made me wrong twice, about context ceilings and
        // about MTP, from numbers that were real and off the wrong path.
        //
        // overwrite=0 throughout, so a developer pointing somewhere else
        // still wins.
        // DEV CANDIDATES ARE DEBUG-ONLY.
        //
        // They used to be compiled unconditionally, which put two absolute
        // paths from the BUILD machine into every release binary: the
        // #filePath-derived staged-engine directory, and a hardcoded
        // checkout path under $HOME.  On any machine where
        // either happens to exist, a shipped binary compiles kernels from a
        // FOREIGN tree -- verified: the release build answered "Name the
        // capital of Japan" with 'orda最值得kaarkaar util...' while
        // reporting only "gdn row kernel -> UNAVAILABLE, using inline-gate
        // row".  Garbage out, no error.
        //
        // AS_SHIP keeps the equivalent string out of the C++ side; it has no
        // effect here, so the release binary carried what the comment below
        // says it must not.  #if DEBUG is what actually enforces it.
        var candidates: [String?] = [
            // A shipped app carries the kernels in its bundle.
            Bundle.main.resourceURL?.appendingPathComponent("metal").path,
        ]
#if DEBUG
        // Development only.  A command-line tool built from this package has
        // no bundle, which is how localctl once spent an evening quietly
        // measuring the CPU path while the app beside it ran on Metal.
        candidates += [
            // The kernels staged NEXT TO the archives this binary was linked
            // against (the staging script copies them).  This outranks any
            // checkout path on purpose: the engine and its kernels must come
            // from the same tree, and the fallback below once had a whole
            // night of kernel A/Bs silently compiling a different checkout's
            // unchanged file.
            URL(fileURLWithPath: #filePath)   // .../Sources/Tempo9/LocalSession.swift
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("staged-engine/metal").path,
            // The engine checkout, for tools. There used to be a third
            // candidate here spelling out one developer's directory layout;
            // it matched on exactly one machine, and when nothing matches
            // this loop falls through SILENTLY -- kernels from the wrong
            // tree produce garbage output rather than an error, which is
            // what the comment above is about. An env var that is unset is
            // at least unset everywhere.
            ProcessInfo.processInfo.environment["TEMPO9_ENGINE_REPO"]
                .map { $0 + "/csrc/device/metal" },
        ]
#endif
        for path in candidates.compactMap({ $0 })
        where FileManager.default.fileExists(
            atPath: path + "/metal_gemm_kernels.metal") {
            setenv("AS_METAL_KERNEL_DIR", path, 0)
            break
        }
    }

    public init(modelName: String, graphPath: String, ggufPath: String,
         maxLength: Int64, maxBatch: Int32 = 1,
         enablePrefixCache: Bool = true) throws {
        LocalSession.selectBackendOnce()
        let gguf = try GGUFFile(path: ggufPath)
        self.hidden = gguf.arch("embedding_length")?.intValue
        // Two tokenizers, chosen by what the file says it is. SentencePiece
        // runs inside the engine; byte-level BPE stays in Swift. Neither is a
        // fallback for the other -- a vocabulary read with the wrong
        // algorithm still yields ids, and those ids decode to fluent text
        // from the wrong distribution, which is why the dispatch is on the
        // declared model and there is no default arm.
        let tokModel = gguf.kv["tokenizer.ggml.model"]?.stringValue
        if engineOwnsTokenizer(model: tokModel) {
            let tokens = gguf.kv["tokenizer.ggml.tokens"]?.stringsValue ?? []
            let types = gguf.kv["tokenizer.ggml.token_type"]?.numbersValue ?? []
            var specials = Set<Int>()
            for (i, t) in types.enumerated() where t == 3 || t == 4 {
                specials.insert(i)
            }
            self.tokenizer = try EngineTokenizer(ggufPath: ggufPath,
                                                 vocabulary: tokens,
                                                 specialIDs: specials)
        } else {
            self.tokenizer = try BPETokenizer(gguf: gguf)
        }
        self.template = try ChatTemplate(gguf: gguf)
        self.eosTokenID = tokenizer.eosTokenID.map(Int64.init)
        self.stopTokenIDs = tokenizer.stopTokenIDs.map(Int64.init)
        self.maxLength = Int(maxLength)
        self.engine = try Engine(modelName: modelName,
                                 graphPath: graphPath,
                                 weightsPath: ggufPath,
                                 maxLength: maxLength,
                                 maxBatch: maxBatch,
                                 enablePrefixCache: enablePrefixCache)
    }

    /// Turn a vision-tower encoding into what the engine needs.
    ///
    /// Three steps that all have to agree, and none of which the engine does
    /// for you: the template emits ONE placeholder token, it is expanded to
    /// one per merged patch, and the M-RoPE table is built over the expanded
    /// ids. Get the count wrong and there is no error — the placeholder run
    /// and the embedding run just disagree, and the model describes an image
    /// nobody encoded.
    private func weave(ids: [Int], encoding: ImagePlacement?,
                       audio: AudioPlacement? = nil) throws
        -> (ids: [Int], embeds: ImageEmbeddings) {
        guard let layout = mediaLayout else {
            throw LocalSessionError.tokenizerMissing("no image_token_id set")
        }
        let imageToken = layout.imageTokenID
        // A tower whose width does not match the LLM is not a degraded
        // pairing, it is a broken one: the C layer only checks that
        // byte_count is self-consistent, and the engine writes the rows
        // into hidden-sized slots regardless, so a 2048-wide embedding in
        // a 4096-wide model decodes as pure noise ("!!!!" captions, hours
        // lost).  Refuse loudly instead.
        for width in [encoding?.hidden, audio?.hidden].compactMap({ $0 }) {
            if let hidden, hidden > 0, width != hidden {
                throw LocalSessionError.towerMismatch(
                    towerHidden: width, llmHidden: hidden)
            }
        }
        if audio != nil && !layout.hearsAudio {
            throw LocalSessionError.tokenizerMissing(
                "\(layout.name) has no audio token — this model cannot be "
                + "handed sound")
        }
        let expanded = try TowerAux.expandMediaPlaceholders(
            inputIDs: ids, layout: layout,
            imageTokens: encoding.map { [$0.tokens] } ?? [],
            audioTokens: audio.map { [$0.tokens] } ?? [])
        // Three schemes, and the difference is what goes IN the table, not
        // whether there is one:
        //
        //   sequential      no table. Gemma 4's media block gets its
        //                   structure from intra-block bidirectional
        //                   attention, which the engine turns on by finding
        //                   the run of layout.imageTokenID in the ids.
        //   mropeImageGrid  t/h/w from the patch grid (Qwen3.5).
        //   mropeTimeline   all three axes are the sequence index. Audio is
        //                   one-dimensional in time, so a grid would be
        //                   inventing structure that is not there.
        let positions: [Int32]
        switch layout.positions {
        case .sequential:
            positions = []
        case .mropeImageGrid:
            positions = encoding == nil ? [] : try TowerAux.llmMRoPEPositions(
                inputIDs: expanded, imageTokenID: imageToken,
                grids: [(t: 1, h: encoding!.gridH, w: encoding!.gridW)])
        case .mropeTimeline:
            var table = [Int32](repeating: 0, count: 3 * expanded.count)
            for i in 0..<expanded.count {
                let p = Int32(i)
                table[i] = p
                table[expanded.count + i] = p
                table[2 * expanded.count + i] = p
            }
            positions = table
        }
        var blocks: [MediaBlock] = []
        if let encoding {
            blocks.append(MediaBlock(
                data: encoding.embedding.withUnsafeBufferPointer {
                    Data(buffer: $0)
                },
                tokenCount: encoding.tokens, hidden: encoding.hidden,
                mediaTokenID: Int64(imageToken),
                isFloat16: false,   // Encoding.embedding is fp32
                deepstack: encoding.deepstack.withUnsafeBufferPointer {
                    Data(buffer: $0)
                },
                numDeepstackLayers: encoding.numDeepstackLayers))
        }
        if let audio, let audioToken = layout.audioTokenID {
            blocks.append(MediaBlock(
                data: audio.embedding.withUnsafeBufferPointer {
                    Data(buffer: $0)
                },
                tokenCount: audio.tokens, hidden: audio.hidden,
                mediaTokenID: Int64(audioToken),
                isFloat16: false))
        }
        // Both keys, mixed: the prefix cache must miss when EITHER the frame
        // or the utterance changed, and keying on the frame alone would
        // serve one question's prefill to the next question about the same
        // view.
        var key = UInt64(bitPattern: encoding?.contentKey ?? 0)
        if let audio {
            key = key &* 1099511628211
                ^ UInt64(bitPattern: audio.contentKey)
        }
        let embeds = ImageEmbeddings(blocks: blocks, mropePositions: positions,
                                     usesMRoPE: layout.usesMRoPE,
                                     contentHash: key)
        return (expanded, embeds)
    }

#if DEV_BUILD
    private static func dumpRequest(prompt: String, ids: [Int],
                                    embeds: ImageEmbeddings,
                                    cfg: SamplingConfig,
                                    enableThinking: Bool) {
        let fm = FileManager.default
        guard let base = fm.urls(for: .applicationSupportDirectory,
                                 in: .userDomainMask).first else { return }
        // Named after this SDK, not after an app that embeds it. These
        // directories are created inside the HOST application's container,
        // so a third-party app linking Tempo9 would otherwise find a folder
        // named after someone else's product on its users' disks.
        let dir = base.appendingPathComponent(
            "Tempo9/watchpoint/debug_requests")
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let stamp = Int(Date().timeIntervalSince1970)
        let meta: [String: Any] = [
            "ids_count": ids.count,
            // Per block, because a request can carry more than one and
            // "which of them was wrong" is the question this dump exists to
            // answer.
            "blocks": embeds.blocks.map { b -> [String: Any] in
                ["bytes": b.data.count, "tokens": b.tokenCount,
                 "hidden": b.hidden, "f16": b.isFloat16,
                 "media_token_id": b.mediaTokenID]
            },
            "content_hash": String(embeds.contentHash),
            "mrope_count": embeds.mropePositions.count,
            "temperature": cfg.temperature,
            "top_p": cfg.topP,
            "top_k": cfg.topK,
            "repetition_penalty": cfg.repetitionPenalty,
            "speculation_k": cfg.speculationK,
            "max_tokens": cfg.maxTokens,
            "do_sample": cfg.doSample,
            "enable_thinking": enableThinking,
            "prompt": prompt,
        ]
        if let j = try? JSONSerialization.data(
            withJSONObject: meta, options: [.prettyPrinted]) {
            try? j.write(to: dir.appendingPathComponent("req-\(stamp).json"))
        }
        var idbuf = Data()
        for i in ids { withUnsafeBytes(of: Int64(i).littleEndian) {
            idbuf.append(contentsOf: $0) } }
        try? idbuf.write(to: dir.appendingPathComponent("req-\(stamp).ids.i64"))
        for (i, b) in embeds.blocks.enumerated() {
            try? b.data.write(to: dir.appendingPathComponent(
                "req-\(stamp).emb\(i)-tok\(b.mediaTokenID).bin"))
        }
        var mbuf = Data()
        for v in embeds.mropePositions {
            withUnsafeBytes(of: v.littleEndian) { mbuf.append(contentsOf: $0) }
        }
        try? mbuf.write(to: dir.appendingPathComponent("req-\(stamp).pos.i32"))
    }
#endif

    /// The engine's counters, straight through.
    public func stats() -> EngineStats? { try? engine.stats() }

    /// Run a throwaway generation so the first REAL request is not the cold
    /// one.
    ///
    /// Loading the model is not the end of the setup cost. Measured on the
    /// 35B-A3B q3km, first request vs the ones after it, in one process:
    ///
    ///     run 1: TTFT 4095 ms, decode 16.1 tok/s
    ///     run 2: TTFT  116 ms, decode 87.6 tok/s
    ///     run 3: TTFT   92 ms, decode 83.3 tok/s
    ///
    /// Same engine, same 17-token prompt. The first pass compiles Metal
    /// pipelines and builds what the graph needs lazily, and it charges all of
    /// that to whoever asks first. Because the load was lazy too, that was
    /// always the user's opening question — which is why the in-process path
    /// looked five times slower than the server while actually being faster.
    ///
    /// Doing it here means the cost lands inside the loading indicator, where
    /// the user is already waiting, instead of on the first answer.
    ///
    /// Text-only: this warms the shared decode path, not the image prefill,
    /// which goes through RichEmbedding and pays its own first-call cost.
    /// - Parameter onPhase: reports which half is running. Both take real
    ///   time, and a single "warming up" label makes the longer one look
    ///   like a stall.
    public func warmUp(onPhase: (String) -> Void = { _ in }) async {
        onPhase("warming up (text)")
        var cfg = SamplingConfig()
        // Enough tokens to get through the ramp -- one token would leave the
        // decode kernels cold, and the ramp is half of what this is for.
        cfg.maxTokens = 8
        cfg.doSample = false
        cfg.topK = 1
        cfg.temperature = 0
        _ = try? await stream(messages: [["role": "user", "content": "hi"]],
                              config: cfg, enableThinking: false,
                              onDelta: { _ in })

        // The image path on top, because text does not cover it: an image
        // prefill goes through RichEmbedding and pays its own first call.
        // Measured after a text-only warm-up, first image request vs steady
        // state: TTFT 445 ms vs 108 ms. Small next to the 7.7 s it used to
        // be, and still the slowest thing the user meets.
        //
        // Zeros of the right SHAPE, not a real picture. What is being
        // compiled depends on dimensions, not on content, and generating a
        // real embedding here would mean loading the vision tower into a
        // target that deliberately does not depend on it.
        guard imageTokenID != nil, let hidden, hidden > 0 else { return }
        onPhase("warming up (image)")
        // 32x32 patches merged to 16x16 = 256 tokens: the demo's usual tier.
        // A different tier warms a different shape, which is why the first
        // image can still be slightly slow after this.
        let grid = 32, tokens = 256
        let dummy = ImagePlacement(
            embedding: [Float](repeating: 0, count: tokens * hidden),
            tokens: tokens, hidden: hidden, gridH: grid, gridW: grid,
            // Its own key so it cannot collide with a real frame's cache
            // entry and answer a later question from a blank image.
            contentKey: Int64(bitPattern: 0x5741_524D_5F49_4D47))
        _ = try? await stream(
            messages: [["role": "user",
                        "content": [["type": "image"],
                                    ["type": "text", "text": "hi"]] as [Any]]],
            config: cfg, enableThinking: false, image: dummy,
            onDelta: { _ in })
    }

    /// Same contract as an HTTP client's stream(): deltas as they arrive, a
    /// reply at the end.
    /// - Parameter tools: JSON-schema tool definitions, handed to the chat
    ///   template rather than pasted into a message. The template renders
    ///   them into its own `<tools>` block and states the reply format it
    ///   was trained on -- which is not JSON. Passing schemas as prose in a
    ///   user message asks the model to follow an ad-hoc convention instead
    ///   of the one it knows.
    /// One turn at a time, across EVERY caller.  The engine runs
    /// max_batch=1, but the crash lived upstream of it: template render and
    /// the suffix-tokenize cache are shared mutable state, and the app's
    /// keep-alive ping racing an API request corrupted a render mid-flight
    /// ("System message must be at the beginning" from a well-formed
    /// request).  The API server's own gate only serialized API callers;
    /// this serializes all of them.
    private let turnGate = TurnGate()

    /// How many prompt tokens `stream` would prefill for these messages:
    /// the same template rendering, the same tokenizer, the same count
    /// `LocalReply.promptTokens` reports afterwards.
    ///
    /// Under the turn gate for the same reason `stream` is -- the template
    /// render is shared mutable state -- but it leaves `tokCache` alone: a
    /// count is not a turn, and priming the cache with a prompt that is
    /// never sent would evict the one the next real turn extends.
    public func countTokens(messages: [[String: Any]],
                            tools: [[String: Any]]? = nil,
                            enableThinking: Bool = false) async throws -> Int {
        await turnGate.acquire()
        defer { turnGate.releaseFromSync() }
        let prompt = try template.render(
            messages: messages,
            addGenerationPrompt: true,
            extra: {
                var e: [String: Any] = ["enable_thinking": enableThinking]
                if let tools, !tools.isEmpty { e["tools"] = tools }
                return e
            }())
        return tokenizer.encode(prompt, addSpecialTokens: false).count
    }

    public func stream(messages: [[String: Any]],
                config: SamplingConfig,
                enableThinking: Bool,
                images: ImageEmbeddings? = nil,
                image: ImagePlacement? = nil,
                audio: AudioPlacement? = nil,
                tools: [[String: Any]]? = nil,
                onDelta: @escaping (LocalDelta) -> Void) async throws
        -> LocalReply {
        let turn = try await prepare(messages: messages, config: config,
                                     enableThinking: enableThinking,
                                     images: images, image: image,
                                     audio: audio, tools: tools)
        return try await run(turn, onDelta: onDelta)
    }

    /// The host-side half of a turn: render, tokenize, weave media, pick
    /// the decoder -- everything that can FAIL before the engine is
    /// touched.  Split from `run` so a server can do this before it
    /// commits to a response: a template that throws on the third
    /// message used to throw after `HTTP/1.1 200` and `message_start`
    /// were already on the wire, and the client saw a stream that simply
    /// stopped.
    public func prepare(messages: [[String: Any]],
                        config: SamplingConfig,
                        enableThinking: Bool,
                        images: ImageEmbeddings? = nil,
                        image: ImagePlacement? = nil,
                        audio: AudioPlacement? = nil,
                        tools: [[String: Any]]? = nil,
                        cacheBreakpoints: [CacheBreakpoint] = []) async throws
        -> PreparedTurn {
        // The gate protects the HOST-side prep — template render, the
        // incremental-tokenization cache (tokCache is shared mutable
        // state), media weaving — and is released BEFORE the engine loop:
        // the engine schedules concurrent requests itself (continuous
        // batching), and holding the gate across decode serialized every
        // HTTP request end to end (serveheadless N=4 showed Running:1 and
        // staircased TTFTs of 0.9/2.7/4.6/6.3 s while the engine was idle
        // between turns).  The app is unaffected: it awaits each reply
        // before sending the next turn, so its turns never overlap anyway.
        await turnGate.acquire()
        defer { turnGate.releaseFromSync() }
        let started = Date()

        // enable_thinking is what the Qwen3.5 template keys its <think>
        // preamble off; passing it through `extra` is how the server does it
        // too, so both paths render the same prompt.
        // AS_NO_CHAT_TEMPLATE=1: feed the user text to the tokenizer
        // verbatim. Only for tensor-level comparison against another
        // implementation -- the two must see the SAME token ids, and a chat
        // template is the one thing the other implementation will not
        // reproduce. Never a serving path.
        let prompt: String
        if ProcessInfo.processInfo.environment["AS_NO_CHAT_TEMPLATE"] == "1" {
            prompt = messages.last.flatMap { $0["content"] as? String } ?? ""
        } else {
            prompt = try template.render(
                messages: messages,
                addGenerationPrompt: true,
                extra: {
                    var e: [String: Any] = ["enable_thinking": enableThinking]
                    if let tools, !tools.isEmpty { e["tools"] = tools }
                    return e
                }())
        }
        let tRendered = Date()
        var ids: [Int]
        if let c = tokCache, prompt == c.prompt {
            ids = c.ids
        } else if let c = tokCache, prompt.hasPrefix(c.prompt),
                  prompt[prompt.index(prompt.startIndex,
                                      offsetBy: c.prompt.count)...]
                      .hasPrefix("<|") {
            // Append-only turn: the delta starts at a special marker, so
            // its tokenization cannot merge into the cached prefix.
            let delta = String(prompt[prompt.index(prompt.startIndex,
                                                   offsetBy: c.prompt.count)...])
            ids = c.ids + tokenizer.encode(delta, addSpecialTokens: false)
        } else {
            ids = tokenizer.encode(prompt, addSpecialTokens: false)
        }
        tokCache = (prompt, ids)
        // TEMPO9_TIME_STAGES=1: where a warm request's TTFT goes.  The
        // engine's prefix cache makes the GPU part ~flat, so past a few K
        // of history the host-side render+tokenize becomes the growing
        // term — measured 18us/token before this was instrumented.
        if ProcessInfo.processInfo.environment["TEMPO9_TIME_STAGES"] != nil {
            let tTok = Date()
            FileHandle.standardError.write(Data(String(
                format: "stage: render %.1f ms, tokenize %.1f ms (%d ids)\n",
                tRendered.timeIntervalSince(started) * 1e3,
                tTok.timeIntervalSince(tRendered) * 1e3,
                ids.count).utf8))
        }
#if DEV_BUILD
        // apiDebugRaw: record every request's rendered prompt + token ids
        // for offline replay.  The prefix-cache zero-hit hunt needs to diff
        // the ids two consecutive turns ACTUALLY used — wire-level captures
        // already proved the JSON identical; the suspect is tokenization.
        if UserDefaults.standard.bool(forKey: "apiDebugRaw") {
            let dir = FileManager.default.urls(
                for: .libraryDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Logs/Tempo9/reqdump",
                                        isDirectory: true)
            try? FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            let stamp = String(format: "%.3f", Date().timeIntervalSince1970)
            try? prompt.write(to: dir.appendingPathComponent("p-\(stamp).txt"),
                              atomically: true, encoding: .utf8)
            var buf = Data()
            for i in ids { withUnsafeBytes(of: Int64(i).littleEndian) {
                buf.append(contentsOf: $0) } }
            try? buf.write(to: dir.appendingPathComponent("i-\(stamp).i64"))
        }
#endif
        // Before weaving: media expands placeholder ids, which would move
        // every offset after it (and the Anthropic path refuses media).
        let cachePoints = (image == nil && audio == nil && !cacheBreakpoints.isEmpty)
            ? resolveCachePoints(cacheBreakpoints, messages: messages,
                                 tools: tools, enableThinking: enableThinking,
                                 prompt: prompt, ids: ids)
            : []
        var embeds = images
        if image != nil || audio != nil {
            let woven = try weave(ids: ids, encoding: image, audio: audio)
            ids = woven.ids
            embeds = woven.embeds
        }

        var cfg = config
        // The engine's max length is prompt + max_tokens.  Decide here,
        // with the exact prompt count (media woven in), what the engine
        // would otherwise decide after the fact with a status code: refuse
        // only when the prompt alone overflows, clamp max_tokens otherwise
        // and say so.  See ContextBudget.
        var warnings: [String] = []
        switch ContextBudget.apply(promptTokens: ids.count,
                                   maxTokens: cfg.maxTokens,
                                   maxLength: maxLength) {
        case .fits:
            break
        case let .clamped(maxTokens, warning):
            cfg.maxTokens = maxTokens
            warnings.append(warning)
            FileHandle.standardError.write(
                Data("[tempo9] warning: \(warning)\n".utf8))
        case let .refused(promptTokens, maxTokens, maxLength):
            throw LocalSessionError.promptTooLong(
                promptTokens: promptTokens, maxTokens: maxTokens,
                maxLength: maxLength)
        }
        // Both: eos_token_id is what the engine checks to stop, and the
        // stop-word list catches the chat template's own end marker when the
        // .gguf declares a different one.
        if cfg.eosTokenID == nil { cfg.eosTokenID = eosTokenID }
        if cfg.stopTokenIDs.isEmpty {
            // ALL of them, not just eos. A Gemma turn ends on <turn|>, which
            // is not the .gguf's declared eos, and stopping only on eos means
            // the model answers, opens a fresh turn and answers again until
            // it hits max_tokens.
            cfg.stopTokenIDs = stopTokenIDs.isEmpty
                ? (eosTokenID.map { [$0] } ?? []) : stopTokenIDs
        }

        // Keep the tags that CARRY MEANING. They are special tokens, so the
        // decoder strips them by default, and each one that goes missing
        // breaks something downstream in a way that looks like a model
        // failure:
        //
        //   <think>/</think>  — the splitter waits forever for a close tag
        //     that was removed before it saw it, and the whole answer stays
        //     filed as reasoning.
        //   <tool_call>/</tool_call> — the model emits a perfectly correct
        //     call and the parser sees only a bare <function=...> with no
        //     wrapper, so it reports NO tool call at all. Measured: five
        //     tool questions, five correct calls, zero detected.
        //
        // Everything else special still goes.
        let keepTags = Set(tokenizer.preservedTagTexts
            .compactMap { tokenizer.id(of: $0) })
        let decoder = IncrementalDecoder(tokenizer: tokenizer,
                                         preserving: keepTags)
        // Does the rendered prompt leave <think> OPEN?
        //
        // Qwen3.5's template opens the block in the generation prompt when
        // thinking is on, so the model's output begins INSIDE it with no
        // opening tag to find; when thinking is off the same template appends
        // a pre-closed pair and nothing in the output is reasoning.
        //
        // Defaulting to false here reproduced the server's bug (#800) exactly:
        // the splitter waited for a <think> that had already been consumed by
        // the prompt, so the reasoning, a literal "</think>" and the answer
        // all arrived as one blob of content. Observed rather than assumed —
        // the template is the only thing that knows what it did.
        //
        // Gemma 4 answers "no" to this in BOTH states, and correctly: with
        // thinking off its template appends a pre-CLOSED empty block, and
        // with thinking on it appends nothing and the model emits its own
        // opener. Either way the output does not begin inside a block. The
        // test below reaches that answer on its own -- an open marker that
        // is always followed by its close is never "still open".
        let tags = tokenizer.dialect.reasoningTags
        // Byte search, NOT `range(of:options:.backwards)`.
        //
        // Foundation's backwards search recurses, and every request is served
        // on a Swift cooperative-pool thread, whose stack is a fraction of the
        // main thread's. With a small prompt that is fine; with an agent
        // framework's prompt -- 34 KB of system text and 34 tool schemas --
        // it walks off the stack guard page and the process takes SIGBUS
        // ("Could not determine thread index for stack guard region").
        //
        // It crashed on the SECOND turn of an OpenClaw session and never on
        // curl, which is why this layer looked healthy: the first request is
        // small enough to survive. Reproduced from the captured body 100% of
        // the time.
        let opensThink = Self.lastByteIndex(of: tags.open, in: prompt).map { open in
            Self.lastByteIndex(of: tags.close, in: prompt).map { open > $0 } ?? true
        } ?? false
        // Harmony is not an open/close pair; it gets its own splitter, and
        // the tag-open probe above is meaningless for it (its template ends
        // with <|start|>assistant, which is a header, not an open block).
        let splitter: StreamSplitter = tokenizer.isHarmony
            ? HarmonySplitter()
            : ThinkSplitter(openTag: tags.open,
                            closeTag: tags.close,
                            forceReasoning: opensThink)
#if DEV_BUILD
        // Dev-only request dump (same switch as the frame dump): the exact
        // ids + embedding bytes + config this request hands the engine, so
        // a garbled in-app answer can be replayed OUTSIDE the app byte for
        // byte.  Everything upstream (tower, template, TowerAux, C API) has
        // been exonerated one piece at a time; this pins whatever is left.
        if UserDefaults.standard.bool(forKey: "watchpointDumpFrames"),
           let e = embeds {
            Self.dumpRequest(prompt: prompt, ids: ids, embeds: e, cfg: cfg,
                             enableThinking: enableThinking)
        }
#endif

        // Prep done -- the gate is released by the defer above, so other
        // requests can prep and enter the engine while this one decodes.
        return PreparedTurn(ids: ids, config: cfg, embeds: embeds,
                            decoder: decoder, splitter: splitter,
                            opensThink: opensThink, started: started,
                            cachePoints: cachePoints,
                            warnings: warnings)
    }

    /// The engine half of a turn.  Reads only what `prepare` handed over;
    /// no shared state, no gate.
    public func run(_ turn: PreparedTurn,
                    onDelta: @escaping (LocalDelta) -> Void) async throws
        -> LocalReply {
        let ids = turn.ids
        let cfg = turn.config
        let embeds = turn.embeds
        let decoder = turn.decoder
        let splitter = turn.splitter
        let opensThink = turn.opensThink
        let started = turn.started
        var content = "", reasoning = ""
        var rawPieces: [String] = []
        var generated = 0
        // The engine's own per-request accounting, delivered on the final
        // chunk.  Preferred over anything this layer could time itself:
        // a wall clock here also measures tokenisation, templating and
        // scheduling, which is exactly the asymmetry that made our
        // published prefill number worse than the forward it reports.
        var engineStats: RequestStats?

        var finalReason: FinishReason = .none
        for try await chunk in engine.generate(
            inputIDs: ids.map(Int64.init), config: cfg, images: embeds,
            cachePoints: turn.cachePoints) {
            generated += chunk.tokenIDs.count
            let text = decoder.feed(chunk.tokenIDs.map(Int.init))
            if !text.isEmpty {
                rawPieces.append(text)
                let split = splitter.feed(text)
                content += split.content
                reasoning += split.reasoning
                if !split.isEmpty {
                    // The UI is main-actor state; hop before touching it.
                    // Doing this at the source means no caller has to
                    // remember, which is the same rule the HTTP path follows.
                    await MainActor.run {
                        onDelta(LocalDelta(content: split.content,
                                           reasoning: split.reasoning,
                                           tokens: generated))
                    }
                }
            }
            if let st = chunk.stats { engineStats = st }
            if chunk.isFinal { finalReason = chunk.finishReason; break }
        }

        // Anything the decoder or splitter was holding back. A partial UTF-8
        // sequence or a trailing "<thi" is real output; dropping it would
        // silently truncate the answer.
        let tailText = decoder.flush()
        if !tailText.isEmpty {
            let split = splitter.feed(tailText)
            content += split.content
            reasoning += split.reasoning
        }
        let tail = splitter.flush()
        content += tail.content
        reasoning += tail.reasoning

        // Stray-think retro-split: with thinking OFF the model occasionally
        // hallucinates a bare `</think>` mid-answer ("answer</think>answer"
        // — field, 02:0x), and a force=false splitter passes it through.
        // The FINAL reply re-splits at the stray tag: everything before it
        // was the model's uninvited thought, everything after is the
        // answer.  History, TTS and API consumers all get clean text; only
        // the live stream briefly showed the duplicate.
        if !opensThink, let r = content.range(of: "</think>") {
            let before = String(content[..<r.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let after = String(content[r.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !after.isEmpty {
                reasoning += (reasoning.isEmpty ? "" : "\n") + before
                content = after
            } else {
                content = before  // tag at the very end: just drop it
            }
        }

        // apiDebugRaw: where did the tokens go?  35 generated tokens once
        // vanished between the engine and this return — this line is the
        // instrument that catches the next such disappearance in one run.
        if UserDefaults.standard.bool(forKey: "apiDebugRaw") {
            let raw = rawPieces.joined()
            FileHandle.standardError.write(Data(
                ("[raw] gen=\(generated) content=\(content.count)ch "
                 + "reasoning=\(reasoning.count)ch raw=\(raw.count)ch "
                 + "rawText=<<<\(raw.prefix(400))>>>\n").utf8))
        }
        if finalReason == .interrupted {
            throw LocalSessionError.interrupted("GPU step failed; the engine interrupted this request")
        }
        return LocalReply(text: content,
                        reasoning: reasoning,
                        promptTokens: ids.count,
                        completionTokens: generated,
                        seconds: Date().timeIntervalSince(started),
                        engine: engineStats,
                        maxTokens: cfg.maxTokens)
    }

    // MARK: - cache_control -> engine cache points

    /// Byte length of each token's decoded text, by id.  Vocabulary-level
    /// and immutable, so it only grows; guarded by the turn gate like
    /// tokCache.
    private var tokenByteLength: [Int: Int] = [:]

    /// Resolve Anthropic breakpoints to prompt-token offsets.
    ///
    /// Every point is found by RENDERING the prefix it closes and checking
    /// that the render is a byte-prefix of the real prompt -- the template
    /// is the only thing that knows where a message ends, and a breakpoint
    /// placed past a byte the next request may change would pin a node no
    /// one can ever hit.  A breakpoint that cannot be placed exactly is
    /// dropped, never guessed: these are hints, and a request must never
    /// fail or slow down over one.
    func resolveCachePoints(_ breakpoints: [CacheBreakpoint],
                            messages: [[String: Any]],
                            tools: [[String: Any]]?,
                            enableThinking: Bool,
                            prompt: String, ids: [Int]) -> [PromptCachePoint] {
        let t0 = Date()
        let promptBytes = Array(prompt.utf8)
        func render(_ msgs: [[String: Any]],
                    _ t: [[String: Any]]?) -> [UInt8]? {
            var e: [String: Any] = ["enable_thinking": enableThinking]
            if let t, !t.isEmpty { e["tools"] = t }
            return (try? template.render(messages: msgs,
                                         addGenerationPrompt: false,
                                         extra: e)).map { Array($0.utf8) }
        }
        var points: [PromptCachePoint] = []
        var notes: [String] = []
        for bp in breakpoints {
            switch bp {
            case let .tools(ttl):
                // Where does the tool list END?  Render the head once as is
                // and once with one more tool appended: the renders agree
                // exactly up to the slot a new tool would occupy.  That is
                // the boundary a client's tool list moves at -- Claude
                // Code's MCP connectors land there when they finish
                // connecting after the first request -- and it needs no
                // knowledge of how this template spells a tool block.
                // The head is the first message plus a probe user turn:
                // Qwen3.5's template refuses a conversation with no user
                // query, and the probe sits after the tools either way.
                let head = Array(messages.prefix(1)) + [Self.probeUser(1)]
                guard let tools, let last = tools.last, !messages.isEmpty,
                      let a = render(head, tools),
                      let b = render(head,
                                     tools + [Self.probeTool(like: last)])
                else { notes.append("tools: no render"); continue }
                let n = min(Self.commonPrefix(a, b),
                            Self.commonPrefix(a, promptBytes))
                guard n > 0 else { notes.append("tools: empty"); continue }
                let endTok = tokenOffset(atByte: n, in: ids)
                points.append(PromptCachePoint(tokenOffset: endTok,
                                               ttlSeconds: ttl))
                notes.append("tools@\(n)B")
                // The end pin covers tools APPENDED.  A tool that appears or
                // disappears inside the list moves the prompt earlier --
                // Claude Code carries a transient WaitForMcpServers in its
                // first request while connectors are still connecting, 24th
                // of 29 tools, ~2.2k tokens before the end, and only the
                // engine's 4096-token ladder caught it (1280 tokens short).
                // So also pin the tool boundaries ~1k/2k/4k/8k tokens before
                // the end: log-spaced by DISTANCE, not "the last N tools",
                // because where a transient tool sorts decides its index
                // and nothing decides its distance.  Each rung is placed by
                // estimate (JSON size) and located exactly by one render
                // with only the first k tools, which agrees with the prompt
                // up to where tool k begins.
                if Self.toolLadderEnabled, tools.count > 1, endTok > 0 {
                    let bytesPerToken = Double(n) / Double(endTok)
                    let sizes = tools.map {
                        Double(((try? JSONSerialization.data(
                            withJSONObject: $0))?.count ?? 0) + 1)
                    }
                    for k in Self.toolLadderRungs(
                        sizes: sizes, bytesPerToken: bytesPerToken,
                        distances: Self.toolLadderDistances) {
                        guard let r = render(head, Array(tools.prefix(k)))
                        else { continue }
                        let m = Self.commonPrefix(r, promptBytes)
                        guard m > 0, m < n else { continue }
                        points.append(PromptCachePoint(
                            tokenOffset: tokenOffset(atByte: m, in: ids),
                            ttlSeconds: ttl))
                        notes.append("tool\(k)@\(m)B")
                    }
                }
            case let .message(k, ttl):
                guard k >= 0, k < messages.count else {
                    notes.append("msg\(k): out of range"); continue
                }
                let prefix = Array(messages[0...k])
                let end: Int
                if let a = render(prefix, tools) {
                    // A template may render a message differently once it
                    // is no longer the last one (Qwen3's reasoning on past
                    // assistant turns); then the prefix is not the
                    // prompt's, and no offset is honest.
                    guard Self.commonPrefix(a, promptBytes) == a.count else {
                        notes.append("msg\(k): not a prefix"); continue
                    }
                    end = a.count
                } else if let a = render(prefix + [Self.probeUser(1)], tools),
                          let b = render(prefix + [Self.probeUser(2)], tools) {
                    // The prefix alone does not render (a system message
                    // with no user query).  Follow it with two different
                    // user turns: the renders agree up to the start of
                    // that turn's CONTENT, which is past the next turn's
                    // header but before anything the client wrote.
                    let n = Self.commonPrefix(a, b)
                    guard Self.commonPrefix(a, promptBytes) >= n else {
                        notes.append("msg\(k): not a prefix"); continue
                    }
                    end = n
                } else {
                    notes.append("msg\(k): no render"); continue
                }
                points.append(PromptCachePoint(
                    tokenOffset: tokenOffset(atByte: end, in: ids),
                    ttlSeconds: ttl))
                notes.append("msg\(k)@\(end)B")
            }
        }
        if ProcessInfo.processInfo.environment["TEMPO9_CACHE_POINT_DEBUG"] == "1" {
            var line = String(format: "[cache-points] %.1f ms, %d ids:",
                              Date().timeIntervalSince(t0) * 1e3, ids.count)
            for (note, p) in zip(notes.filter { $0.contains("@") }, points) {
                let lo = max(0, p.tokenOffset - 3), hi = min(ids.count, p.tokenOffset + 3)
                let before = String(decoding: tokenizer.decodeBytes(
                    Array(ids[lo..<p.tokenOffset]), skipSpecialTokens: false,
                    preserving: []), as: UTF8.self)
                let after = String(decoding: tokenizer.decodeBytes(
                    Array(ids[p.tokenOffset..<hi]), skipSpecialTokens: false,
                    preserving: []), as: UTF8.self)
                line += " \(note)->tok \(p.tokenOffset) ttl \(p.ttlSeconds) "
                    + "[\(before.debugDescription)|\(after.debugDescription)]"
            }
            for n in notes where !n.contains("@") { line += " (" + n + ")" }
            FileHandle.standardError.write(Data((line + "\n").utf8))
        }
        return points
    }

    /// The number of leading tokens whose decoded bytes fit in `limit`
    /// bytes of the prompt.  `ids` IS the prompt's tokenization, so its
    /// decoded bytes are the prompt's bytes; a token straddling the limit
    /// is excluded, which keeps the point at or before the boundary.
    private func tokenOffset(atByte limit: Int, in ids: [Int]) -> Int {
        var bytes = 0
        for (i, id) in ids.enumerated() {
            let len: Int
            if let l = tokenByteLength[id] {
                len = l
            } else {
                len = tokenizer.decodeBytes([id], skipSpecialTokens: false,
                                            preserving: []).count
                tokenByteLength[id] = len
            }
            if bytes + len > limit { return i }
            bytes += len
        }
        return ids.count
    }

    static func commonPrefix(_ a: [UInt8], _ b: [UInt8]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    /// TEMPO9_TOOL_LADDER=0 pins only the end of the tool list.
    ///
    /// ON for the worst case, at a measured cost in the common one
    /// (Qwen3.5-9B, M5 Pro, interleaved arms x2, turn-2 prefill):
    ///   WaitForMcpServers race (2.2k from the end)  22.15 -> 20.59 s
    ///   transient tool ~6k from the end             32.12 -> 28.55 s
    ///   tools only appended (no mid-list change)    +0.36 s per session
    ///     (four more snapshots captured on turns 1 and 2; nothing after)
    /// plus ~20 ms of host rendering per request and 4 x 25 MB pinned
    /// for the breakpoint's TTL.
    static let toolLadderEnabled =
        ProcessInfo.processInfo.environment["TEMPO9_TOOL_LADDER"] != "0"
    /// How far before the end of the tool list (in tokens) the extra tool
    /// boundary pins go.  A change x tokens before the end resumes from the
    /// first rung at or beyond x: at most one rung gap (< x + one tool)
    /// short, for x up to 8k; four snapshots.
    static let toolLadderDistances = [1024, 2048, 4096, 8192]

    /// For each distance (tokens before the end of the tool list), the
    /// index k of the tool whose leading boundary is the first one at or
    /// beyond that distance, by estimated rendered size.  Sorted, deduped;
    /// a distance longer than the whole list (past tool 1) gets no rung.
    static func toolLadderRungs(sizes: [Double], bytesPerToken: Double,
                                distances: [Int]) -> [Int] {
        var rungs = Set<Int>()
        for distance in distances {
            let want = Double(distance) * bytesPerToken
            var k = sizes.count, tail = 0.0
            while k > 1 && tail < want { k -= 1; tail += sizes[k] }
            if tail >= want { rungs.insert(k) }
        }
        return rungs.sorted()
    }

    /// A user turn that exists only to make a prefix renderable; two probes
    /// with different text locate where user content begins.
    static func probeUser(_ n: Int) -> [String: Any] {
        ["role": "user", "content": n == 1 ? "\u{1}" : "\u{2}"]
    }

    /// A copy of `tool` under another name: same shape, so the template
    /// renders it the way it renders any tool.
    static func probeTool(like tool: [String: Any]) -> [String: Any] {
        var probe = tool
        if var f = tool["function"] as? [String: Any] {
            f["name"] = "tempo9_cache_probe"
            probe["function"] = f
        } else {
            probe["name"] = "tempo9_cache_probe"
        }
        return probe
    }

    /// Byte offset of the last occurrence of `needle`, found iteratively.
    ///
    /// Deliberately not `range(of:options:.backwards)` — see the call site.
    /// A plain reverse scan over UTF-8: no recursion, no Foundation, and the
    /// callers only need an ordering between two markers, so an offset does.
    static func lastByteIndex(of needle: String, in haystack: String) -> Int? {
        let n = Array(needle.utf8)
        if n.isEmpty { return nil }
        let h = Array(haystack.utf8)
        if n.count > h.count { return nil }
        var i = h.count - n.count
        while i >= 0 {
            var k = 0
            while k < n.count, h[i + k] == n[k] { k += 1 }
            if k == n.count { return i }
            i -= 1
        }
        return nil
    }

}


/// FIFO mutual exclusion for async callers.  `releaseFromSync` exists so a
/// non-async `defer` can release; ordering with the next acquire is
/// preserved by the actor's mailbox.
actor TurnGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !busy { busy = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }

    private func release() {
        if waiters.isEmpty { busy = false }
        else { waiters.removeFirst().resume() }
    }

    nonisolated func releaseFromSync() {
        Task { await self.release() }
    }
}
