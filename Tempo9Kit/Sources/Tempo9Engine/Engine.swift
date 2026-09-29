// Copyright (c) 2026 Jiejing Zhang.
//
// The engine, in-process, as Swift sees it.
//
// The C ABI is deliberately small and polling-based: `te9_request_wait` blocks
// for a deadline and hands back whatever tokens arrived. That shape is right
// for a C boundary and wrong for app code, so everything awkward about it is
// confined to this file:
//
//   * the wait loop becomes an AsyncStream, so a caller writes `for await`
//   * cancellation works, because a wait that times out with nothing is a
//     normal tick rather than an error -- that is what gives the loop a place
//     to notice Task.isCancelled between tokens
//   * handles are freed on every exit path, including cancellation and throw
//
// Threading: an Engine is safe from multiple threads; a single request is
// driven from one. Each generate() therefore owns its request end to end.

import Foundation
import CTempo9Engine

/// LocalizedError as well as CustomStringConvertible: without
/// errorDescription, `localizedDescription` -- which is what a catch-all
/// `catch { ... error.localizedDescription }` reaches for -- reads
/// "The operation couldn't be completed. (Tempo9Engine.Tempo9Error error
/// 0.)": the case index, and neither the status nor the detail.  That is
/// the text a client got for an engine that had said, precisely,
/// `max_length (33700) > engine_max_length_ (32768)`.
public enum Tempo9Error: Error, CustomStringConvertible, LocalizedError {
    case engine(status: Int32, detail: String)

    public var description: String {
        switch self {
        case let .engine(status, detail):
            return detail.isEmpty
                ? "tempo9 error \(status)"
                : "tempo9 error \(status): \(detail)"
        }
    }

    public var errorDescription: String? { description }
}

/// Why generation stopped. Comes from the engine, never inferred by comparing
/// a token count against a limit -- that inference reports a KV-eviction abort
/// as a truncation, which both Python servers shipped before it was fixed.
public enum FinishReason: Sendable {
    case none, eos, length, stop, interrupted

    init(_ raw: te9_finish_reason) {
        switch raw {
        case TE9_FINISH_EOS: self = .eos
        case TE9_FINISH_LENGTH: self = .length
        case TE9_FINISH_STOP: self = .stop
        case TE9_FINISH_INTERRUPTED: self = .interrupted
        default: self = .none
        }
    }
}

public struct SamplingConfig: Sendable {
    /// NEW tokens, not a total budget -- the prompt is added on inside.
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int32
    public var repetitionPenalty: Float
    public var seed: UInt64
    public var doSample: Bool
    /// MTP draft depth. 0 uses the model's default.
    public var speculationK: Int32
    public var stopTokenIDs: [Int64]
    /// Must be set. The engine's own default is another model's EOS, so
    /// leaving this nil produces generation that runs past the end of the
    /// answer and starts over.
    public var eosTokenID: Int64?
    /// Guided decoding: "json_object" or "json_schema" (+ `responseSchema`,
    /// a JSON Schema document). nil = free text.
    public var responseFormat: String? = nil
    public var responseSchema: String? = nil

    public init(maxTokens: Int = 512, temperature: Float = 0.7,
                topP: Float = 0.9, topK: Int32 = 0,
                repetitionPenalty: Float = 1.0,
                // A FRESH seed per request. The engine takes the seed
                // literally, so a constant one makes sampling deterministic:
                // the same prompt draws the same tokens forever, and a
                // thinking pass that wanders into a repetition ("k8s", "k8s",
                // "k8s", …) wanders into the same one every time. That is the
                // degeneration the sampling presets were meant to prevent.
                seed: UInt64 = UInt64.random(in: 1...UInt64(Int32.max)),
                doSample: Bool = true, speculationK: Int32 = 0,
                stopTokenIDs: [Int64] = [], eosTokenID: Int64? = nil) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.doSample = doSample
        self.speculationK = speculationK
        self.stopTokenIDs = stopTokenIDs
        self.eosTokenID = eosTokenID
    }
}

/// One media block: an image's embedding, or an utterance's.
///
/// The engine stores it under `mediaTokenID`, because MultiMediaInfo is keyed
/// by token id and that is how each block is found where its placeholder run
/// is. Image and audio differ only in that id and in what produced the rows.
public struct MediaBlock: Sendable {
    public var data: Data
    public var tokenCount: Int
    public var hidden: Int
    public var isFloat16: Bool
    public var mediaTokenID: Int64
    /// Qwen3-VL deepstack features, LAYER-major
    /// [numDeepstackLayers, tokenCount, hidden], same dtype as `data`.
    /// Empty for models without them.
    public var deepstack: Data = Data()
    public var numDeepstackLayers: Int = 0

    public init(data: Data, tokenCount: Int, hidden: Int,
                mediaTokenID: Int64, isFloat16: Bool = false,
                deepstack: Data = Data(), numDeepstackLayers: Int = 0) {
        self.data = data
        self.tokenCount = tokenCount
        self.hidden = hidden
        self.mediaTokenID = mediaTokenID
        self.isFloat16 = isFloat16
        self.deepstack = deepstack
        self.numDeepstackLayers = numDeepstackLayers
    }
}

/// Vision-tower output for one image. VisionTowerKit produces this; the engine
/// never sees pixels.
public struct ImageEmbeddings: Sendable {
    /// Every block in the request, in prompt order. One prompt can carry
    /// more than one -- "look at this and answer what I am asking" is an
    /// image block and an audio block.
    public var blocks: [MediaBlock]
    /// 0 lets the engine derive one, so prefix-cache reuse is never silently
    /// lost by omitting it.
    public var contentHash: UInt64
    /// Interleaved M-RoPE table, int32, [3, seq] row major, built over the
    /// EXPANDED ids. VisionTowerKit has both halves:
    /// TowerAux.expandImagePlaceholder then TowerAux.llmMRoPEPositions.
    /// Without it the image tokens carry text positions and the model reads
    /// the picture as a flat run of words — no error, just a wrong answer.
    public var mropePositions: [Int32]
    /// Whether this model's media tokens use M-RoPE at all.
    ///
    /// Qwen3.5 does. Gemma 4 does NOT: its image run carries ordinary
    /// sequential positions and gets its two-dimensional structure from
    /// intra-block bidirectional attention instead.
    ///
    /// It has no default, on purpose. "Empty table means no M-RoPE" would
    /// make a caller that merely FORGOT the table indistinguishable from one
    /// that declared it unnecessary — and the engine cannot tell either, so
    /// the model would answer fluently about an image it misread. Making it
    /// a required argument turns that into a compile error.
    public var usesMRoPE: Bool

    public init(blocks: [MediaBlock], mropePositions: [Int32],
                usesMRoPE: Bool, contentHash: UInt64 = 0) {
        self.blocks = blocks
        self.contentHash = contentHash
        self.mropePositions = mropePositions
        self.usesMRoPE = usesMRoPE
    }
}

public struct GenerationStats: Sendable {
    public var promptTokens: Int
    public var generatedTokens: Int
    public var prefixCacheHitTokens: Int
    public var prefillMilliseconds: Double
    public var decodeMilliseconds: Double
}

/// One chunk of generated ids, plus the terminal state when it has arrived.
public struct GenerationChunk: Sendable {
    public var tokenIDs: [Int64]
    public var isFinal: Bool
    public var finishReason: FinishReason
    /// Present only on the FINAL chunk: the engine's own accounting for
    /// this request.  Everything here was already being computed inside
    /// the engine and simply never crossed the C ABI into Swift, so hosts
    /// have been re-deriving worse versions of it -- a client-side TTFT
    /// that carries tokenisation and scheduling, and no per-request cache
    /// hit at all.
    public var stats: RequestStats?
}

/// Per-request timing and cache accounting, straight from the engine.
///
/// `prefillMs` is the engine's time-to-first-token and `decodeMs` is what
/// followed it, so the two separate the phases without a profiler -- which
/// is the measurement the long-context decode investigation needs, and the
/// one a wall-clock stopwatch in the host cannot give.
///
/// `prefixCacheHitTokens` is what OpenAI calls
/// `usage.prompt_tokens_details.cached_tokens`; reporting it makes every
/// prefix-cache claim self-verifying instead of resting on a log grep.
public struct RequestStats: Sendable {
    public var promptTokens: Int64
    public var generatedTokens: Int64
    public var prefixCacheHitTokens: Int64
    public var prefillMs: Double
    public var decodeMs: Double
}

/// What the engine is doing right now, for the inspector and bug reports.
///
/// Rates are over TOKENS, not requests: a 300-token prompt that reuses 256
/// cached tokens is 85% hit, and counting it as one hit or one miss would
/// throw away the number that predicts the wait.
public struct EngineStats: Sendable {
    public var totalTokens: Int64
    public var freeTokens: Int64
    public var spanSize: Int64
    public var runningRequests: Int
    public var pendingRequests: Int
    public var generatedTokens: Int64
    public var prefillTokens: Int64
    public var prefixCacheHitTokens: Int64
    public var prefixCacheMissTokens: Int64
    public var prefixCacheHitRate: Float
    public var tokenUsage: Float
}

public final class Engine: @unchecked Sendable {
    private let handle: te9_engine_t
    private let modelName: String

    /// Engine build string, for a runtime inspector and bug reports.
    public static var version: String { String(cString: te9_version()) }

    /// "metal" or "cpu" — which GEMM path is actually running.
    ///
    /// Reported by the engine rather than assumed by the UI. A silent
    /// fallback to the CPU is four times slower and otherwise invisible:
    /// the app answers correctly, and a badge printing "Apple GPU" from a
    /// literal keeps printing it.
    public static var gemmBackend: String {
        String(cString: te9_gemm_backend())
    }

    /// Build the C media chain and keep every pointer alive for one call.
    ///
    /// One contiguous payload with offsets rather than nesting
    /// withUnsafeBytes once per block: the block count is not known at
    /// compile time, and recursion to build N nested scopes would be a lot
    /// of machinery for "keep these bytes alive across one call". The C side
    /// copies before it returns.
    private static func withMediaChain(
        _ images: ImageEmbeddings,
        _ body: (UnsafePointer<te9_image_embeds>?) throws -> Void) throws {
        var payload = [UInt8]()
        var spans: [(offset: Int, count: Int)] = []
        var dsSpans: [(offset: Int, count: Int)] = []
        for b in images.blocks {
            spans.append((payload.count, b.data.count))
            payload.append(contentsOf: b.data)
            dsSpans.append((payload.count, b.deepstack.count))
            payload.append(contentsOf: b.deepstack)
        }
        // The structs need stable addresses too: each one's `next` points at
        // the following element, so the array cannot be a temporary that
        // moves between being filled and being passed.
        var embeds = [te9_image_embeds](repeating: te9_image_embeds(),
                                       count: images.blocks.count)
        try images.mropePositions.withUnsafeBufferPointer { mrope in
            try payload.withUnsafeBytes { raw in
                try embeds.withUnsafeMutableBufferPointer { chain in
                    let base = chain.baseAddress
                    for i in 0..<chain.count {
                        let b = images.blocks[i]
                        chain[i].struct_size =
                            MemoryLayout<te9_image_embeds>.size
                        chain[i].data =
                            raw.baseAddress?.advanced(by: spans[i].offset)
                        chain[i].byte_count = spans[i].count
                        chain[i].token_count = Int64(b.tokenCount)
                        chain[i].hidden = Int64(b.hidden)
                        chain[i].is_float16 = b.isFloat16 ? 1 : 0
                        chain[i].image_token_id = b.mediaTokenID
                        chain[i].num_deepstack_layers =
                            Int32(b.numDeepstackLayers)
                        chain[i].deepstack_data = b.numDeepstackLayers > 0
                            ? raw.baseAddress?.advanced(by: dsSpans[i].offset)
                            : nil
                        chain[i].deepstack_byte_count =
                            b.numDeepstackLayers > 0 ? dsSpans[i].count : 0
                        chain[i].next = i + 1 < chain.count
                            ? UnsafePointer(base?.advanced(by: i + 1)) : nil
                    }
                    // Request-wide, and therefore on the HEAD block only: one
                    // hash keys the whole prompt's prefix cache, and one
                    // M-RoPE table covers the whole sequence.
                    chain[0].content_hash = images.contentHash
                    // The C side wants NULL/0 alongside no_mrope=1, and
                    // rejects the half-set combination — so an empty table
                    // and a stale pointer cannot be confused for each other.
                    chain[0].no_mrope = images.usesMRoPE ? 0 : 1
                    chain[0].mrope_positions =
                        images.usesMRoPE ? mrope.baseAddress : nil
                    chain[0].mrope_position_count =
                        images.usesMRoPE ? mrope.count : 0
                    try body(UnsafePointer(base))
                }
            }
        }
    }

    private static func check(_ status: te9_status,
                              _ what: String) throws {
        guard status != TE9_OK else { return }
        // te9_last_error is thread-local and only valid until this thread's
        // next failure, so copy it into the error now.
        throw Tempo9Error.engine(status: Int32(status.rawValue),
                                    detail: "\(what): "
                                        + String(cString: te9_last_error()))
    }

    /// Loads a model and starts it. `graphPath` is a prebuilt .asgraph, or
    /// empty when `weightsPath` is a .gguf: the engine then builds the graph
    /// itself (C++ builder, leaf-identical to the Python one on all 16
    /// support-matrix models) and caches it keyed by the gguf's size+mtime.
    public init(modelName: String,
                graphPath: String,
                weightsPath: String,
                computeUnit: String = "CPU:0",
                maxLength: Int64 = 8192,
                maxBatch: Int32 = 1,
                enablePrefixCache: Bool = true) throws {
        var engine: te9_engine_t?
        try Engine.check(te9_engine_create(&engine), "te9_engine_create")
        guard let engine else {
            throw Tempo9Error.engine(status: -1, detail: "null engine")
        }
        self.handle = engine
        self.modelName = modelName

        // Every string has to outlive the call, so the withCString nest is
        // load-bearing rather than style.
        do {
            try modelName.withCString { nameC in
                try graphPath.withCString { graphC in
                    try weightsPath.withCString { weightsC in
                        try computeUnit.withCString { unitC in
                            var cfg = te9_model_config()
                            cfg.struct_size = MemoryLayout<te9_model_config>.size
                            cfg.model_name = nameC
                            // Empty graphPath => nil: the engine builds the
                            // graph from the .gguf itself.
                            cfg.graph_path = graphPath.isEmpty ? nil : graphC
                            cfg.weights_path = weightsC
                            cfg.compute_unit = unitC
                            cfg.max_length = maxLength
                            cfg.max_batch = maxBatch
                            cfg.enable_prefix_cache = enablePrefixCache ? 1 : 0
                            try Engine.check(te9_engine_build_model(engine, &cfg),
                                             "build_model")
                        }
                    }
                }
            }
            try modelName.withCString {
                try Engine.check(te9_engine_start_model(engine, $0),
                                 "start_model")
            }
            started = true
        }
        // No destroy here.  Every stored property is initialised by now, so
        // a throw from this init still runs deinit on the object -- and a
        // destroy in a catch block handed deinit a freed handle: its
        // te9_engine_stop_model dereferenced garbage inside
        // AsEngine::StopModel and the process died with SIGSEGV ~2 s after
        // start, before the refusal ever reached stderr.  deinit owns the
        // handle, and stops only what was started.
    }

    /// Set once start_model succeeds.  A model whose build was refused was
    /// never started, and stopping it is not merely useless: it is the call
    /// that crashed the E4B refusal.
    private var started = false

    deinit {
        if started {
            modelName.withCString {
                _ = te9_engine_stop_model(handle, $0)
                _ = te9_engine_release_model(handle, $0)
            }
        }
        te9_engine_destroy(handle)
    }

    /// A snapshot of the engine's counters. Cheap: it copies a struct the
    /// engine already maintains, so polling it once a second is fine.
    public func stats() throws -> EngineStats {
        var raw = te9_engine_stats()
        raw.struct_size = MemoryLayout<te9_engine_stats>.size
        try modelName.withCString {
            try Engine.check(te9_engine_get_stats(handle, $0, &raw), "get_stats")
        }
        return EngineStats(
            totalTokens: raw.total_token, freeTokens: raw.free_token,
            spanSize: raw.span_size,
            runningRequests: Int(raw.running_request),
            pendingRequests: Int(raw.pending_request),
            generatedTokens: raw.total_generated_token,
            prefillTokens: raw.total_prefill_token,
            prefixCacheHitTokens: raw.prefix_cache_hit_token,
            prefixCacheMissTokens: raw.prefix_cache_miss_token,
            prefixCacheHitRate: raw.prefix_cache_hit_rate,
            tokenUsage: raw.token_usage_percentage)
    }

    /// Generate from already-tokenized ids. Tokenization and chat-template
    /// rendering belong to GGUFKit / ChatTemplateKit -- keeping them out of
    /// the engine is what lets it ship without Python.
    public func generate(inputIDs: [Int64],
                         config: SamplingConfig = .init(),
                         images: ImageEmbeddings? = nil)
        -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            let work = Task.detached { [handle, modelName] in
                var request: te9_request_t?
                do {
                    request = try Engine.startRequest(
                        handle: handle, modelName: modelName,
                        inputIDs: inputIDs, config: config, images: images)
                } catch {
                    continuation.finish(throwing: error)
                    return
                }
                guard let request else {
                    continuation.finish(throwing: Tempo9Error.engine(
                        status: -1, detail: "null request"))
                    return
                }
                defer { te9_request_release(request) }

                var buffer = [Int64](repeating: 0, count: 256)
                while true {
                    if Task.isCancelled {
                        _ = te9_request_stop(request)
                        continuation.finish()
                        return
                    }
                    var produced = 0
                    let waitStatus = buffer.withUnsafeMutableBufferPointer {
                        te9_request_wait(request, 100, $0.baseAddress,
                                        $0.count, &produced)
                    }
                    if waitStatus != TE9_OK {
                        continuation.finish(throwing: Tempo9Error.engine(
                            status: Int32(waitStatus.rawValue),
                            detail: "wait: " + String(cString: te9_last_error())))
                        return
                    }

                    var state = TE9_GEN_RUNNING
                    _ = te9_request_status(request, &state)
                    let done = state != TE9_GEN_RUNNING

                    // A wait that returns nothing and is not finished is an
                    // ordinary tick; emitting an empty chunk for it would make
                    // every consumer filter noise.
                    if produced > 0 || done {
                        var reason = TE9_FINISH_NONE
                        var stats: RequestStats?
                        if done {
                            _ = te9_request_finish_reason(request, &reason)
                            // Only on the final chunk: before the request
                            // finishes these counters are still moving, and
                            // a mid-flight read would report a prefill that
                            // has not ended.
                            var raw = te9_request_stats()
                            raw.struct_size = MemoryLayout<te9_request_stats>.size
                            if te9_request_get_stats(request, &raw) == TE9_OK {
                                stats = RequestStats(
                                    promptTokens: raw.prompt_tokens,
                                    generatedTokens: raw.generated_tokens,
                                    prefixCacheHitTokens: raw.prefix_cache_hit_tokens,
                                    prefillMs: raw.prefill_ms,
                                    decodeMs: raw.decode_ms)
                            }
                        }
                        continuation.yield(GenerationChunk(
                            tokenIDs: Array(buffer[0..<produced]),
                            isFinal: done,
                            finishReason: FinishReason(reason),
                            stats: stats))
                    }
                    if done { break }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in work.cancel() }
        }
    }

    private static func startRequest(handle: te9_engine_t,
                                     modelName: String,
                                     inputIDs: [Int64],
                                     config: SamplingConfig,
                                     images: ImageEmbeddings?) throws
        -> te9_request_t? {
        var gen = te9_generate_config()
        gen.struct_size = MemoryLayout<te9_generate_config>.size
        gen.max_tokens = Int64(config.maxTokens)
        gen.temperature = config.temperature
        gen.top_p = config.topP
        gen.top_k = config.topK
        gen.repetition_penalty = config.repetitionPenalty
        gen.seed = config.seed
        gen.do_sample = config.doSample ? 1 : 0
        gen.speculation_k = config.speculationK
        gen.eos_token_id = config.eosTokenID ?? 0
        // C strings for the guided-decoding fields must outlive the call;
        // strdup here, free after te9_request_start returns (below).
        let fmtC: UnsafeMutablePointer<CChar>? = config.responseFormat.flatMap { strdup($0) }
        let schemaC: UnsafeMutablePointer<CChar>? = config.responseSchema.flatMap { strdup($0) }
        defer { free(fmtC); free(schemaC) }
        gen.response_format = UnsafePointer(fmtC)
        gen.response_schema = UnsafePointer(schemaC)

        var request: te9_request_t?
        // The C side copies everything before returning, so none of these
        // pointers has to outlive the call -- that is the one ownership rule
        // this binding does not have to enforce.
        try config.stopTokenIDs.withUnsafeBufferPointer { stops in
            gen.stop_token_ids = stops.baseAddress
            gen.stop_token_count = stops.count
            try inputIDs.withUnsafeBufferPointer { ids in
                try modelName.withCString { nameC in
                    if let images, !images.blocks.isEmpty {
                        try Engine.withMediaChain(images) { head in
                            try Engine.check(
                                te9_request_start(handle, nameC,
                                                 ids.baseAddress, ids.count,
                                                 &gen, head, &request),
                                "request_start")
                        }
                    } else {
                        try Engine.check(
                            te9_request_start(handle, nameC,
                                             ids.baseAddress, ids.count,
                                             &gen, nil, &request),
                            "request_start")
                    }
                }
            }
        }
        return request
    }
}
