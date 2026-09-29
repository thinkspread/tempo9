// Copyright (c) 2026 Jiejing Zhang.
//
// The facade: image in, LLM-ready embedding out.
//
//   CGImage ──preprocess──▶ patches ──Core ML (gpu|ane)──▶ merger embedding
//                                └── L1 cache, content addressed ──┘
//
// Tier policy mirrors the Python server's: the GPU tower is ~7x faster but
// contends with a Metal LLM (concurrent decode x0.24), while the ANE tower is
// nearly free next to one. So "fastest" and "cheapest while the LLM is busy"
// are different tiers, and `.auto` picks per call.

import CoreGraphics
import Foundation

public enum VisionTowerError: LocalizedError {
    case missingAsset(String)
    case imageDecodeFailed(String)
    case gridTooLarge(patches: Int, bucket: Int)
    case extremeAspectRatio(height: Int, width: Int)
    case placeholderMismatch(placeholders: Int, images: Int)
    case runLengthMismatch(run: Int, expected: Int)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingAsset(let path):
            return "vision tower asset not found: \(path)"
        case .imageDecodeFailed(let what):
            return "could not decode image (\(what))"
        case .gridTooLarge(let patches, let bucket):
            return "image needs \(patches) patches, largest bucket is \(bucket)"
        case .extremeAspectRatio(let h, let w):
            return "aspect ratio must be under 200:1, got \(h)x\(w)"
        case .placeholderMismatch(let p, let i):
            return "\(p) image placeholders but \(i) images"
        case .runLengthMismatch(let run, let expected):
            return "image token run \(run) != grid tokens \(expected)"
        case .predictionFailed(let what):
            return "vision tower prediction failed: \(what)"
        }
    }
}

public enum TierPolicy: Sendable {
    /// GPU while the LLM is idle, ANE while a generation is in flight.
    case auto
    case fixed(ComputeTier)
}

public struct AudioEncoding {
    public var embedding: [Float]      // [tokens, outHidden] fp32
    public var tokens: Int
    public var outHidden: Int
    public var seconds: Double
    /// 40 ms a token, by construction (640 samples at 16 kHz).
    public var duration: Double { Double(tokens) * 0.04 }
}

public struct Encoding {
    public var embedding: [Float]      // [tokens, outHidden] fp32
    /// Qwen3-VL deepstack feature maps, LAYER-major
    /// [numDeepstackLayers, tokens, outHidden] fp32; empty for towers
    /// without them. Kept beside `embedding` rather than fused into it so
    /// outHidden keeps meaning "what the LLM embeds" -- the tower/model
    /// pairing check compares it against the LLM's hidden size, and a
    /// fused width would pass nothing.
    public var deepstack: [Float] = []
    public var numDeepstackLayers: Int = 0
    public var tokens: Int
    public var outHidden: Int
    public var gridH: Int
    public var gridW: Int
    public var contentKey: Int64
    public var tier: ComputeTier
    public var cacheHit: Bool
    public var preprocessSeconds: Double
    public var towerSeconds: Double
    public var bucketN: Int
}

public final class VisionTower {
    public let directory: URL
    public let policy: TierPolicy
    /// Which media convention this tower's model follows. The caller needs it
    /// to build the prompt, and it belongs here because the tower and the
    /// convention are two halves of one pairing — a Gemma tower feeding a
    /// Qwen layout produces a fluent answer about nothing.
    public let layout: MediaLayout
    private let cache: EmbeddingCache
    private var towers: [ComputeTier: CoreMLTower] = [:]
    private let towersLock = NSLock()
    private var preprocessConfig: PreprocessConfig
    /// Gemma 4 has no Core ML tower at all: its whole front end is eleven
    /// tensors in the mmproj, run here in-process. So this class is the
    /// facade over BOTH, and every existing call site keeps working.
    private let gemma: Gemma4Tower?
    /// Qwen2.5-Omni hears through an exported Core ML graph rather than an
    /// in-process port: its encoder is 32 layers of 1280, not Gemma's eleven
    /// tensors, and it belongs on the Neural Engine where it does not
    /// contend with the LLM.
    private let omniAudio: OmniAudioTower?
    private let omniVision: OmniVisionTower?

    /// Set by the host so `.auto` knows whether the LLM is busy.
    public var isLLMBusy: () -> Bool = { false }

    /// Gemma 4: `mmproj` is the 116 MB projector .gguf beside the model.
    ///
    /// `useGPU` picks MPSMatrixMultiplication over Accelerate for the two
    /// GEMMs — 5.5 ms against 16.1 for a 280-token image. It is not `.auto`
    /// like the Core ML tier policy is: there is no ANE path to fall back
    /// to, and 16 ms of tower next to 1.2 s of inference is not worth a
    /// scheduling decision.
    /// Qwen2.5-Omni: `directory` holds the exported audio tower, and the
    /// vision tower beside it once that is exported too.
    public init(omniDirectory directory: URL,
                policy: TierPolicy = .fixed(.ane),
                cache: EmbeddingCache = EmbeddingCache()) throws {
        self.directory = directory
        self.policy = policy
        self.layout = .qwen25omni
        self.cache = cache
        self.gemma = nil
        self.preprocessConfig = PreprocessConfig()
        let tier: ComputeTier
        if case .fixed(let t) = policy { tier = t } else { tier = .ane }
        self.omniAudio = try OmniAudioTower(
            directory: directory.appendingPathComponent("coreml-audio"),
            tier: tier)
        // Optional: an install can hear before it can see, and that is a
        // smaller model rather than a broken one.
        let visDir = directory.appendingPathComponent("coreml-vision")
        self.omniVision = FileManager.default.fileExists(
            atPath: visDir.appendingPathComponent("tower_meta.json").path)
            ? try OmniVisionTower(directory: visDir, tier: tier) : nil
    }

    public init(gemma4Mmproj mmproj: URL, useGPU: Bool = true,
                cache: EmbeddingCache = EmbeddingCache()) throws {
        self.directory = mmproj
        self.policy = .fixed(.gpu)
        self.layout = .gemma4
        self.cache = cache
        self.preprocessConfig = PreprocessConfig()
        let matmul: Gemma4MatMul =
            (useGPU ? Gemma4MetalMatMul() : nil) ?? AccelerateMatMul()
        self.omniAudio = nil
        self.omniVision = nil
        self.gemma = try Gemma4Tower(mmprojPath: mmproj.path, matmul: matmul)
    }

    public init(directory: URL, policy: TierPolicy = .auto,
                cache: EmbeddingCache = EmbeddingCache()) throws {
        self.directory = directory
        self.policy = policy
        self.cache = cache
        self.gemma = nil
        self.omniAudio = nil
        self.omniVision = nil
        self.preprocessConfig = PreprocessConfig()

        let eager: ComputeTier
        switch policy {
        case .auto: eager = .gpu
        case .fixed(let tier): eager = tier
        }
        // Constructed directly rather than through load(tier:): layout is a
        // stored let that depends on the tower's meta, and Swift will not
        // let an instance method run before every stored property has a
        // value. The tower still lands in the cache dict below.
        let tower = try CoreMLTower(directory: directory, tier: eager)
        // The meta says which LLM convention this tower pairs with;
        // absent means the facade's historical default, Qwen3.5.
        self.layout = tower.meta.layout == "qwen3vl" ? .qwen3vl : .qwen35
        towers[eager] = tower

        // Cap preprocessing at the largest exported bucket, so an oversized
        // image is scaled down rather than patchified past every bucket and
        // rejected after the expensive part has already run.
        if let patch = tower.meta.patchSize { preprocessConfig.patchSize = patch }
        if let t = tower.meta.temporalPatchSize { preprocessConfig.temporalPatchSize = t }
        if let m = tower.meta.spatialMergeSize { preprocessConfig.mergeSize = m }
        if let mean = tower.meta.imageMean { preprocessConfig.imageMean = mean }
        if let std = tower.meta.imageStd { preprocessConfig.imageStd = std }
        preprocessConfig.maxPixels =
            tower.largestBucket * preprocessConfig.patchSize * preprocessConfig.patchSize

        if case .auto = policy {
            // Unlike the Python runtime, this really is concurrent: no GIL
            // sits between the ANE load and the rest of the process.
            Thread.detachNewThread { [weak self] in
                _ = try? self?.load(tier: .ane)
            }
        }
    }

    @discardableResult
    private func load(tier: ComputeTier) throws -> CoreMLTower {
        towersLock.lock()
        if let existing = towers[tier] {
            towersLock.unlock()
            return existing
        }
        towersLock.unlock()

        let tower = try CoreMLTower(directory: directory, tier: tier)

        towersLock.lock()
        towers[tier] = tower
        towersLock.unlock()
        return tower
    }

    private func pickTier() -> ComputeTier {
        switch policy {
        case .fixed(let tier): return tier
        case .auto:
            towersLock.lock()
            let aneReady = towers[.ane] != nil
            towersLock.unlock()
            return (isLLMBusy() && aneReady) ? .ane : .gpu
        }
    }

    /// Can this tower see? False for an Omni install whose vision half has
    /// not been exported -- it can still hear, and saying so beats failing
    /// deep inside a Core ML load with a missing-asset path.
    public var canSeeImages: Bool { omniAudio == nil || omniVision != nil }

    public func encode(image: CGImage) throws -> Encoding {
        if let gemma { return try encodeGemma(image: image, tower: gemma) }
        if let omniVision { return try omniVision.encode(image: image) }
        if omniAudio != nil {
            throw VisionTowerError.missingAsset(
                directory.appendingPathComponent("coreml-vision").path
                + " — this Omni install can hear but not see; the vision "
                + "tower has not been exported")
        }
        let preStart = Date()
        let pre = try Preprocess.run(image: image, config: preprocessConfig)
        let preSeconds = Date().timeIntervalSince(preStart)

        let tier = pickTier()
        let tower = try load(tier: tier)
        // Fingerprint only: resolving the bucket must not load its model,
        // or a cache hit would still pay for one.
        let bucket = try tower.bucketInfo(patchCount: pre.patchCount)
        let key = ContentKey.make(patches: pre.patches, gridH: pre.gridH,
                                  gridW: pre.gridW,
                                  fingerprint: bucket.fingerprint)

        if let hit = cache.get(key) {
            var enc = Self.split(fused: hit, meta: tower.meta,
                                 tokens: pre.patchCount / 4)
            enc.gridH = pre.gridH; enc.gridW = pre.gridW
            enc.contentKey = key; enc.tier = tier; enc.cacheHit = true
            enc.preprocessSeconds = preSeconds
            enc.bucketN = bucket.n
            return enc
        }

        let towerStart = Date()
        let result = try tower.encode(patches: pre.patches,
                                      gridH: pre.gridH, gridW: pre.gridW)
        let towerSeconds = Date().timeIntervalSince(towerStart)
        // The cache stores the FUSED row (main + deepstack): one entry, one
        // key, and the split below is cheap next to the tower.
        let stored = cache.put(key, embedding: result.embedding)

        var enc = Self.split(fused: stored, meta: tower.meta,
                             tokens: pre.patchCount / 4)
        enc.gridH = pre.gridH; enc.gridW = pre.gridW
        enc.contentKey = key; enc.tier = tier; enc.cacheHit = false
        enc.preprocessSeconds = preSeconds
        enc.towerSeconds = towerSeconds
        enc.bucketN = result.bucketN
        return enc
    }

    /// Un-fuse a tower row [tokens, outHidden*(1+L)] into the LLM embedding
    /// and the layer-major deepstack planes. L == 0 passes straight through.
    private static func split(fused: [Float], meta: TowerMeta, tokens: Int)
        -> Encoding {
        let oh = meta.outHidden
        let layers = meta.numDeepstack ?? 0
        if layers == 0 {
            return Encoding(embedding: fused, tokens: tokens, outHidden: oh,
                            gridH: 0, gridW: 0, contentKey: 0, tier: .cpu,
                            cacheHit: false, preprocessSeconds: 0,
                            towerSeconds: 0, bucketN: 0)
        }
        let row = oh * (1 + layers)
        var main = [Float](repeating: 0, count: tokens * oh)
        var ds = [Float](repeating: 0, count: layers * tokens * oh)
        fused.withUnsafeBufferPointer { f in
            main.withUnsafeMutableBufferPointer { m in
                ds.withUnsafeMutableBufferPointer { d in
                    for t in 0..<tokens {
                        let base = t * row
                        for c in 0..<oh { m[t * oh + c] = f[base + c] }
                        for l in 0..<layers {
                            let src = base + oh * (1 + l)
                            let dst = (l * tokens + t) * oh
                            for c in 0..<oh { d[dst + c] = f[src + c] }
                        }
                    }
                }
            }
        }
        var enc = Encoding(embedding: main, tokens: tokens, outHidden: oh,
                           gridH: 0, gridW: 0, contentKey: 0, tier: .cpu,
                           cacheHit: false, preprocessSeconds: 0,
                           towerSeconds: 0, bucketN: 0)
        enc.deepstack = ds
        enc.numDeepstackLayers = layers
        return enc
    }

    /// Same cache shape as the Core ML path: key over the preprocessed
    /// patches, look up, and only then run the tower. The grid is reported
    /// in MERGED units (20x14 = 280 tokens), which is what the prompt's
    /// expanded run has to match — Gemma has no un-merged patch grid to
    /// confuse it with.
    private func encodeGemma(image: CGImage, tower: Gemma4Tower)
        throws -> Encoding {
        let preStart = Date()
        let (patches, cols, rows) = try tower.imagePatches(image)
        let preSeconds = Date().timeIntervalSince(preStart)
        let tokens = cols * rows
        let key = ContentKey.make(patches: patches, gridH: rows, gridW: cols,
                                  fingerprint: "gemma4/" + tower.matmul.label)
        let tier: ComputeTier = tower.matmul.label == "gpu" ? .gpu : .cpu

        if let hit = cache.get(key) {
            return Encoding(embedding: hit, tokens: tokens,
                            outHidden: tower.hidden, gridH: rows, gridW: cols,
                            contentKey: key, tier: tier, cacheHit: true,
                            preprocessSeconds: preSeconds, towerSeconds: 0,
                            bucketN: tokens)
        }
        let towerStart = Date()
        let embedding = try tower.embedPatches(patches, cols: cols, rows: rows)
        let towerSeconds = Date().timeIntervalSince(towerStart)
        let stored = cache.put(key, embedding: embedding)
        return Encoding(embedding: stored, tokens: tokens,
                        outHidden: tower.hidden, gridH: rows, gridW: cols,
                        contentKey: key, tier: tier, cacheHit: false,
                        preprocessSeconds: preSeconds,
                        towerSeconds: towerSeconds, bucketN: tokens)
    }

    public func encode(imageAt url: URL) throws -> Encoding {
        try encode(image: try Preprocess.loadImage(url: url))
    }

    /// 16 kHz mono float in [-1, 1] -> one embedding per 40 ms.
    ///
    /// Nil when the loaded front end cannot hear. There is no cache: unlike a
    /// frame, the same utterance never arrives twice, so a key would cost a
    /// hash of the samples and save nothing.
    public func encodeAudio(_ samples: [Float]) throws -> AudioEncoding? {
        if let omniAudio { return try omniAudio.encode(samples) }
        guard let gemma, gemma.hasAudio else { return nil }
        let start = Date()
        let embedding = try gemma.embedAudio(samples)
        let tokens = embedding.count / gemma.hidden
        return AudioEncoding(embedding: embedding, tokens: tokens,
                             outHidden: gemma.hidden,
                             seconds: Date().timeIntervalSince(start))
    }

    /// Where this tower's compute actually runs, for the UI to report.
    ///
    /// Not the POLICY — the policy is a Core ML tier setting, and a badge
    /// showing "ane" for an encoder-free model whose GEMMs are on the GPU is
    /// the same class of untruth as a badge that says "Apple GPU" while every
    /// weight GEMM is on the CPU. Ask the tower, not the setting.
    public var backendLabel: String {
        if omniAudio != nil { return "ane" }
        if let gemma { return gemma.matmul.label }
        switch policy {
        case .fixed(let tier): return tier.rawValue
        case .auto: return "auto"
        }
    }

    public var cacheStats: (hits: Int, misses: Int, entries: Int, bytes: Int) {
        cache.stats
    }

    public var loadedTiers: [ComputeTier] {
        towersLock.lock()
        defer { towersLock.unlock() }
        return towers.keys.sorted { $0.rawValue < $1.rawValue }
    }
}
