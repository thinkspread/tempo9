// Copyright (c) 2026 Jiejing Zhang.
//
// The exported Qwen3.5 vision tower, run through CoreML.framework directly.
//
// No coremltools, no Python, no GIL. That last one is the point: in the
// Python runtime, loading the ANE tier on a "background" thread still pushed
// server startup out by 22s because coremltools holds the GIL through the
// load, and the same interpreter lock sits between a tower predict and
// anything else the process wants to do. Here the two tiers really are
// independent.

import CoreML
import Foundation

public enum ComputeTier: String, CaseIterable, Sendable {
    case gpu, gpuOnly, ane, cpu

    var computeUnits: MLComputeUnits {
        switch self {
        // .all lets Core ML use GPU + ANE + CPU; measured ~139ms at N=4096.
        case .gpu: return .all
        // No ANE at all. A host that runs its LLM on Metal and wants the
        // tower to behave identically on every M-series machine asks for
        // this: the ANE differs by generation, is fp16-only, and builds a
        // per-executable e5 compile cache — .cpuAndGPU trades a little
        // tower latency for a device profile that does not vary.
        case .gpuOnly: return .cpuAndGPU
        // .cpuAndNeuralEngine keeps it off the GPU, which is what makes it
        // nearly free alongside a Metal LLM decode (~1.0s at N=4096).
        case .ane: return .cpuAndNeuralEngine
        case .cpu: return .cpuOnly
        }
    }
}

public struct TowerMeta: Decodable {
    public struct Bucket: Decodable {
        public let n: Int
        public let mlpackage: String
        public let fingerprint: String
    }
    public let buckets: [Bucket]?
    public let bucketN: Int
    public let mlpackage: String
    public let fingerprint: String
    public let hidden: Int
    public let outHidden: Int
    public let patchDim: Int
    public let headDim: Int?
    public let patchSize: Int?
    public let temporalPatchSize: Int?
    public let spatialMergeSize: Int?
    public let ropeTheta: Float?
    public let imageMean: [Float]?
    public let imageStd: [Float]?
    public let posEmbedFile: String?
    public let posEmbedShape: [Int]?
    /// Qwen3-VL: the export fuses the deepstack features into merger_out,
    /// [n/4, outHidden * (1 + numDeepstack)] -- LLM embedding first, then
    /// the deepstack maps in layer order. nil/0 for towers without them.
    public let fusedOutHidden: Int?
    public let numDeepstack: Int?
    /// Which MediaLayout the paired LLM follows ("qwen3vl"); nil means the
    /// facade's historical default (qwen35).
    public let layout: String?

    enum CodingKeys: String, CodingKey {
        case buckets
        case bucketN = "bucket_n"
        case mlpackage, fingerprint, hidden
        case outHidden = "out_hidden"
        case patchDim = "patch_dim"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case ropeTheta = "rope_theta"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case posEmbedFile = "pos_embed_file"
        case posEmbedShape = "pos_embed_shape"
        case fusedOutHidden = "fused_out_hidden"
        case numDeepstack = "num_deepstack"
        case layout
    }

    /// Buckets ascending, with the single-bucket layout upgraded.
    public var sortedBuckets: [Bucket] {
        (buckets ?? [Bucket(n: bucketN, mlpackage: mlpackage,
                            fingerprint: fingerprint)]).sorted { $0.n < $1.n }
    }
}

/// One compiled tower at one compute tier, across all resolution buckets.
public final class CoreMLTower {
    /// A bucket, compiled and ready to load — but not loaded until used.
    private struct BucketRef {
        let n: Int
        let compiled: URL
        let fingerprint: String
    }

    public let tier: ComputeTier
    public let meta: TowerMeta
    private let buckets: [BucketRef]
    private let config: MLModelConfiguration
    /// Loaded lazily, keyed by bucket size. See `model(for:)`.
    private var loadedModels: [Int: MLModel] = [:]
    private let posEmbed: [Float]
    private let lock = NSLock()

    public private(set) var loadSeconds: Double = 0
    public private(set) var lastPredictSeconds: Double = 0
    public private(set) var lastBucketN: Int = 0

    public var largestBucket: Int { buckets.last?.n ?? 0 }

    public init(directory: URL, tier: ComputeTier, warmup: Bool = true) throws {
        self.tier = tier
        let started = Date()

        let metaURL = directory.appendingPathComponent("tower_meta.json")
        guard let metaData = try? Data(contentsOf: metaURL) else {
            throw VisionTowerError.missingAsset(metaURL.path)
        }
        self.meta = try JSONDecoder().decode(TowerMeta.self, from: metaData)

        // Raw fp32 table rather than the .npz beside it: parsing numpy's
        // container to read one matrix is not a dependency worth having.
        let peName = meta.posEmbedFile ?? "tower_pos_embed.f32"
        let peURL = directory.appendingPathComponent(peName)
        guard let peData = try? Data(contentsOf: peURL, options: .mappedIfSafe) else {
            throw VisionTowerError.missingAsset(
                peURL.path + " (re-run export_vit_coreml.py to emit it)")
        }
        self.posEmbed = peData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }

        let config = MLModelConfiguration()
        config.computeUnits = tier.computeUnits
        self.config = config

        // Compiling to a stable .mlmodelc is cheap once it exists; *loading*
        // is what costs, and on the ANE it is a full recompile of the model
        // into the engine's own program — a minute or more per bucket, and
        // measurably not cached (e5bundlecache stays at 2 MB across launches).
        //
        // So the buckets are only resolved here, not loaded. Eagerly loading
        // all three tripled startup for nothing: a 640x360 camera frame is
        // 880 patches, which is the 1024 bucket, and the other two were
        // compiled and warmed for a resolution the demo never sends.
        var refs: [BucketRef] = []
        for bucket in meta.sortedBuckets {
            let url = directory.appendingPathComponent(bucket.mlpackage)
            let compiled: URL
            if FileManager.default.fileExists(atPath: url.path) {
                compiled = try Self.compiledModel(for: url, in: directory)
            } else if let shipped = Self.precompiled(besides: url) {
                // A shipped tower carries only the COMPILED model. The
                // .mlpackage is the source Core ML compiles from, 847 MB of
                // it, and an app that has already been through the compiler
                // has no use for it — but the metadata still names it,
                // because that is what was exported.
                //
                // Not a fallback for a corrupt install: this is the normal
                // case inside a distributed .app, and the loader failing
                // here left the vision tower silently "cold" with a camera
                // that answered nothing.
                compiled = shipped
            } else {
                throw VisionTowerError.missingAsset(url.path)
            }
            refs.append(BucketRef(n: bucket.n, compiled: compiled,
                                  fingerprint: bucket.fingerprint))
        }
        self.buckets = refs
        self.loadSeconds = Date().timeIntervalSince(started)

        if warmup, let first = refs.first {
            // Only the smallest, which is the one a camera frame lands in.
            // The first predict also pays shader compilation / ANE planning,
            // so folding it in here keeps the first real image at
            // steady-state latency. Any larger bucket pays its own load the
            // first time something actually needs that resolution.
            if let model = try? model(for: first) {
                _ = try? predict(patches: [Float](repeating: 0,
                                                  count: first.n * meta.patchDim),
                                 gridH: 2, gridW: 2, bucketN: first.n,
                                 model: model)
            }
            self.loadSeconds = Date().timeIntervalSince(started)
        }
    }

    /// Load a bucket's model, once.
    private func model(for bucket: BucketRef) throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }
        if let existing = loadedModels[bucket.n] { return existing }
        let model = try MLModel(contentsOf: bucket.compiled, configuration: config)
        loadedModels[bucket.n] = model
        return model
    }

    /// The bucket a patch count lands in, identified without loading it.
    ///
    /// The content-addressed cache key includes the bucket's fingerprint, and
    /// it has to be computable *before* we know whether the model needs to
    /// run at all — otherwise every cache hit would still pay a model load.
    public func bucketInfo(patchCount n: Int) throws -> (n: Int, fingerprint: String) {
        let bucket = try pickBucket(patchCount: n)
        return (bucket.n, bucket.fingerprint)
    }

    /// Smallest bucket that fits `n` patches.
    private func pickBucket(patchCount n: Int) throws -> BucketRef {
        for bucket in buckets where n <= bucket.n { return bucket }
        throw VisionTowerError.gridTooLarge(patches: n, bucket: largestBucket)
    }

    /// Run the tower. Returns the merger output for the image's valid tokens,
    /// `[n/4, outHidden]` fp32.
    public func encode(patches: [Float], gridH: Int, gridW: Int)
        throws -> (embedding: [Float], fingerprint: String, bucketN: Int) {
        let n = gridH * gridW
        let bucket = try pickBucket(patchCount: n)
        lastBucketN = bucket.n

        var padded = [Float](repeating: 0, count: bucket.n * meta.patchDim)
        padded.replaceSubrange(0..<patches.count, with: patches)

        let model = try model(for: bucket)
        let started = Date()
        let full = try predict(patches: padded, gridH: gridH, gridW: gridW,
                               bucketN: bucket.n, model: model)
        lastPredictSeconds = Date().timeIntervalSince(started)

        // A deepstack tower's merger_out rows are wider than outHidden; the
        // caller (VisionTower.encode) splits main from deepstack, this layer
        // just must not truncate the row.
        let cols = meta.fusedOutHidden ?? meta.outHidden
        let valid = (n / 4) * cols
        return (Array(full[0..<valid]), bucket.fingerprint, bucket.n)
    }

    private func predict(patches: [Float], gridH: Int, gridW: Int,
                         bucketN: Int, model: MLModel) throws -> [Float] {
        let aux = try TowerAux.buildAux(
            table: posEmbed, hidden: meta.hidden,
            headDim: meta.headDim ?? 64,
            gridH: gridH, gridW: gridW, bucketN: bucketN,
            theta: meta.ropeTheta ?? 10000)

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray:
                try Self.array(patches, shape: [bucketN, meta.patchDim])),
            "pos_embed": MLFeatureValue(multiArray:
                try Self.array(aux.posEmbed, shape: [bucketN, meta.hidden])),
            "cos": MLFeatureValue(multiArray:
                try Self.array(aux.cos, shape: [bucketN, meta.headDim ?? 64])),
            "sin": MLFeatureValue(multiArray:
                try Self.array(aux.sin, shape: [bucketN, meta.headDim ?? 64])),
            "attn_bias": MLFeatureValue(multiArray:
                try Self.array(aux.attnBias, shape: [1, 1, 1, bucketN])),
        ])

        // Core ML predict is not documented as thread-safe per MLModel, and
        // the tiers are shared across requests here.
        lock.lock()
        defer { lock.unlock() }
        let out = try model.prediction(from: inputs)
        guard let merger = out.featureValue(for: "merger_out")?.multiArrayValue
        else {
            throw VisionTowerError.predictionFailed("merger_out missing")
        }
        return Self.floats(from: merger)
    }

    // MARK: - MLMultiArray bridging

    static func array(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init),
                                     dataType: .float32)
        let dst = array.dataPointer.bindMemory(to: Float.self,
                                               capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return array
    }

    static func floats(from array: MLMultiArray) -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float32:
            let src = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: src, count: count))
        case .float16:
            let src = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(src[$0]) }
        case .double:
            let src = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            return (0..<count).map { Float(src[$0]) }
        default:
            return (0..<count).map { array[$0].floatValue }
        }
    }
}

extension CoreMLTower {
    /// Compile the `.mlpackage` once, to a path that survives the process.
    ///
    /// `MLModel.compileModel(at:)` writes its `.mlmodelc` to a **temporary
    /// directory that differs on every call**, and Apple's docs say the
    /// caller must move it somewhere permanent to reuse it. Calling it at
    /// every load, as this did, meant the ANE saw a model at a new path each
    /// launch, missed its own plan cache, and recompiled from scratch:
    /// ANECompilerService burned ten minutes of CPU across a handful of app
    /// launches, and the first question of each one waited through it.
    ///
    /// Keeping the compiled model next to its source fixes that, and costs a
    /// directory whose contents can be deleted at any time -- it is rebuilt
    /// from the `.mlpackage` on the next launch.
    /// The already-compiled model sitting where the .mlpackage would be.
    ///
    /// Same base name, .mlmodelc extension — which is what `compileModel`
    /// produces and what a shipped bundle carries instead of the source.
    /// Compile every bucket in `directory` to its stable .mlmodelc and,
    /// when asked, delete the .mlpackage sources.
    ///
    /// The policy this implements: the package is compiler INPUT, and the
    /// compiler ships with macOS (MLModel.compileModel — no Xcode on user
    /// machines).  Once the .mlmodelc exists the package is a re-derivable
    /// intermediate costing ~865 MB per bucket; the export pipeline ends
    /// with this call so caches and shipped apps carry compiled models
    /// only.  fp16 weights measure 1.28x under zstd, so compression was
    /// never the lever — deletion is.
    public static func precompileAll(directory: URL,
                                     prunePackages: Bool) throws -> [URL] {
        let metaURL = directory.appendingPathComponent("tower_meta.json")
        let meta = try JSONDecoder().decode(
            TowerMeta.self, from: Data(contentsOf: metaURL))
        var compiled: [URL] = []
        for bucket in meta.sortedBuckets {
            let source = directory.appendingPathComponent(bucket.mlpackage)
            if FileManager.default.fileExists(atPath: source.path) {
                let out = try compiledModel(for: source, in: directory)
                compiled.append(out)
                if prunePackages {
                    try? FileManager.default.removeItem(at: source)
                }
            } else if let shipped = precompiled(besides: source) {
                compiled.append(shipped)   // already package-less
            } else {
                throw VisionTowerError.missingAsset(source.path)
            }
        }
        return compiled
    }

    static func precompiled(besides source: URL) -> URL? {
        let candidate = source.deletingPathExtension()
            .appendingPathExtension("mlmodelc")
        return FileManager.default.fileExists(atPath: candidate.path)
            ? candidate : nil
    }

    static func compiledModel(for source: URL, in directory: URL) throws -> URL {
        let fm = FileManager.default
        let cached = compileCacheDirectory(besides: directory)
            .appendingPathComponent(
                source.deletingPathExtension().lastPathComponent + ".mlmodelc")
        try? fm.createDirectory(at: cached.deletingLastPathComponent(),
                                withIntermediateDirectories: true)

        // Rebuild if the source is newer, so re-exporting a bucket is not
        // silently ignored.
        if fm.fileExists(atPath: cached.path) {
            let sourceDate = (try? source.resourceValues(
                forKeys: [.contentModificationDateKey]))?.contentModificationDate
            let cachedDate = (try? cached.resourceValues(
                forKeys: [.contentModificationDateKey]))?.contentModificationDate
            if let sourceDate, let cachedDate, sourceDate <= cachedDate {
                return cached
            }
            try? fm.removeItem(at: cached)
        }

        let temporary = try MLModel.compileModel(at: source)
        do {
            try fm.moveItem(at: temporary, to: cached)
            return cached
        } catch {
            // Read-only asset directory, a race with another process, a full
            // disk: none of those should stop the model from loading. Use the
            // temporary copy and pay the compile again next time.
            return temporary
        }
    }

    /// Where the compiled `.mlmodelc` may be written.
    ///
    /// Next to the source, when that is writable -- which is the case for an
    /// exported tower in a cache directory. Once the tower ships *inside the
    /// app bundle* it is not: a signed bundle is read-only, and writing into
    /// it would break the signature even if the filesystem allowed it. So the
    /// fallback is the app's own cache directory, and it has to exist,
    /// because losing it means recompiling on every launch -- on the ANE that
    /// is one to two minutes, every time.
    static func compileCacheDirectory(besides directory: URL) -> URL {
        if FileManager.default.isWritableFile(atPath: directory.path) {
            return directory
        }
        let caches = FileManager.default.urls(for: .cachesDirectory,
                                              in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        // Named after this SDK: the fallback cache lands in the HOST app's
        // caches directory, so it must not carry another product's name.
        // Renaming it orphans any previously-cached towers -- they are a
        // cache, so the cost is one recompile, not lost data.
        return caches.appendingPathComponent("Tempo9/tower",
                                             isDirectory: true)
    }
}
