// Copyright (c) 2026 Jiejing Zhang.
//
// Qwen2.5-Omni's audio encoder, as an exported Core ML graph.
//
// One fixed shape and no buckets, which is what makes this simpler than the
// vision tower next door: the encoder's attention is block-diagonal over
// two-second chunks, so a chunk never attends outside itself and the graph
// is [1, 128, 200] -> [1, 50, 3584] for every input there will ever be.
// Longer audio is more calls, not a bigger graph.
//
// Measured on this machine: 10.5 ms a chunk on CPU_AND_NE, so 21 minutes of
// audio encodes in about 6.6 s — and on the Neural Engine, so it does not
// contend with a Metal LLM.

import CoreML
import Foundation

// AudioEncoding is declared in VisionTower.swift and shared: both towers
// answer the same question in the same units, and a second identical struct
// would only invite them to drift apart.

public final class OmniAudioTower {
    public static let tokensPerChunk = OmniMel.chunkFrames / 4   // 50
    public let outHidden: Int

    private let model: MLModel
    private let lock = NSLock()

    /// `directory` holds omni_audio_chunk.mlpackage and tower_meta.json.
    public init(directory: URL, tier: ComputeTier = .ane) throws {
        let pkg = directory.appendingPathComponent("omni_audio_chunk.mlpackage")
        guard FileManager.default.fileExists(atPath: pkg.path) else {
            throw VisionTowerError.missingAsset(pkg.path)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = tier.computeUnits
        // The compiled form is cached beside the package by Core ML itself;
        // the first load pays for it and later ones do not.
        model = try MLModel(contentsOf: MLModel.compileModel(at: pkg),
                            configuration: cfg)
        let meta = directory.appendingPathComponent("tower_meta.json")
        let j = (try? Data(contentsOf: meta)).flatMap {
            try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
        } ?? [:]
        outHidden = (j["out_hidden"] as? Int) ?? 3584
    }

    /// 16 kHz mono float in [-1, 1] -> one embedding per 40 ms.
    ///
    /// Chunks are run separately because that is what the model does, not as
    /// an approximation: its attention cannot cross a chunk boundary.
    public func encode(_ pcm: [Float]) throws -> AudioEncoding {
        let start = Date()
        let (mel, frames) = OmniMel.logMel(pcm)
        guard frames >= 2 else {
            return AudioEncoding(embedding: [], tokens: 0,
                                 outHidden: outHidden, seconds: 0)
        }
        let nMel = OmniMel.nMel
        let chunk = OmniMel.chunkFrames
        var out = [Float]()
        out.reserveCapacity((frames / 4 + 1) * outHidden)

        var s = 0
        while s < frames {
            let take = min(chunk, frames - s)
            if take < 2 { break }
            let input = try MLMultiArray(shape: [1, NSNumber(value: nMel),
                                                 NSNumber(value: chunk)],
                                         dataType: .float32)
            let dst = input.dataPointer.bindMemory(
                to: Float.self, capacity: nMel * chunk)
            // mel is [nMel][frames]; the graph wants [1, nMel, chunk] with
            // the tail zero-padded. A short final chunk is padded, and only
            // the tokens it really covers are kept below.
            for m in 0..<nMel {
                let srcBase = m * frames + s
                let dstBase = m * chunk
                for t in 0..<take { dst[dstBase + t] = mel[srcBase + t] }
                if take < chunk {
                    for t in take..<chunk { dst[dstBase + t] = 0 }
                }
            }
            lock.lock()
            let result = try model.prediction(
                from: try MLDictionaryFeatureProvider(
                    dictionary: ["mel": MLFeatureValue(multiArray: input)]))
            lock.unlock()
            guard let emb = result.featureValue(for: "embeddings")?
                .multiArrayValue else {
                throw VisionTowerError.predictionFailed("no embeddings output")
            }
            // conv stride 2 then a stride-2 pool: four mel frames a token,
            // rounded up so a partial final token is not dropped.
            let keep = min(Self.tokensPerChunk, (take + 3) / 4)
            let p = emb.dataPointer.bindMemory(
                to: Float.self, capacity: emb.count)
            out.append(contentsOf: UnsafeBufferPointer(
                start: p, count: keep * outHidden))
            s += chunk
        }
        return AudioEncoding(embedding: out, tokens: out.count / outHidden,
                             outHidden: outHidden,
                             seconds: Date().timeIntervalSince(start))
    }
}
