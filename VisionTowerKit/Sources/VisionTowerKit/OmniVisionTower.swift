// Copyright (c) 2026 Jiejing Zhang.
//
// Qwen2.5-Omni's vision encoder, as exported Core ML buckets.
//
// The host does everything that depends on the image's shape, which is what
// lets the graph be fixed per bucket:
//
//   smart_resize      each side rounded to a multiple of 28
//   patch packing     temporal-major inside the patch vector
//   window order      tokens permuted into 112-pixel windows, each padded
//                     to its own 64-token stride
//   2D RoPE           (h, w) from the patch grid, half the head dim each
//
// Three of those are silent when wrong and were paid for in the port:
// packing channel-major instead of temporal-major reads a red rectangle as
// green while every pixel value matches to 3e-7; packing windows back to
// back splits the ragged edge ones across group boundaries; and sizing a
// bucket by patch count rather than padded slots overflows it.

import Accelerate
import CoreGraphics
import CoreML
import Foundation

public final class OmniVisionTower {
    public static let patch = 14
    public static let merge = 2
    public static let temporal = 2
    public static let channels = 3
    public static let headDim = 80
    public static let windowUnits = 4              // 112 / 2 / 14
    public static let windowTokens = windowUnits * windowUnits * merge * merge
    public static let patchDim = channels * temporal * patch * patch  // 1176
    static let factor = patch * merge               // 28
    static let fullLayers: Set<Int> = [7, 15, 23, 31]
    static let maskNeg: Float = -1e4

    static let imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]

    public let outHidden: Int
    private struct Bucket { let n: Int; let model: MLModel }
    private var buckets: [Bucket] = []
    private let lock = NSLock()

    public init(directory: URL, tier: ComputeTier = .ane) throws {
        let metaURL = directory.appendingPathComponent("tower_meta.json")
        guard let data = try? Data(contentsOf: metaURL),
              let meta = try? JSONSerialization.jsonObject(with: data)
                  as? [String: Any] else {
            throw VisionTowerError.missingAsset(metaURL.path)
        }
        outHidden = (meta["out_hidden"] as? Int) ?? 3584
        let cfg = MLModelConfiguration()
        // CPU for an fp32 graph, and that is the SLOWER answer in isolation.
        //
        // The Neural Engine is fp16-only, so asking it to run an fp32 graph
        // is not a preference it declines politely -- the work lands on the
        // CPU regardless. Alone, the GPU is 2.4x faster than that: 4169 ms
        // against 9918 ms on the 3072 bucket. In the app it is twice as
        // SLOW, 29 s against 16 s, because the LLM is on the GPU too and
        // they fight -- which is the same contention the Core ML tier
        // policy next door was written for.
        //
        // fp16 would put this on the ANE and is not available: the
        // activations do not overflow (15308 against fp16's 65504) but they
        // are large enough that fp16's spacing at magnitude 8000 is about 8,
        // and the tower comes out uncorrelated with the reference (cosine
        // mean 0.079). Keeping only the norm statistics in fp32 does not
        // rescue it either (0.101) -- the loss is spread through the
        // residual stream, not concentrated in the norms.
        let fp32 = (meta["precision"] as? String) != "fp16"
        cfg.computeUnits = fp32 ? .cpuOnly : tier.computeUnits
        for b in (meta["buckets"] as? [[String: Any]]) ?? [] {
            guard let n = b["n"] as? Int,
                  let name = b["mlpackage"] as? String else { continue }
            let pkg = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: pkg.path) else {
                throw VisionTowerError.missingAsset(pkg.path)
            }
            buckets.append(Bucket(
                n: n,
                model: try MLModel(contentsOf: MLModel.compileModel(at: pkg),
                                   configuration: cfg)))
        }
        guard !buckets.isEmpty else {
            throw VisionTowerError.missingAsset(
                directory.path + " has no exported buckets")
        }
        buckets.sort { $0.n < $1.n }
    }

    public var largestBucket: Int { buckets.last?.n ?? 0 }

    // MARK: - host side

    /// Qwen2.5-VL's rule: round each side to a multiple of 28, and rescale
    /// only when the rounded box leaves the pixel budget.
    static func smartResize(_ h: Int, _ w: Int, maxPixels: Int) -> (Int, Int) {
        let minPixels = 56 * 56
        var hb = Int((Double(h) / Double(factor)).rounded()) * factor
        var wb = Int((Double(w) / Double(factor)).rounded()) * factor
        if hb * wb > maxPixels {
            let beta = (Double(h * w) / Double(maxPixels)).squareRoot()
            hb = max(factor, Int(Double(h) / beta / Double(factor)) * factor)
            wb = max(factor, Int(Double(w) / beta / Double(factor)) * factor)
        } else if hb * wb < minPixels {
            let beta = (Double(minPixels) / Double(h * w)).squareRoot()
            hb = Int((Double(h) * beta / Double(factor)).rounded(.up)) * factor
            wb = Int((Double(w) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (max(factor, hb), max(factor, wb))
    }

    /// Merged-unit order window by window, and the way back.
    static func windowIndex(_ gridH: Int, _ gridW: Int)
        -> (order: [Int], reverse: [Int], lengths: [Int]) {
        let llmH = gridH / merge, llmW = gridW / merge
        let padH = (windowUnits - llmH % windowUnits) % windowUnits
        let padW = (windowUnits - llmW % windowUnits) % windowUnits
        let numH = (llmH + padH) / windowUnits
        let numW = (llmW + padW) / windowUnits
        var order: [Int] = []
        var lengths: [Int] = []
        order.reserveCapacity(llmH * llmW)
        for wh in 0..<numH {
            for ww in 0..<numW {
                var count = 0
                for i in 0..<windowUnits {
                    for j in 0..<windowUnits {
                        let r = wh * windowUnits + i
                        let c = ww * windowUnits + j
                        if r < llmH && c < llmW {
                            order.append(r * llmW + c)
                            count += 1
                        }
                    }
                }
                lengths.append(count * merge * merge)
            }
        }
        var reverse = [Int](repeating: 0, count: order.count)
        for (i, o) in order.enumerated() { reverse[o] = i }
        return (order, reverse, lengths)
    }

    /// CGImage -> ([patches, 1176], gridH, gridW), fitted to `bucket`.
    ///
    /// Fitted, not truncated: per-window padding inflates the slot count
    /// above the patch count, and an image that overflows must be made
    /// smaller rather than have its bottom cut off.
    static func fit(_ image: CGImage, bucket: Int)
        -> (patches: [Float], gridH: Int, gridW: Int)? {
        var budget = bucket * patch * patch
        for _ in 0..<8 {
            guard let (p, gh, gw) = preprocess(image, maxPixels: budget)
            else { return nil }
            let lens = windowIndex(gh, gw).lengths
            let slots = lens.count * windowTokens
            if slots <= bucket { return (p, gh, gw) }
            budget = Int(Double(budget) * Double(bucket) / Double(slots) * 0.95)
        }
        return nil
    }

    static func preprocess(_ image: CGImage, maxPixels: Int)
        -> (patches: [Float], gridH: Int, gridW: Int)? {
        let (hb, wb) = smartResize(image.height, image.width,
                                   maxPixels: maxPixels)
        // Resample on 8-bit and convert after, the order the reference uses.
        var rgba = [UInt8](repeating: 0, count: wb * hb * 4)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: &rgba, width: wb, height: hb,
                                  bitsPerComponent: 8, bytesPerRow: wb * 4,
                                  space: cs,
                                  bitmapInfo: CGImageAlphaInfo
                                      .premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: wb, height: hb))

        let gh = hb / patch, gw = wb / patch
        let unit = merge * merge
        var out = [Float](repeating: 0, count: gh * gw * patchDim)
        // Layout, and every index here matters: merge units are contiguous
        // and TEMPORAL is outermost inside a patch vector, because the
        // .gguf stores patch_embd as one [1280, 3, 14, 14] tensor per
        // temporal slice, concatenated. Channel-outermost gives identical
        // pixel VALUES and swaps the colours.
        let mh = gh / merge, mw = gw / merge
        for by in 0..<mh {
            for bx in 0..<mw {
                for iy in 0..<merge {
                    for ix in 0..<merge {
                        let patchIdx = ((by * mw + bx) * merge + iy) * merge + ix
                        let py = (by * merge + iy) * patch
                        let px = (bx * merge + ix) * patch
                        var o = patchIdx * patchDim
                        for t in 0..<temporal {
                            _ = t   // a still image is the same frame twice
                            for c in 0..<channels {
                                for y in 0..<patch {
                                    let row = (py + y) * wb * 4
                                    for x in 0..<patch {
                                        let v = Float(rgba[row + (px + x) * 4 + c])
                                            / 255.0
                                        out[o] = (v - imageMean[c]) / imageStd[c]
                                        o += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, gh, gw)
    }

    /// 2D RoPE for the merged-and-permuted order.
    static func ropeTables(_ gridH: Int, _ gridW: Int, order: [Int])
        -> (cos: [Float], sin: [Float]) {
        let unit = merge * merge
        let mw = gridW / merge
        let dim = headDim / 2
        var invFreq = [Double]()
        var i = 0
        while i < dim { invFreq.append(1.0 / pow(10000.0, Double(i) / Double(dim))); i += 2 }
        let half = invFreq.count
        let n = order.count * unit
        var cosT = [Float](repeating: 0, count: n * headDim)
        var sinT = [Float](repeating: 0, count: n * headDim)
        for (slot, u) in order.enumerated() {
            let by = u / mw, bx = u % mw
            for k in 0..<unit {
                let iy = k / merge, ix = k % merge
                let h = Double(by * merge + iy), w = Double(bx * merge + ix)
                let row = (slot * unit + k) * headDim
                for f in 0..<half {
                    let a = h * invFreq[f], b = w * invFreq[f]
                    // [h-freqs | w-freqs] then duplicated: the same
                    // concatenation the reference builds.
                    cosT[row + f] = Float(cos(a))
                    sinT[row + f] = Float(sin(a))
                    cosT[row + half + f] = Float(cos(b))
                    sinT[row + half + f] = Float(sin(b))
                    cosT[row + 2 * half + f] = Float(cos(a))
                    sinT[row + 2 * half + f] = Float(sin(a))
                    cosT[row + 3 * half + f] = Float(cos(b))
                    sinT[row + 3 * half + f] = Float(sin(b))
                }
            }
        }
        return (cosT, sinT)
    }

    // MARK: - encode

    /// CGImage -> [tokens, outHidden] in reading order.
    public func encode(image: CGImage) throws -> Encoding {
        let preStart = Date()
        guard let bucket = buckets.last,
              let (patches, gh, gw) = Self.fit(image, bucket: bucket.n) else {
            throw VisionTowerError.gridTooLarge(patches: 0,
                                                bucket: largestBucket)
        }
        // The LARGEST bucket, not the smallest that fits.
        //
        // `fit` has already shrunk the image to the largest bucket, so the
        // slot count is as close to it as the window padding allows; picking
        // the smallest bucket that holds THAT would be free, but the earlier
        // version sized the image to the small bucket too and threw away
        // resolution to save padding. Resolution is the whole game here:
        // the same screenshot at 228 tokens invents two of the three strings
        // it is asked to quote, and at 620 quotes all three exactly.
        let (order, reverse, lengths) = Self.windowIndex(gh, gw)
        let slots = lengths.count * Self.windowTokens
        guard let use = buckets.first(where: { $0.n >= slots }) else {
            throw VisionTowerError.gridTooLarge(patches: slots,
                                                bucket: largestBucket)
        }
        let N = use.n
        let unit = Self.merge * Self.merge
        let preSeconds = Date().timeIntervalSince(preStart)

        let (cosF, sinF) = Self.ropeTables(gh, gw, order: order)
        // Lay each window on its own stride. Packing them back to back
        // splits the ragged edge windows across group boundaries.
        let pIn = try MLMultiArray(shape: [NSNumber(value: N),
                                           NSNumber(value: Self.patchDim)],
                                   dataType: .float32)
        let cIn = try MLMultiArray(shape: [NSNumber(value: N),
                                           NSNumber(value: Self.headDim)],
                                   dataType: .float32)
        let sIn = try MLMultiArray(shape: [NSNumber(value: N),
                                           NSNumber(value: Self.headDim)],
                                   dataType: .float32)
        let pp = pIn.dataPointer.bindMemory(to: Float.self,
                                            capacity: N * Self.patchDim)
        let cp = cIn.dataPointer.bindMemory(to: Float.self,
                                            capacity: N * Self.headDim)
        let sp = sIn.dataPointer.bindMemory(to: Float.self,
                                            capacity: N * Self.headDim)
        for i in 0..<(N * Self.patchDim) { pp[i] = 0 }
        for i in 0..<(N * Self.headDim) { cp[i] = 0; sp[i] = 0 }

        var src = 0
        for (wi, len) in lengths.enumerated() {
            let base = wi * Self.windowTokens
            for t in 0..<len {
                let from = order[(src + t) / unit] * unit + (src + t) % unit
                for d in 0..<Self.patchDim {
                    pp[(base + t) * Self.patchDim + d] =
                        patches[from * Self.patchDim + d]
                }
                for d in 0..<Self.headDim {
                    cp[(base + t) * Self.headDim + d] =
                        cosF[(src + t) * Self.headDim + d]
                    sp[(base + t) * Self.headDim + d] =
                        sinF[(src + t) * Self.headDim + d]
                }
            }
            src += len
        }

        let g = N / Self.windowTokens
        let bw = try MLMultiArray(shape: [NSNumber(value: g), 1, 1,
                                          NSNumber(value: Self.windowTokens)],
                                  dataType: .float32)
        let bf = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: N)],
                                  dataType: .float32)
        let bwp = bw.dataPointer.bindMemory(to: Float.self,
                                            capacity: g * Self.windowTokens)
        let bfp = bf.dataPointer.bindMemory(to: Float.self, capacity: N)
        for i in 0..<(g * Self.windowTokens) { bwp[i] = Self.maskNeg }
        for i in 0..<N { bfp[i] = Self.maskNeg }
        for (wi, len) in lengths.enumerated() where wi < g {
            for t in 0..<min(len, Self.windowTokens) {
                bwp[wi * Self.windowTokens + t] = 0
                bfp[wi * Self.windowTokens + t] = 0
            }
        }

        let towerStart = Date()
        lock.lock()
        let result = try use.model.prediction(
            from: try MLDictionaryFeatureProvider(dictionary: [
                "patches": MLFeatureValue(multiArray: pIn),
                "cos": MLFeatureValue(multiArray: cIn),
                "sin": MLFeatureValue(multiArray: sIn),
                "bias_win": MLFeatureValue(multiArray: bw),
                "bias_full": MLFeatureValue(multiArray: bf)]))
        lock.unlock()
        guard let merged = result.featureValue(for: "merged")?.multiArrayValue
        else {
            throw VisionTowerError.predictionFailed("no merged output")
        }
        let towerSeconds = Date().timeIntervalSince(towerStart)
        let mp = merged.dataPointer.bindMemory(to: Float.self,
                                               capacity: merged.count)

        // Pull the real units out of the padded windows, then un-permute
        // back into reading order.
        var permuted = [Float]()
        permuted.reserveCapacity(order.count * outHidden)
        for (wi, len) in lengths.enumerated() {
            let base = wi * Self.windowTokens / unit
            for u in 0..<(len / unit) {
                let row = (base + u) * outHidden
                permuted.append(contentsOf: UnsafeBufferPointer(
                    start: mp + row, count: outHidden))
            }
        }
        var out = [Float](repeating: 0, count: order.count * outHidden)
        for (slot, dst) in reverse.enumerated() {
            _ = slot
            _ = dst
        }
        for (pos, u) in order.enumerated() {
            let from = pos * outHidden, to = u * outHidden
            for d in 0..<outHidden { out[to + d] = permuted[from + d] }
        }
        return Encoding(embedding: out, tokens: order.count,
                        outHidden: outHidden,
                        gridH: gh / Self.merge, gridW: gw / Self.merge,
                        contentKey: ContentKey.make(
                            patches: patches, gridH: gh, gridW: gw,
                            fingerprint: "omni-vit/b\(N)"),
                        tier: .ane, cacheHit: false,
                        preprocessSeconds: preSeconds,
                        towerSeconds: towerSeconds, bucketN: N)
    }
}
