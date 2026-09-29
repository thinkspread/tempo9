// Copyright (c) 2026 Jiejing Zhang.
// Portions ported from llama.cpp (MIT, Copyright (c) 2023-2026 The ggml
// authors): calc_size_preserved_ratio and resize_bilinear. See NOTICE.
//
// Gemma 4 Unified's "tower", which is not one.
//
// Qwen's vision tower is a real ViT: transformer blocks, a Core ML export
// per patch-count bucket, an ANE/GPU tier policy and 1.7 GB on disk.
// Gemma 4 12B Unified is ENCODER-FREE — its whole visual front end is
// eleven tensors in the mmproj:
//
//     LayerNorm -> patch_embd (3840x6912) -> LayerNorm
//               -> + position tables -> LayerNorm -> RMSNorm
//               -> mm.input_projection (3840x3840)
//
// No attention, no blocks, no Core ML, no buckets. Two GEMMs and some
// element-wise work, over 116 MB of weights that vision and audio share.
// Audio is smaller still: 640 raw samples per token, one RMSNorm and one
// projection, with no FFT and no filterbank.
//
// So this file exists instead of an export pipeline. The arithmetic is a
// port of the engine repository's Python reference embedder, which agrees
// with llama.cpp's own embeddings to 0.03 % (f16-vs-f32 noise) — that
// number is the acceptance gate, not a nice-to-have.
//
// PREPROCESSING IS EXACT OR IT IS WRONG. The reference is llama.cpp's
// calc_size_preserved_ratio + img_tool::resize(PAD_CEIL): round each side
// to the nearest 48, rescale only if that box leaves the pixel budget,
// scale to FIT inside preserving aspect (ceil), centre on black. "Scale up
// to the budget", "stretch" and "crop" were all tried upstream and all are
// wrong — a woman typing at a desk came back as "there is no person in this
// image". Three details of the resize differ from every stock bilinear and
// all three matter; they are marked below.

import Accelerate
import CoreGraphics
import Foundation
import GGUFKit

public enum Gemma4TowerError: Error, LocalizedError {
    case notUnified(String)
    case missingTensor(String)
    case unsupportedType(String, UInt32)
    case badImage

    public var errorDescription: String? {
        switch self {
        case .notUnified(let p):
            return "\(p) is not an encoder-free gemma4uv mmproj — the tower "
                + "variants (gemma4v/gemma4a) need a real ViT."
        case .missingTensor(let n): return "mmproj is missing \(n)"
        case .unsupportedType(let n, let t):
            return "\(n): ggml type \(t) unsupported (use the f16 mmproj)"
        case .badImage: return "could not read the image"
        }
    }
}

/// Where the two GEMMs run. Split out because correctness and speed are
/// separate questions: Accelerate is the reference the Metal path is checked
/// against, and it is also the fallback on a machine without a usable device.
public protocol Gemma4MatMul {
    /// C[m, n] = A[m, k] * Bᵀ[n, k]   (B is row-major [n, k], as ggml stores it)
    func mulTransposed(a: [Float], m: Int, k: Int,
                       b: [Float], n: Int) -> [Float]
    var label: String { get }
}

public struct AccelerateMatMul: Gemma4MatMul {
    public init() {}
    public var label: String { "cpu" }
    public func mulTransposed(a: [Float], m: Int, k: Int,
                              b: [Float], n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(m), Int32(n), Int32(k), 1,
                                ap.baseAddress, Int32(k),
                                bp.baseAddress, Int32(k), 0,
                                cp.baseAddress, Int32(n))
                }
            }
        }
        return c
    }
}

public final class Gemma4Tower {
    public static let patch = 48            // 16 * n_merge(3)
    public static let minTokens = 40        // llama.cpp: poor on small images
    public static let maxTokens = 280       // vision_soft_tokens_per_image
    public static let videoTokens = 70      // per video frame
    public static let audioSamplesPerToken = 640   // 40 ms @ 16 kHz

    /// Where the last embedImage spent its time. The GEMMs moved to the GPU
    /// and the total barely moved, which is worth being able to see rather
    /// than assume: for a 280-token image the matmuls are ~1 ms and the
    /// element-wise work around them is most of the rest.
    public private(set) var lastPreprocessSeconds: Double = 0
    public private(set) var lastMatmulSeconds: Double = 0
    public private(set) var lastElementwiseSeconds: Double = 0

    public let hidden: Int
    public let hasAudio: Bool
    public var matmul: Gemma4MatMul

    private let epsLN: Float = 1e-5         // torch LayerNorm default
    private let epsRMS: Float
    private var w: [String: [Float]] = [:]
    /// [2][1120][hidden] — table 0 indexed by COLUMN, table 1 by ROW.
    private var pos: [[Float]] = []
    private let posStride: Int

    public init(mmprojPath: String,
                matmul: Gemma4MatMul = AccelerateMatMul()) throws {
        self.matmul = matmul
        let file = try GGUFFile(path: mmprojPath, readArrays: false)
        let proj = file.kv["clip.vision.projector_type"]?.stringValue
        guard proj == "gemma4uv" else {
            throw Gemma4TowerError.notUnified(mmprojPath)
        }
        hidden = file.optionalInt("clip.vision.projection_dim") ?? 3840
        epsRMS = Float(file.kv["clip.vision.attention.layer_norm_epsilon"]?
            .doubleValue ?? 1e-6)

        let blob = try Data(contentsOf: URL(fileURLWithPath: mmprojPath),
                            options: .mappedIfSafe)
        for (name, info) in file.tensors {
            w[name] = try Self.floats(blob, file.dataOffset + info.offset,
                                      info)
        }
        hasAudio = w["mm.a.input_projection.weight"] != nil
        guard let table = w["v.position_embd.weight"] else {
            throw Gemma4TowerError.missingTensor("v.position_embd.weight")
        }
        // ne is innermost-first: [hidden, 1120, 2].
        posStride = hidden
        let per = table.count / 2
        pos = [Array(table[0..<per]), Array(table[per...])]
        for n in ["v.patch_embd.weight", "v.patch_embd.bias",
                  "mm.input_projection.weight",
                  "v.patch_norm.1.weight", "v.patch_norm.1.bias",
                  "v.patch_norm.2.weight", "v.patch_norm.2.bias",
                  "v.patch_norm.3.weight", "v.patch_norm.3.bias"] {
            guard w[n] != nil else { throw Gemma4TowerError.missingTensor(n) }
        }
    }

    private static func floats(_ blob: Data, _ at: UInt64,
                               _ info: GGUFTensorInfo) throws -> [Float] {
        var count = 1
        for d in info.ne { count *= Int(d) }
        let start = Int(at)
        let width = info.typeID == 1 ? 2 : 4
        // A read past the mapping is a SIGBUS, which says nothing about which
        // tensor or by how much. Check it and say.
        guard start >= 0, start + count * width <= blob.count else {
            throw Gemma4TowerError.unsupportedType(
                "\(info.name): needs [\(start), \(start + count * width)) "
                + "of a \(blob.count)-byte file", info.typeID)
        }
        switch info.typeID {
        case 0:                                     // F32
            var out = [Float](repeating: 0, count: count)
            blob.withUnsafeBytes { raw in
                let base = raw.baseAddress!.advanced(by: start)
                out.withUnsafeMutableBufferPointer { op in
                    for i in 0..<count {
                        op[i] = base.advanced(by: i * 4)
                            .loadUnaligned(as: Float.self)
                    }
                }
            }
            return out
        case 1:                                     // F16
            // Element-wise rather than vImage: a single "row" of 2.4 M
            // pixels crashed vImageConvert_Planar16FtoPlanarF with SIGBUS,
            // and loadUnaligned makes no alignment assumption about a
            // memory-mapped offset the way assumingMemoryBound does.
            var out = [Float](repeating: 0, count: count)
            blob.withUnsafeBytes { raw in
                let base = raw.baseAddress!.advanced(by: start)
                out.withUnsafeMutableBufferPointer { op in
                    for i in 0..<count {
                        let h = base.advanced(by: i * 2)
                            .loadUnaligned(as: Float16.self)
                        op[i] = Float(h)
                    }
                }
            }
            return out
        default:
            throw Gemma4TowerError.unsupportedType(info.name, info.typeID)
        }
    }

    // MARK: - geometry

    /// llama.cpp img_tool::calc_size_preserved_ratio, ported exactly.
    /// Align to PATCH by ROUNDING — not floor, and not "scale up to the
    /// budget" — and rescale only when the rounded box leaves the budget.
    public func targetSize(width: Int, height: Int,
                           maxTokens: Int = Gemma4Tower.maxTokens,
                           minTokens: Int = Gemma4Tower.minTokens)
        -> (w: Int, h: Int) {
        let a = Gemma4Tower.patch
        let lo = minTokens * a * a, hi = maxTokens * a * a
        var wb = max(a, Int((Double(width) / Double(a)).rounded()) * a)
        var hb = max(a, Int((Double(height) / Double(a)).rounded()) * a)
        if hb * wb > hi {
            let beta = (Double(height) * Double(width) / Double(hi)).squareRoot()
            wb = max(a, Int((Double(width) / beta / Double(a)).rounded(.down)) * a)
            hb = max(a, Int((Double(height) / beta / Double(a)).rounded(.down)) * a)
        } else if hb * wb < lo {
            let beta = (Double(lo) / (Double(height) * Double(width))).squareRoot()
            wb = Int((Double(width) * beta / Double(a)).rounded(.up)) * a
            hb = Int((Double(height) * beta / Double(a)).rounded(.up)) * a
        }
        return (wb, hb)
    }

    // MARK: - image

    /// -> (embedding [tokens * hidden], cols, rows)
    public func embedImage(_ image: CGImage,
                           maxTokens: Int = Gemma4Tower.maxTokens)
        throws -> (embedding: [Float], cols: Int, rows: Int) {
        let (patches, cols, rows) = try imagePatches(image, maxTokens: maxTokens)
        return (try embedPatches(patches, cols: cols, rows: rows), cols, rows)
    }

    /// The two halves separately, because a caller that caches needs the
    /// patch matrix BEFORE the tower runs: the cache key is taken over the
    /// patches, and a key that only exists afterwards cannot save the work
    /// it is supposed to save.
    public func imagePatches(_ image: CGImage,
                             maxTokens: Int = Gemma4Tower.maxTokens)
        throws -> (patches: [Float], cols: Int, rows: Int) {
        let tPre = Date()
        let out = try preprocess(image, maxTokens: maxTokens)
        lastPreprocessSeconds = Date().timeIntervalSince(tPre)
        return out
    }

    public func embedPatches(_ patches: [Float], cols: Int, rows: Int)
        throws -> [Float] {
        let tokens = cols * rows
        let dim = Gemma4Tower.patch * Gemma4Tower.patch * 3
        var mmSeconds = 0.0
        let tRest = Date()

        var x = patches
        layerNorm(&x, rows: tokens, dim: dim,
                  gamma: w["v.patch_norm.1.weight"]!,
                  beta: w["v.patch_norm.1.bias"]!, eps: epsLN)
        var t0 = Date()
        x = matmul.mulTransposed(a: x, m: tokens, k: dim,
                                 b: w["v.patch_embd.weight"]!, n: hidden)
        mmSeconds += Date().timeIntervalSince(t0)
        addBias(&x, rows: tokens, dim: hidden, bias: w["v.patch_embd.bias"]!)
        layerNorm(&x, rows: tokens, dim: hidden,
                  gamma: w["v.patch_norm.2.weight"]!,
                  beta: w["v.patch_norm.2.bias"]!, eps: epsLN)
        // Table 0 is indexed by COLUMN, table 1 by ROW. Swapping them
        // measurably degrades text reading — established by A/B upstream,
        // so do not "simplify" this.
        for t in 0..<tokens {
            let c = t % cols, r = t / cols
            for d in 0..<hidden {
                x[t * hidden + d] += pos[0][c * posStride + d]
                    + pos[1][r * posStride + d]
            }
        }
        layerNorm(&x, rows: tokens, dim: hidden,
                  gamma: w["v.patch_norm.3.weight"]!,
                  beta: w["v.patch_norm.3.bias"]!, eps: epsLN)
        rmsNorm(&x, rows: tokens, dim: hidden, eps: epsRMS)
        t0 = Date()
        x = matmul.mulTransposed(a: x, m: tokens, k: hidden,
                                 b: w["mm.input_projection.weight"]!,
                                 n: hidden)
        mmSeconds += Date().timeIntervalSince(t0)
        lastMatmulSeconds = mmSeconds
        lastElementwiseSeconds =
            Date().timeIntervalSince(tRest) - mmSeconds
        return x
    }

    /// RGB -> [tokens, 6912] patch matrix.
    private func preprocess(_ image: CGImage, maxTokens: Int)
        throws -> ([Float], Int, Int) {
        let sw = image.width, sh = image.height
        let (tw, th) = targetSize(width: sw, height: sh, maxTokens: maxTokens)
        guard let src = Self.rgbBytes(image) else {
            throw Gemma4TowerError.badImage
        }
        let scale = min(Double(tw) / Double(sw), Double(th) / Double(sh))
        let nw = min(Int((Double(sw) * scale).rounded(.up)), tw)
        let nh = min(Int((Double(sh) * scale).rounded(.up)), th)
        let resized = (nw == sw && nh == sh)
            ? src
            : Self.resizeBilinear(src, sw: sw, sh: sh, tw: nw, th: nh)

        // Centre on a BLACK canvas — pad colour is 0,0,0.
        var canvas = [UInt8](repeating: 0, count: tw * th * 3)
        let oy = (th - nh) / 2, ox = (tw - nw) / 2
        for y in 0..<nh {
            let s = y * nw * 3, d = ((y + oy) * tw + ox) * 3
            for i in 0..<(nw * 3) { canvas[d + i] = resized[s + i] }
        }

        let p = Gemma4Tower.patch
        let cols = tw / p, rows = th / p
        var out = [Float](repeating: 0, count: rows * cols * p * p * 3)
        // Within a patch the element order must be CHW (R..R G..G B..B), NOT
        // the HWC the pixels arrive in: llama.cpp permutes patch_dense's
        // columns to match ggml im2col's CHW output. Feeding HWC produces
        // plausible-but-wrong captions — colours survive, structure does not.
        for r in 0..<rows {
            for c in 0..<cols {
                let base = (r * cols + c) * p * p * 3
                for ch in 0..<3 {
                    let chBase = base + ch * p * p
                    for y in 0..<p {
                        let sy = r * p + y
                        for x in 0..<p {
                            let sx = c * p + x
                            out[chBase + y * p + x] =
                                Float(canvas[(sy * tw + sx) * 3 + ch]) / 255
                        }
                    }
                }
            }
        }
        return (out, cols, rows)
    }

    /// llama.cpp mtmd-image.cpp resize_bilinear, ported exactly.
    ///
    /// Three things differ from every stock bilinear and all three matter:
    /// the ratio is ALIGN-CORNERS ((src-1)/(dst-1), not src/dst with
    /// half-pixel centres), there is NO antialiasing on downscale, and the
    /// store TRUNCATES to uint8 instead of rounding.
    private static func resizeBilinear(_ src: [UInt8], sw: Int, sh: Int,
                                       tw: Int, th: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: tw * th * 3)
        let xr = tw > 1 ? Double(sw - 1) / Double(tw - 1) : 0
        let yr = th > 1 ? Double(sh - 1) / Double(th - 1) : 0
        for y in 0..<th {
            let py = Double(y) * yr
            let y0 = min(Int(py), sh - 1), y1 = min(y0 + 1, sh - 1)
            let yf = Float(py - Double(y0))
            for x in 0..<tw {
                let px = Double(x) * xr
                let x0 = min(Int(px), sw - 1), x1 = min(x0 + 1, sw - 1)
                let xf = Float(px - Double(x0))
                for c in 0..<3 {
                    let p00 = Float(src[(y0 * sw + x0) * 3 + c])
                    let p10 = Float(src[(y0 * sw + x1) * 3 + c])
                    let p01 = Float(src[(y1 * sw + x0) * 3 + c])
                    let p11 = Float(src[(y1 * sw + x1) * 3 + c])
                    let top = p00 + (p10 - p00) * xf
                    let bot = p01 + (p11 - p01) * xf
                    out[(y * tw + x) * 3 + c] = UInt8(top + (bot - top) * yf)
                }
            }
        }
        return out
    }

    private static func rgbBytes(_ image: CGImage) -> [UInt8]? {
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = rgba.withUnsafeMutableBytes({ raw -> CGContext? in
            CGContext(data: raw.baseAddress, width: w, height: h,
                      bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = rgba[i * 4]
            out[i * 3 + 1] = rgba[i * 4 + 1]
            out[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return out
    }

    // MARK: - audio

    /// 16 kHz mono float in [-1, 1] -> [tokens * hidden].
    /// One token per 640 samples (40 ms); a trailing partial frame is
    /// zero-padded. No FFT, no filterbank: the frame IS the feature.
    public func embedAudio(_ samples: [Float]) throws -> [Float] {
        guard let proj = w["mm.a.input_projection.weight"] else {
            throw Gemma4TowerError.missingTensor("mm.a.input_projection.weight")
        }
        let per = Gemma4Tower.audioSamplesPerToken
        let n = (samples.count + per - 1) / per
        guard n > 0 else { return [] }
        var x = [Float](repeating: 0, count: n * per)
        for i in 0..<min(samples.count, n * per) { x[i] = samples[i] }
        rmsNorm(&x, rows: n, dim: per, eps: epsRMS)
        return matmul.mulTransposed(a: x, m: n, k: per, b: proj, n: hidden)
    }

    // MARK: - element-wise

    /// `&array[i]` does NOT promise a pointer into the array's contiguous
    /// storage: Swift may hand a one-element temporary, and a vDSP call that
    /// reads `dim` elements from it walks off the end — SIGBUS, with a stack
    /// that points at Accelerate rather than at this line. Everything here
    /// goes through withUnsafeMutableBufferPointer for that reason.
    private func layerNorm(_ x: inout [Float], rows: Int, dim: Int,
                           gamma: [Float], beta: [Float], eps: Float) {
        let n = vDSP_Length(dim)
        x.withUnsafeMutableBufferPointer { xp in
            gamma.withUnsafeBufferPointer { gp in
                beta.withUnsafeBufferPointer { bp in
                    guard let x0 = xp.baseAddress, let g = gp.baseAddress,
                          let b = bp.baseAddress else { return }
                    for r in 0..<rows {
                        let row = x0 + r * dim
                        var mean: Float = 0
                        vDSP_meanv(row, 1, &mean, n)
                        var neg = -mean
                        vDSP_vsadd(row, 1, &neg, row, 1, n)
                        var sq: Float = 0
                        vDSP_svesq(row, 1, &sq, n)
                        var inv = 1 / (sq / Float(dim) + eps).squareRoot()
                        vDSP_vsmul(row, 1, &inv, row, 1, n)
                        vDSP_vmul(row, 1, g, 1, row, 1, n)
                        vDSP_vadd(row, 1, b, 1, row, 1, n)
                    }
                }
            }
        }
    }

    /// Scale-less RMS norm — there is no gamma on this one.
    private func rmsNorm(_ x: inout [Float], rows: Int, dim: Int, eps: Float) {
        let n = vDSP_Length(dim)
        x.withUnsafeMutableBufferPointer { xp in
            guard let x0 = xp.baseAddress else { return }
            for r in 0..<rows {
                let row = x0 + r * dim
                var sq: Float = 0
                vDSP_svesq(row, 1, &sq, n)
                var inv = 1 / (sq / Float(dim) + eps).squareRoot()
                vDSP_vsmul(row, 1, &inv, row, 1, n)
            }
        }
    }

    private func addBias(_ x: inout [Float], rows: Int, dim: Int,
                         bias: [Float]) {
        let n = vDSP_Length(dim)
        x.withUnsafeMutableBufferPointer { xp in
            bias.withUnsafeBufferPointer { bp in
                guard let x0 = xp.baseAddress, let b = bp.baseAddress else {
                    return
                }
                for r in 0..<rows {
                    let row = x0 + r * dim
                    vDSP_vadd(row, 1, b, 1, row, 1, n)
                }
            }
        }
    }
}
