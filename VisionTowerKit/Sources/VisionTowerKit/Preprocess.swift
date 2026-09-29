// Copyright (c) 2026 Jiejing Zhang.
// Portions derived from HuggingFace transformers
// (models/qwen2_vl/image_processing_qwen2_vl.py), Apache-2.0,
// Copyright The HuggingFace Inc. team. See NOTICE.
//
// Qwen2VL-style image preprocessing, ported from
// transformers/models/qwen2_vl/image_processing_qwen2_vl.py.
//
// Three steps: smart_resize to a multiple of patch*merge under a pixel cap,
// rescale + normalize, then patchify into the block-major [n, 1536] layout
// the tower's first Linear expects.
//
// KNOWN DEVIATION: when a resize is actually needed we use Core Image
// Lanczos, not PIL's bicubic-with-antialias. The two do not agree pixel for
// pixel. This costs nothing in the case that matters -- a caller that hands
// over an image already sized to a multiple of 32 under the cap gets no
// resampling at all on either side, which is the path the camera app and the
// parity gate both take. Feed it an arbitrary photo and the tower output will
// differ slightly from the Python chain; that is a real difference, not a
// rounding one, and it is why the parity gate uses pre-sized images.

import CoreGraphics
import CoreImage
import Foundation
import ImageIO

public struct PreprocessedImage {
    /// `[gridH*gridW, patchDim]` fp32, block-major.
    public var patches: [Float]
    public var gridH: Int
    public var gridW: Int
    public var patchDim: Int

    public var patchCount: Int { gridH * gridW }
}

public struct PreprocessConfig {
    public var patchSize: Int = 16
    public var temporalPatchSize: Int = 2
    public var mergeSize: Int = 2
    public var imageMean: [Float] = [0.5, 0.5, 0.5]
    public var imageStd: [Float] = [0.5, 0.5, 0.5]
    /// Pixel bounds, as the HF processor spells them.
    public var minPixels: Int = 65536
    public var maxPixels: Int = 16_777_216

    public init() {}

    /// Grid dims must be multiples of this for the merge step to tile.
    public var factor: Int { patchSize * mergeSize }
}

public enum Preprocess {

    /// Port of `smart_resize`: multiples of `factor`, pixels inside
    /// [minPixels, maxPixels], aspect ratio preserved as closely as possible.
    public static func smartResize(height: Int, width: Int,
                                   config: PreprocessConfig) throws -> (h: Int, w: Int) {
        let factor = config.factor
        let hi = Double(max(height, width)), lo = Double(min(height, width))
        guard lo > 0, hi / lo <= 200 else {
            throw VisionTowerError.extremeAspectRatio(height: height, width: width)
        }
        var hBar = Int((Double(height) / Double(factor)).rounded()) * factor
        var wBar = Int((Double(width) / Double(factor)).rounded()) * factor

        if hBar * wBar > config.maxPixels {
            let beta = (Double(height) * Double(width) / Double(config.maxPixels)).squareRoot()
            hBar = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            wBar = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if hBar * wBar < config.minPixels {
            let beta = (Double(config.minPixels) / (Double(height) * Double(width))).squareRoot()
            hBar = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    /// Decode an image file into a CGImage.
    public static func loadImage(url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VisionTowerError.imageDecodeFailed(url.path)
        }
        return image
    }

    /// Full preprocessing: CGImage -> patches.
    public static func run(image: CGImage, config: PreprocessConfig = PreprocessConfig())
        throws -> PreprocessedImage {
        let (targetH, targetW) = try smartResize(height: image.height,
                                                 width: image.width,
                                                 config: config)
        let rgb = try rgbBytes(from: image, width: targetW, height: targetH)
        return patchify(rgb: rgb, height: targetH, width: targetW, config: config)
    }

    /// Draw into a tightly-packed 8-bit sRGB RGBX buffer, resizing if needed.
    static func rgbBytes(from image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var source = image
        if image.width != width || image.height != height {
            source = try resized(image, width: width, height: height)
        }

        let bytesPerRow = width * 4
        var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw VisionTowerError.imageDecodeFailed("sRGB unavailable")
        }
        try buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
                throw VisionTowerError.imageDecodeFailed("CGContext")
            }
            ctx.interpolationQuality = .high
            ctx.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return buffer
    }

    static func resized(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
        let ci = CIImage(cgImage: image)
        let scaled = ci.transformed(by: CGAffineTransform(
            scaleX: CGFloat(width) / CGFloat(image.width),
            y: CGFloat(height) / CGFloat(image.height)))
        let context = CIContext(options: [.useSoftwareRenderer: false])
        guard let out = context.createCGImage(
            scaled, from: CGRect(x: 0, y: 0, width: width, height: height)) else {
            throw VisionTowerError.imageDecodeFailed("resize")
        }
        return out
    }

    /// Rescale + normalize + reshape into the tower's row layout.
    ///
    /// Row order is spatial-merge-block-major -- row index
    /// `((bh*blocksW + bw)*merge + mh)*merge + mw` -- and inside a row the
    /// elements run `[channel][temporal][py][px]`. A single image is repeated
    /// across the temporal axis, exactly as the HF processor does when it pads
    /// a 1-frame "video" to temporalPatchSize.
    static func patchify(rgb: [UInt8], height: Int, width: Int,
                         config: PreprocessConfig) -> PreprocessedImage {
        let patch = config.patchSize
        let merge = config.mergeSize
        let temporal = config.temporalPatchSize
        let channels = 3
        let gridH = height / patch
        let gridW = width / patch
        let blocksW = gridW / merge
        let patchDim = channels * temporal * patch * patch
        let n = gridH * gridW

        var mean = config.imageMean
        var std = config.imageStd
        if mean.count < channels { mean = [Float](repeating: mean.first ?? 0.5, count: channels) }
        if std.count < channels { std = [Float](repeating: std.first ?? 0.5, count: channels) }

        var out = [Float](repeating: 0, count: n * patchDim)
        let bytesPerRow = width * 4

        out.withUnsafeMutableBufferPointer { dst in
            rgb.withUnsafeBufferPointer { src in
                for row in 0..<n {
                    let mw = row % merge
                    let mh = (row / merge) % merge
                    let bw = (row / (merge * merge)) % blocksW
                    let bh = row / (merge * merge * blocksW)
                    let y0 = (bh * merge + mh) * patch
                    let x0 = (bw * merge + mw) * patch
                    let rowBase = row * patchDim

                    for c in 0..<channels {
                        for py in 0..<patch {
                            let lineBase = (y0 + py) * bytesPerRow + x0 * 4 + c
                            for px in 0..<patch {
                                let raw = Float(src[lineBase + px * 4])
                                let value = (raw / 255.0 - mean[c]) / std[c]
                                // Same value into every temporal slot: one
                                // still image occupies both halves.
                                for t in 0..<temporal {
                                    let idx = ((c * temporal + t) * patch + py) * patch + px
                                    dst[rowBase + idx] = value
                                }
                            }
                        }
                    }
                }
            }
        }

        return PreprocessedImage(patches: out, gridH: gridH, gridW: gridW,
                                 patchDim: patchDim)
    }
}
