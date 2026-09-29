// Copyright (c) 2026 Jiejing Zhang.
// Portions derived from HuggingFace transformers (Apache-2.0, Copyright The
// HuggingFace Inc. team) via DashInfer/AllSpark (Apache-2.0, Copyright (c)
// Alibaba, Inc. and its affiliates). See NOTICE.
//
// Host-side aux computation for the Qwen3.5 vision tower.
//
// A direct port of python/pyhie/allspark/vlm/vit_aux.py, which is itself a
// numpy port of transformers 5.15 (vision_utils.get_vision_position_ids /
// get_vision_interpolation_indices_and_weights, Qwen3_5VisionRotaryEmbedding,
// Qwen3_5Model.get_rope_index). Everything data-dependent lives here so the
// Core ML graph stays static per bucket.
//
// Deliberately dependency-free: Foundation only, no Accelerate, no numpy
// equivalent. It is a few hundred multiply-adds per image against a tower
// forward pass measured in tens of milliseconds, and staying portable is
// worth more here than staying fast.

import Foundation

public enum TowerAux {

    /// Spatial merge size. 2 for every Qwen3.5 size.
    public static let merge = 2

    /// (row, col) of each patch in spatial-merge-block-major order.
    ///
    /// This ordering is not a convention we chose -- it is the order the
    /// image processor emits rows in, so the position tables must agree with
    /// it patch for patch or the tower silently sees scrambled geometry.
    public static func blockMajorRowCol(gridH: Int, gridW: Int)
        -> (row: [Int], col: [Int]) {
        let n = gridH * gridW
        let blocksW = gridW / merge
        var row = [Int](repeating: 0, count: n)
        var col = [Int](repeating: 0, count: n)
        for i in 0..<n {
            let inCol = i % merge
            let inRow = (i / merge) % merge
            let blockCol = (i / (merge * merge)) % blocksW
            let blockRow = i / (merge * merge * blocksW)
            row[i] = blockRow * merge + inRow
            col[i] = blockCol * merge + inCol
        }
        return (row, col)
    }

    /// Bilinear taps and weights along one axis, align_corners=True.
    static func axisTapsWeights(index: [Int], size: Int, side: Int)
        -> (taps: [[Int]], weights: [[Float]]) {
        let denom = Float(max(size - 1, 1))
        var taps = [[Int]](repeating: [0, 0], count: index.count)
        var weights = [[Float]](repeating: [0, 0], count: index.count)
        for (i, idx) in index.enumerated() {
            let src = Float(idx) * Float(side - 1) / denom
            let floorV = src.rounded(.down)
            for o in 0..<2 {
                let tap = Int(floorV) + o
                taps[i][o] = min(max(tap, 0), side - 1)
                let dist = abs(src - floorV - Float(o))
                weights[i][o] = max(1.0 - dist, 0.0)
            }
        }
        return (taps, weights)
    }

    /// Learned position embedding, bilinearly interpolated to this grid.
    /// Returns `[gridH*gridW, hidden]` in block-major order.
    public static func posEmbedInterpolated(table: [Float], hidden: Int,
                                            gridH: Int, gridW: Int) -> [Float] {
        let side = Int((Double(table.count / hidden)).squareRoot().rounded())
        let (row, col) = blockMajorRowCol(gridH: gridH, gridW: gridW)
        let (hTaps, hW) = axisTapsWeights(index: row, size: gridH, side: side)
        let (wTaps, wW) = axisTapsWeights(index: col, size: gridW, side: side)

        let n = row.count
        var out = [Float](repeating: 0, count: n * hidden)
        out.withUnsafeMutableBufferPointer { dst in
            table.withUnsafeBufferPointer { src in
                for i in 0..<n {
                    for a in 0..<2 {
                        for b in 0..<2 {
                            let weight = hW[i][a] * wW[i][b]
                            if weight == 0 { continue }
                            let base = (hTaps[i][a] * side + wTaps[i][b]) * hidden
                            let dstBase = i * hidden
                            for k in 0..<hidden {
                                dst[dstBase + k] += src[base + k] * weight
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// cos/sin `[n, headDim]` for the vision 2D rope (h and w axes).
    public static func ropeCosSin(gridH: Int, gridW: Int, headDim: Int,
                                  theta: Float = 10000) -> (cos: [Float], sin: [Float]) {
        let (row, col) = blockMajorRowCol(gridH: gridH, gridW: gridW)
        let n = row.count
        let dim = headDim / 2                    // rotary module dim
        var invFreq = [Float]()
        var k = 0
        while k < dim {
            invFreq.append(1.0 / powf(theta, Float(k) / Float(dim)))
            k += 2
        }
        let half = invFreq.count                 // dim/2 per axis
        var cosOut = [Float](repeating: 0, count: n * headDim)
        var sinOut = [Float](repeating: 0, count: n * headDim)
        for i in 0..<n {
            // f = [row*invFreq, col*invFreq]; emb = [f, f]
            for j in 0..<(2 * half) {
                let value: Float = j < half
                    ? Float(row[i]) * invFreq[j]
                    : Float(col[i]) * invFreq[j - half]
                let base = i * headDim
                cosOut[base + j] = cosf(value)
                sinOut[base + j] = sinf(value)
                cosOut[base + j + 2 * half] = cosf(value)
                sinOut[base + j + 2 * half] = sinf(value)
            }
        }
        return (cosOut, sinOut)
    }

    public struct Aux {
        public var posEmbed: [Float]     // [bucketN, hidden]
        public var cos: [Float]          // [bucketN, headDim]
        public var sin: [Float]          // [bucketN, headDim]
        public var attnBias: [Float]     // [1,1,1,bucketN]
    }

    /// All tower inputs except the patches, padded to `bucketN`.
    ///
    /// The pad values matter: cos pads with 1 and sin with 0 (an identity
    /// rotation) so padded rows cannot produce NaNs, and attnBias pads with a
    /// large negative so they cannot attract attention.
    public static func buildAux(table: [Float], hidden: Int, headDim: Int,
                                gridH: Int, gridW: Int, bucketN: Int,
                                theta: Float = 10000,
                                maskValue: Float = -1e4) throws -> Aux {
        let n = gridH * gridW
        guard n <= bucketN else {
            throw VisionTowerError.gridTooLarge(patches: n, bucket: bucketN)
        }
        let pe = posEmbedInterpolated(table: table, hidden: hidden,
                                      gridH: gridH, gridW: gridW)
        let (cos, sin) = ropeCosSin(gridH: gridH, gridW: gridW,
                                    headDim: headDim, theta: theta)

        var pePad = [Float](repeating: 0, count: bucketN * hidden)
        pePad.replaceSubrange(0..<pe.count, with: pe)
        var cosPad = [Float](repeating: 1, count: bucketN * headDim)
        cosPad.replaceSubrange(0..<cos.count, with: cos)
        var sinPad = [Float](repeating: 0, count: bucketN * headDim)
        sinPad.replaceSubrange(0..<sin.count, with: sin)
        var bias = [Float](repeating: maskValue, count: bucketN)
        for i in 0..<n { bias[i] = 0 }

        return Aux(posEmbed: pePad, cos: cosPad, sin: sinPad, attnBias: bias)
    }

    // MARK: - LLM side

    /// Expand each single placeholder into the run the model expects,
    /// wrapping it in the layout's open/close tokens when it has them.
    ///
    /// The wrap is part of the run, not decoration around it: Gemma 4 was
    /// trained with `<|image>` … `<image|>` around every picture, and the
    /// engine's block scan keys on the SOFT tokens between them, so the
    /// count that has to match the embedding is `nTokens` — not `nTokens+2`.
    public static func expandMediaPlaceholder(inputIDs: [Int],
                                              layout: MediaLayout,
                                              nTokens: [Int]) throws -> [Int] {
        try expandMediaPlaceholders(inputIDs: inputIDs, layout: layout,
                                    imageTokens: nTokens, audioTokens: [])
    }

    /// Both kinds in one pass, because a prompt can carry both and the two
    /// runs must keep their prompt order — the template emits `<|image|>`
    /// then `<|audio|>` where the message parts put them, and the engine
    /// finds each block by scanning for ITS token id.
    public static func expandMediaPlaceholders(inputIDs: [Int],
                                               layout: MediaLayout,
                                               imageTokens: [Int],
                                               audioTokens: [Int])
        throws -> [Int] {
        var out = [Int]()
        out.reserveCapacity(inputIDs.count
                            + imageTokens.reduce(0, +)
                            + audioTokens.reduce(0, +) + 4)
        var imageIndex = 0
        var audioIndex = 0
        for token in inputIDs {
            if token == layout.imageTokenID {
                guard imageIndex < imageTokens.count else {
                    throw VisionTowerError.placeholderMismatch(
                        placeholders: imageIndex + 1,
                        images: imageTokens.count)
                }
                if let open = layout.openTokenID { out.append(open) }
                out.append(contentsOf: [Int](repeating: layout.imageTokenID,
                                             count: imageTokens[imageIndex]))
                if let close = layout.closeTokenID { out.append(close) }
                imageIndex += 1
            } else if let audioToken = layout.audioTokenID,
                      token == audioToken {
                guard audioIndex < audioTokens.count else {
                    throw VisionTowerError.placeholderMismatch(
                        placeholders: audioIndex + 1,
                        images: audioTokens.count)
                }
                if let open = layout.audioOpenTokenID { out.append(open) }
                out.append(contentsOf: [Int](repeating: audioToken,
                                             count: audioTokens[audioIndex]))
                if let close = layout.audioCloseTokenID { out.append(close) }
                audioIndex += 1
            } else {
                out.append(token)
            }
        }
        guard imageIndex == imageTokens.count,
              audioIndex == audioTokens.count else {
            throw VisionTowerError.placeholderMismatch(
                placeholders: imageIndex + audioIndex,
                images: imageTokens.count + audioTokens.count)
        }
        return out
    }

    /// Expand each single image placeholder token into `nTokens` copies.
    public static func expandImagePlaceholder(inputIDs: [Int],
                                              imageTokenID: Int,
                                              nTokens: [Int]) throws -> [Int] {
        var out = [Int]()
        out.reserveCapacity(inputIDs.count + nTokens.reduce(0, +))
        var imageIndex = 0
        for token in inputIDs {
            if token == imageTokenID {
                guard imageIndex < nTokens.count else {
                    throw VisionTowerError.placeholderMismatch(
                        placeholders: imageIndex + 1, images: nTokens.count)
                }
                out.append(contentsOf:
                    [Int](repeating: token, count: nTokens[imageIndex]))
                imageIndex += 1
            } else {
                out.append(token)
            }
        }
        guard imageIndex == nTokens.count else {
            throw VisionTowerError.placeholderMismatch(
                placeholders: imageIndex, images: nTokens.count)
        }
        return out
    }

    /// Interleaved M-RoPE position table, `[3, inputIDs.count]`, row-major.
    ///
    /// Text runs advance all three axes together; an image run of grid
    /// (1, h, w) pins t to the current position and lays h/w out over the
    /// (h/2, w/2) LLM grid; afterwards the position advances by
    /// max(h, w) / merge.
    public static func llmMRoPEPositions(inputIDs: [Int], imageTokenID: Int,
                                         grids: [(t: Int, h: Int, w: Int)]) throws -> [Int32] {
        let count = inputIDs.count
        var pos = [Int32](repeating: 0, count: 3 * count)
        var gridIndex = 0
        var i = 0
        var current = 0

        while i < count {
            var j = i
            if inputIDs[i] == imageTokenID {
                while j < count && inputIDs[j] == imageTokenID { j += 1 }
                guard gridIndex < grids.count else {
                    throw VisionTowerError.placeholderMismatch(
                        placeholders: gridIndex + 1, images: grids.count)
                }
                let grid = grids[gridIndex]
                gridIndex += 1
                let lh = grid.h / merge
                let lw = grid.w / merge
                guard (j - i) == lh * lw else {
                    throw VisionTowerError.runLengthMismatch(
                        run: j - i, expected: lh * lw)
                }
                for k in 0..<(j - i) {
                    pos[i + k] = Int32(current)
                    pos[count + i + k] = Int32(current + k / lw)
                    pos[2 * count + i + k] = Int32(current + k % lw)
                }
                current += max(lh, lw)
            } else {
                while j < count && inputIDs[j] != imageTokenID { j += 1 }
                for k in 0..<(j - i) {
                    let v = Int32(current + k)
                    pos[i + k] = v
                    pos[count + i + k] = v
                    pos[2 * count + i + k] = v
                }
                current += j - i
            }
            i = j
        }
        return pos
    }
}
