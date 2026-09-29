// Copyright (c) 2026 Jiejing Zhang.
//
// The two GEMMs of the Gemma 4 tower, on the GPU.
//
// This is the whole reason an encoder-free tower is cheap: there is no graph
// to export and no bucket to compile, just C = A · Bᵀ twice. On CPU that is
// ~22 ms for a 280-token image (23 GFLOP at about 1 TFLOP/s); on Metal it is
// ~1 ms. The weights are uploaded once at init and stay resident, so a
// per-image call moves only the activations.
//
// MPSMatrixMultiplication rather than a hand-written kernel: the shapes are
// ordinary (280×6912×3840 and 280×3840×3840), MPS is already tuned for them,
// and a kernel of our own would be one more thing to keep correct for no
// measured gain.
//
// Falls back to Accelerate rather than failing: a tower that runs slowly is a
// product, a tower that refuses to run is not.

import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

public final class Gemma4MetalMatMul: Gemma4MatMul {
    public var label: String { "gpu" }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let fallback = AccelerateMatMul()
    /// Weights uploaded once, keyed by the array's identity. The tower holds
    /// exactly two of them and calls with the same arrays every frame, so a
    /// cache of two entries removes the only per-call copy that matters:
    /// mm.input_projection alone is 59 MB as f32.
    private var resident: [ObjectIdentifier: MTLBuffer] = [:]
    private var residentByHash: [Int: MTLBuffer] = [:]
    /// Activations and result, grown to the largest shape seen and then
    /// reused. Allocating them per call was most of the GPU path's time: the
    /// matmuls themselves are ~1 ms for a 280-token image, and makeBuffer
    /// plus the copies around them were another six.
    private var scratchA: MTLBuffer?
    private var scratchC: MTLBuffer?

    private func scratch(_ buf: inout MTLBuffer?, bytes: Int) -> MTLBuffer? {
        if let b = buf, b.length >= bytes { return b }
        buf = device.makeBuffer(length: bytes, options: .storageModeShared)
        return buf
    }
    private let lock = NSLock()
    private var warned = false

    /// A silent fallback is indistinguishable from a GPU that ran, and the
    /// only tell is that CPU and GPU agree to the last bit. Say it once.
    private func fellBack(_ why: String) {
        lock.lock(); defer { lock.unlock() }
        guard !warned else { return }
        warned = true
        FileHandle.standardError.write(Data(
            "Gemma4MetalMatMul: falling back to CPU — \(why)\n".utf8))
    }

    public init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { return nil }
        device = dev
        queue = q
    }

    /// Upload once, keyed on (count, first, last) — cheap, and the tower's two
    /// weight arrays differ in all three.
    private func weightBuffer(_ b: [Float], key: Int) -> MTLBuffer? {
        lock.lock()
        defer { lock.unlock() }
        if let cached = residentByHash[key] { return cached }
        let buf = b.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!,
                              length: raw.count,
                              options: .storageModeShared)
        }
        residentByHash[key] = buf
        return buf
    }

    public func mulTransposed(a: [Float], m: Int, k: Int,
                              b: [Float], n: Int) -> [Float] {
        var key = b.count &* 31
        key = key &+ (b.first.map { Int($0.bitPattern) } ?? 0)
        key = key &* 31 &+ (b.last.map { Int($0.bitPattern) } ?? 0)
        let aBytes = a.count * MemoryLayout<Float>.size
        let cBytes = m * n * MemoryLayout<Float>.size
        lock.lock()
        let aBuf = scratch(&scratchA, bytes: aBytes)
        let cBuf = scratch(&scratchC, bytes: cBytes)
        lock.unlock()
        if let aBuf {
            a.withUnsafeBytes { raw in
                memcpy(aBuf.contents(), raw.baseAddress!, aBytes)
            }
        }
        guard let bBuf = weightBuffer(b, key: key), let aBuf, let cBuf,
              let cmd = queue.makeCommandBuffer()
        else {
            fellBack("could not make buffers or a command buffer")
            return fallback.mulTransposed(a: a, m: m, k: k, b: b, n: n)
        }

        let rowF = MemoryLayout<Float>.size
        let da = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * rowF,
                                     dataType: .float32)
        // B is [n, k] row-major and the multiply wants Bᵀ — described as
        // [n, k] and transposed by the kernel, not repacked by us.
        let db = MPSMatrixDescriptor(rows: n, columns: k, rowBytes: k * rowF,
                                     dataType: .float32)
        let dc = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * rowF,
                                     dataType: .float32)
        let mm = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: m, resultColumns: n, interiorColumns: k,
            alpha: 1, beta: 0)
        mm.encode(commandBuffer: cmd,
                  leftMatrix: MPSMatrix(buffer: aBuf, descriptor: da),
                  rightMatrix: MPSMatrix(buffer: bBuf, descriptor: db),
                  resultMatrix: MPSMatrix(buffer: cBuf, descriptor: dc))
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            fellBack("\(err.localizedDescription)")
            return fallback.mulTransposed(a: a, m: m, k: k, b: b, n: n)
        }
        var out = [Float](repeating: 0, count: m * n)
        out.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, cBuf.contents(), m * n * rowF)
        }
        return out
    }
}
