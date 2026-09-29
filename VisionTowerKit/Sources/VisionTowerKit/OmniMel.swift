// Copyright (c) 2026 Jiejing Zhang.
//
// The mel front end Qwen2.5-Omni's audio encoder expects.
//
// Whisper's convention, inherited: 400-point Hann STFT, hop 160 (100 frames
// a second at 16 kHz), 128 SLANEY-scale filters over 0-8000 Hz, log10
// floored 8 dB below the whole utterance's maximum, then scaled to roughly
// [-1, 1].
//
// Two of those are the kind that fail silently, and both were paid for once
// in the Python port:
//
//   slaney, not HTK   The two mel curves share a name and nothing else --
//                     built with 2595*log10(1+f/700) the filterbank differs
//                     by 1.0 in places and whole rows come out zero. The
//                     tower then produces embeddings the model reads as
//                     speech and cannot decode: it answers "The audio says:
//                     '嗯嗯嗯嗯...'", audibly hearing something and
//                     transcribing nothing.
//   whole-utterance   The 8 dB floor is against the maximum of the ENTIRE
//   clamp             signal, not per frame. Per frame normalises silence up
//                     to the level of speech, which is audible in the answer
//                     and invisible in the numbers.

import Accelerate
import Foundation

public enum OmniMel {
    public static let sampleRate = 16000
    public static let nFFT = 400
    public static let hop = 160
    public static let nMel = 128
    /// One encoder chunk: 200 mel frames = 2 s.
    public static let chunkFrames = 200

    /// Slaney-scale, slaney-normalised filters, [nMel][nFFT/2 + 1].
    ///
    /// Built once and cached: it is 128 x 201 floats and depends on nothing
    /// but the constants above.
    static let filters: [[Float]] = makeFilters()

    private static func makeFilters() -> [[Float]] {
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep = log(6.4) / 27.0

        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + log(f / minLogHz) / logstep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * exp(logstep * (m - minLogMel)) : fSp * m
        }

        let nFreq = nFFT / 2 + 1
        let fftHz = (0..<nFreq).map {
            Double($0) * Double(sampleRate) / 2.0 / Double(nFreq - 1)
        }
        let melLo = hzToMel(0), melHi = hzToMel(Double(sampleRate) / 2)
        let hzPts = (0..<(nMel + 2)).map {
            melToHz(melLo + (melHi - melLo) * Double($0) / Double(nMel + 1))
        }
        var fb = [[Float]](repeating: [Float](repeating: 0, count: nFreq),
                           count: nMel)
        for i in 0..<nMel {
            let d0 = hzPts[i + 1] - hzPts[i]
            let d1 = hzPts[i + 2] - hzPts[i + 1]
            // Slaney normalisation: unit area per filter, so a wide
            // high-frequency band does not outweigh a narrow low one.
            let enorm = 2.0 / (hzPts[i + 2] - hzPts[i])
            for j in 0..<nFreq {
                let lower = (fftHz[j] - hzPts[i]) / d0
                let upper = (hzPts[i + 2] - fftHz[j]) / d1
                let v = max(0.0, min(lower, upper)) * enorm
                fb[i][j] = Float(v)
            }
        }
        return fb
    }

    /// The DFT as a matrix: [nFFT, 2 * nFreq], cos in even columns, sin in
    /// odd, so one GEMM turns every frame into every bin.
    ///
    /// A matrix rather than an FFT because **nFFT is 400, which is not a
    /// power of two**. vDSP's radix-2 FFT cannot do it -- asking for
    /// log2(400) rounds to 9 and the transform then reads 512 samples out
    /// of a 400-sample frame, which is a 17-exabyte allocation failure if
    /// you are lucky and silent garbage if you are not. Accelerate's DFT
    /// only supports lengths f * 2^n for f in {1, 3, 5, 15}, and 400 is
    /// 25 * 16. Zero-padding to 512 is not an option either: it changes the
    /// bin spacing, and the filterbank is built for 201 bins of 0-8000 Hz.
    ///
    /// The GEMM is exact and, at 400 x 402 per frame, fast: BLAS does 21
    /// minutes of audio in well under a second.
    static let dftMatrix: [Float] = {
        let nFreq = nFFT / 2 + 1
        var m = [Float](repeating: 0, count: nFFT * 2 * nFreq)
        for n in 0..<nFFT {
            for k in 0..<nFreq {
                let a = -2.0 * Double.pi * Double(k) * Double(n) / Double(nFFT)
                m[n * 2 * nFreq + 2 * k] = Float(cos(a))
                m[n * 2 * nFreq + 2 * k + 1] = Float(sin(a))
            }
        }
        return m
    }()

    /// 16 kHz mono float in [-1, 1] -> [nMel][frames], frame-major outer.
    public static func logMel(_ pcm: [Float]) -> (mel: [Float], frames: Int) {
        let nFreq = nFFT / 2 + 1
        let frames = 1 + pcm.count / hop
        // Reflect-pad by nFFT/2 each side: torch.stft(center: true), which
        // is what the reference does and what puts frame k at sample k*hop.
        let half = nFFT / 2
        var padded = [Float](repeating: 0, count: pcm.count + nFFT)
        for i in 0..<half {
            padded[i] = pcm[min(half - i, max(pcm.count - 1, 0))]
        }
        for i in 0..<pcm.count { padded[half + i] = pcm[i] }
        for i in 0..<half {
            let src = pcm.count - 2 - i
            padded[half + pcm.count + i] = pcm[max(0, min(src, pcm.count - 1))]
        }

        var window = [Float](repeating: 0, count: nFFT)
        // Periodic Hann (numpy's hanning(n+1)[:-1]), not symmetric: the
        // symmetric one is off by one sample and smears every bin slightly.
        for i in 0..<nFFT {
            window[i] = Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(i)
                                              / Double(nFFT)))
        }

        // Every frame, windowed, as rows of [frames, nFFT].
        var win = [Float](repeating: 0, count: frames * nFFT)
        for f in 0..<frames {
            let s0 = f * hop
            for i in 0..<nFFT {
                win[f * nFFT + i] =
                    (s0 + i < padded.count ? padded[s0 + i] : 0) * window[i]
            }
        }
        // One GEMM: [frames, nFFT] x [nFFT, 2*nFreq] -> [frames, 2*nFreq].
        var spec = [Float](repeating: 0, count: frames * 2 * nFreq)
        dftMatrix.withUnsafeBufferPointer { dp in
            win.withUnsafeBufferPointer { wp in
                spec.withUnsafeMutableBufferPointer { sp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(frames), Int32(2 * nFreq), Int32(nFFT),
                                1.0, wp.baseAddress!, Int32(nFFT),
                                dp.baseAddress!, Int32(2 * nFreq),
                                0.0, sp.baseAddress!, Int32(2 * nFreq))
                }
            }
        }
        var power = [Float](repeating: 0, count: frames * nFreq)
        for f in 0..<frames {
            for k in 0..<nFreq {
                let re = spec[f * 2 * nFreq + 2 * k]
                let im = spec[f * 2 * nFreq + 2 * k + 1]
                power[f * nFreq + k] = re * re + im * im
            }
        }

        // Filterbank, then log10 with the whole-utterance floor.
        var mel = [Float](repeating: 0, count: nMel * frames)
        for m in 0..<nMel {
            let row = filters[m]
            for f in 0..<frames {
                var acc: Float = 0
                let base = f * nFreq
                for k in 0..<nFreq where row[k] != 0 {
                    acc += row[k] * power[base + k]
                }
                mel[m * frames + f] = acc
            }
        }
        var maxLog: Float = -.greatestFiniteMagnitude
        for i in 0..<mel.count {
            let v = log10(max(mel[i], 1e-10))
            mel[i] = v
            if v > maxLog { maxLog = v }
        }
        let floorV = maxLog - 8.0
        for i in 0..<mel.count {
            mel[i] = (max(mel[i], floorV) + 4.0) / 4.0
        }
        return (mel, frames)
    }
}
