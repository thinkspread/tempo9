// Copyright (c) 2026 Jiejing Zhang.
//
// vitctl -- run the Swift vision tower from the command line.
//
// Exists so the port can be checked against the Python/HF reference by a
// parity script in the maintainers' tree: it writes the merger output as raw
// fp32 and prints a JSON line of everything else.
//
//   vitctl --tower <dir> --image a.png [--tier gpu|ane|cpu] [--out emb.f32]
//          [--repeat 3]

import Foundation
import VisionTowerKit

/// mean/std/min/max/sum — the same five numbers the Python reference and
/// llama.cpp's MTMD_DEBUG_EMBEDDINGS print, so a parity check is a diff.
func statsOf(_ v: [Float]) -> [String: Double] {
    guard !v.isEmpty else { return [:] }
    var sum = 0.0, mn = Double(v[0]), mx = Double(v[0])
    for f in v { let d = Double(f); sum += d; mn = min(mn, d); mx = max(mx, d) }
    let mean = sum / Double(v.count)
    var sq = 0.0
    for f in v { let d = Double(f) - mean; sq += d * d }
    return ["mean": mean, "std": (sq / Double(v.count)).squareRoot(),
            "min": mn, "max": mx, "sum": sum]
}

/// Minimal 16-bit PCM WAV reader — the tower wants 16 kHz mono floats and
/// the parity clips are exactly that.
func readWavMono16k(_ path: String) throws -> [Float] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    guard d.count > 44 else { return [] }
    var off = 12
    var dataStart = 44, dataLen = d.count - 44
    while off + 8 <= d.count {
        let id = String(bytes: d[off..<off+4], encoding: .ascii) ?? ""
        let sz = d[(off+4)..<(off+8)].withUnsafeBytes {
            $0.loadUnaligned(as: UInt32.self)
        }
        if id == "data" { dataStart = off + 8; dataLen = Int(sz); break }
        off += 8 + Int(sz) + (Int(sz) % 2)
    }
    let n = min(dataLen, d.count - dataStart) / 2
    var out = [Float](repeating: 0, count: n)
    d.withUnsafeBytes { raw in
        let p = raw.baseAddress!.advanced(by: dataStart)
        for i in 0..<n {
            let s = p.advanced(by: i * 2).loadUnaligned(as: Int16.self)
            out[i] = Float(s) / 32768
        }
    }
    return out
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("vitctl: \(message)\n".utf8))
    exit(1)
}

var towerDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".cache/dashinfer/vlm/Qwen3.5-0.8B/coreml").path
var precompile = false
var prunePackages = false
var imagePath: String?
var outPath: String?
var patchesPath: String?
var tierName = "gpu"
var repeats = 1
/// Gemma 4 Unified has no tower directory to point at: its whole visual front
/// end is in the mmproj GGUF. --gemma4 <mmproj> runs that path instead.
var gemmaMMProj: String?
var audioPath: String?

var args = Array(CommandLine.arguments.dropFirst())
while let flag = args.first {
    args.removeFirst()
    func value() -> String {
        guard let v = args.first else { fail("\(flag) needs a value") }
        args.removeFirst()
        return v
    }
    switch flag {
    case "--tower": towerDir = value()
    case "--image": imagePath = value()
    case "--out": outPath = value()
    case "--dump-patches": patchesPath = value()
    case "--mel":
        // Dump the Swift mel for the cross-language parity check and exit.
        // Its own flag rather than a mode of the tower path: the point is
        // to test the front end ALONE, so nothing else may run.
        let w = value()
        MelProbe.run(wav: w, out: args.first ?? "/tmp/mel_swift.bin")
        exit(0)
    case "--omni-vision":
        // <dir> <image> <out.bin>: run the exported vision buckets.
        let d = value(); let im = value()
        MelProbe.vision(dir: d, image: im,
                        out: args.first ?? "/tmp/omnivis_swift.bin")
        exit(0)
    case "--omni-audio":
        // <dir> <wav> <out.bin>: run the exported Core ML audio tower and
        // dump its embeddings for the cross-language check.
        let dir = value(); let w = value()
        MelProbe.tower(dir: dir, wav: w, out: args.first ?? "/tmp/omni_swift.bin")
        exit(0)
    case "--gemma4": gemmaMMProj = value()
    case "--audio": audioPath = value()
    case "--tier": tierName = value()
    case "--precompile": precompile = true
    case "--prune-packages": prunePackages = true
    case "--repeat": repeats = Int(value()) ?? 1
    case "-h", "--help":
        print("vitctl --tower <dir> --image <file> [--tier gpu|ane|cpu] "
              + "[--out <file.f32>] [--dump-patches <file.f32>] [--repeat N]")
        exit(0)
    default: fail("unknown flag \(flag)")
    }
}


if precompile {
    do {
        let out = try CoreMLTower.precompileAll(
            directory: URL(fileURLWithPath: towerDir),
            prunePackages: prunePackages)
        for u in out { print("compiled: \(u.lastPathComponent)") }
        print(prunePackages ? "packages pruned" : "packages kept")
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("precompile failed: \(error)\n".utf8))
        exit(1)
    }
}

if let mmproj = gemmaMMProj {
    do {
        let t0 = Date()
        let tower = try Gemma4Tower(mmprojPath: mmproj)
        // --tier cpu forces the Accelerate reference the Metal path is
        // checked against; anything else takes the GPU when there is one.
        if tierName != "cpu", let gpu = Gemma4MetalMatMul() {
            tower.matmul = gpu
        }
        _ = tower
        let load = Date().timeIntervalSince(t0)
        var report: [String: Any] = [
            "hidden": tower.hidden, "has_audio": tower.hasAudio,
            "load_seconds": load, "matmul": tower.matmul.label,
        ]
        if let audioPath {
            let pcm = try readWavMono16k(audioPath)
            let t1 = Date()
            var emb: [Float] = []
            for _ in 0..<max(1, repeats) { emb = try tower.embedAudio(pcm) }
            report["audio_seconds"] = Date().timeIntervalSince(t1)
                / Double(max(1, repeats))
            report["audio_tokens"] = emb.count / tower.hidden
            report["audio_stats"] = statsOf(emb)
            if let outPath { try writeFloats(emb, to: outPath) }
        } else if let path = imagePath {
            let img = try Preprocess.loadImage(
                url: URL(fileURLWithPath: path))
            let t1 = Date()
            var out: (embedding: [Float], cols: Int, rows: Int)!
            for _ in 0..<max(1, repeats) {
                out = try tower.embedImage(img)
            }
            report["seconds"] = Date().timeIntervalSince(t1)
                / Double(max(1, repeats))
            report["grid"] = [out.cols, out.rows]
            report["tokens"] = out.cols * out.rows
            report["stats"] = statsOf(out.embedding)
            report["preprocess_seconds"] = tower.lastPreprocessSeconds
            report["matmul_seconds"] = tower.lastMatmulSeconds
            report["elementwise_seconds"] = tower.lastElementwiseSeconds
            if let outPath { try writeFloats(out.embedding, to: outPath) }
        } else {
            fail("--gemma4 needs --image or --audio")
        }
        let json = try JSONSerialization.data(withJSONObject: report,
                                              options: [.sortedKeys])
        print(String(data: json, encoding: .utf8)!)
        exit(0)
    } catch {
        fail(error.localizedDescription)
    }
}

guard let imagePath else { fail("--image is required") }
guard let tier = ComputeTier(rawValue: tierName) else {
    fail("--tier must be one of \(ComputeTier.allCases.map(\.rawValue))")
}

func writeFloats(_ values: [Float], to path: String) throws {
    var data = Data(capacity: values.count * 4)
    for value in values {
        withUnsafeBytes(of: value.bitPattern.littleEndian) {
            data.append(contentsOf: $0)
        }
    }
    try data.write(to: URL(fileURLWithPath: path))
}

do {
    let loadStart = Date()
    let tower = try VisionTower(directory: URL(fileURLWithPath: towerDir),
                                policy: .fixed(tier))
    let loadSeconds = Date().timeIntervalSince(loadStart)

    var encoding: Encoding!
    for _ in 0..<max(1, repeats) {
        encoding = try tower.encode(imageAt: URL(fileURLWithPath: imagePath))
    }

    if let outPath { try writeFloats(encoding.embedding, to: outPath) }
    if let patchesPath {
        // The parity gate needs the preprocessing output on its own: a
        // disagreement there and a disagreement inside the tower look the
        // same from the embedding alone.
        let pre = try Preprocess.run(
            image: try Preprocess.loadImage(url: URL(fileURLWithPath: imagePath)))
        try writeFloats(pre.patches, to: patchesPath)
    }

    let report: [String: Any] = [
        "grid": [encoding.gridH, encoding.gridW],
        "tokens": encoding.tokens,
        "out_hidden": encoding.outHidden,
        "bucket_n": encoding.bucketN,
        "tier": encoding.tier.rawValue,
        "cache_hit": encoding.cacheHit,
        "load_seconds": loadSeconds,
        "preprocess_seconds": encoding.preprocessSeconds,
        "tower_seconds": encoding.towerSeconds,
        "content_key": encoding.contentKey,
    ]
    let json = try JSONSerialization.data(withJSONObject: report,
                                          options: [.sortedKeys])
    print(String(data: json, encoding: .utf8)!)
} catch {
    fail(error.localizedDescription)
}
