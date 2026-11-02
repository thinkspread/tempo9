// Copyright (c) 2026 Jiejing Zhang.
//
// What `--tower` names, and whether it fits the model.
//
// A GGUF is the language model only.  Seeing needs a second half, and which
// half depends on the family: Gemma 4 12B's projector is an mmproj .gguf
// that usually sits in the same repo, while Qwen's vision encoder runs as an
// exported Core ML directory that no GGUF repo carries.  So `--tower` takes
// either, on disk or on Hugging Face, plus `auto`, which finds the one that
// belongs to the model being served.
//
// The pairing is checked before the server listens.  A wrong tower whose
// width differs is refused by the engine on every image request -- after a
// clean startup, which reads as a working server.  One whose width matches
// is worse: Qwen3.5-35B-A3B and Qwen3-VL-30B are both 2048 wide, and the
// wrong encoder answers fluently about nothing.

import CoreGraphics
import Foundation
import GGUFKit
import Tempo9
import VisionTowerKit

struct TowerError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

/// The served model, as its GGUF header names it.
struct ModelIdentity {
    let architecture: String
    let hidden: Int?
    /// Lowercased lookup keys, most specific first: basename + size label
    /// + finetune, then shorter.  "Qwen3.5-9B" -> ["qwen3.5-9b"];
    /// qwen3vl / 4B / instruct -> ["qwen3vl-4b-instruct", "qwen3vl-4b", "qwen3vl"].
    let keys: [String]
    let displayName: String

    init(ggufPath: String) {
        let f = try? GGUFFile(path: ggufPath, readArrays: false)
        architecture = f?.architecture ?? ""
        hidden = f?.arch("embedding_length")?.intValue
        let base = f?.kv["general.basename"]?.stringValue ?? ""
        let size = f?.kv["general.size_label"]?.stringValue ?? ""
        let tune = f?.kv["general.finetune"]?.stringValue ?? ""
        displayName = f?.kv["general.name"]?.stringValue
            ?? (ggufPath as NSString).lastPathComponent
        var k: [String] = []
        func add(_ name: String) {
            var part: [String] = []
            var stem = name.lowercased()
            guard !stem.isEmpty else { return }
            part.append(stem)
            if !size.isEmpty, !stem.hasSuffix("-" + size.lowercased()) {
                stem += "-" + size.lowercased()
                part.append(stem)
            }
            if !tune.isEmpty { part.append(stem + "-" + tune.lowercased()) }
            k += part.reversed()
        }
        add(base)
        // Converters that start from the Hugging Face id write its org into
        // the basename: lmstudio-community's Qwen3.8-27B says "Qwen_Qwen3.8",
        // unsloth's says "Qwen3.8-27B".  Same model, so the name without the
        // org is tried after the name as written.
        if let us = base.firstIndex(of: "_"), base.index(after: us) < base.endIndex {
            add(String(base[base.index(after: us)...]))
        }
        keys = k
    }
}

enum Towers {
    /// Core ML encoders published for models a GGUF cannot see with.
    /// Keyed by ModelIdentity.keys.  Each entry was run end to end before
    /// it was added; a model that shares an architecture is NOT covered by
    /// its neighbour -- every Qwen3.5 size, and Qwen3.6, has its own vision
    /// weights.
    static let published: [String: String] = [
        "qwen3.5-9b": "thinkspread/Qwen3.5-9B-Vision-CoreML-fp16",
        "qwen3.5-35b-a3b": "thinkspread/Qwen3.5-35B-A3B-Vision-CoreML-fp16",
        "qwen3.6-35b-a3b": "thinkspread/Qwen3.6-35B-A3B-Vision-CoreML-fp16",
        "qwen3.8-27b": "thinkspread/Qwen3.8-27B-Vision-CoreML-fp16",
        "qwen3vl-4b-instruct": "thinkspread/Qwen3-VL-4B-Instruct-Vision-CoreML-fp16",
        "qwen3vl-30b-a3b-instruct":
            "thinkspread/Qwen3-VL-30B-A3B-Instruct-Vision-CoreML-fp16",
    ]

    static func publishedTower(for model: ModelIdentity) -> String? {
        model.keys.lazy.compactMap { published[$0] }.first
    }

    /// Which tower family each GGUF architecture takes, by MediaLayout.name.
    /// An architecture missing here is not checked.
    static let family: [String: String] = [
        "qwen35": "qwen3.5", "qwen35moe": "qwen3.5",
        "qwen3vl": "qwen3vl", "qwen3vlmoe": "qwen3vl",
        "gemma4": "gemma4",
    ]

    // MARK: - Resolving

    /// A local directory or .gguf, `owner/repo[:tag]` on Hugging Face, or
    /// `auto`.  Returns what VisionTower should open.
    static func resolve(_ spec: String, model: ModelIdentity, ggufPath: String,
                        hfRepo: String?, hfListing: HuggingFaceStore.Listing?)
        throws -> URL {
        if spec == "auto" {
            if let u = try gemmaProjector(model: model, ggufPath: ggufPath,
                                          hfRepo: hfRepo, hfListing: hfListing) {
                return u
            }
            guard let repo = publishedTower(for: model) else {
                throw TowerError("--tower auto: no published vision encoder for "
                    + "\(model.displayName) (\(model.architecture)); pass a "
                    + "directory, an mmproj .gguf, or a Hugging Face repo")
            }
            return try fetch(repo)
        }
        let local = (spec as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: local) {
            return URL(fileURLWithPath: local)
        }
        guard HuggingFaceStore.parse(spec) != nil else {
            throw TowerError("\(spec): no such file or directory")
        }
        return try fetch(spec)
    }

    /// Gemma 4's projector, if the model is Gemma 4 and an encoder-free
    /// (gemma4uv) mmproj sits beside the GGUF or in the `--hf` repo.  It is
    /// part of the model's own release, 122 MB for 12B, so it is used
    /// without being asked for -- as llama.cpp's -hf does.  The other Gemma
    /// 4 projectors carry a full vision encoder this path does not run;
    /// they are recognised from their header and left alone.
    static func gemmaProjector(model: ModelIdentity, ggufPath: String,
                               hfRepo: String?,
                               hfListing: HuggingFaceStore.Listing?) throws -> URL? {
        guard model.architecture == "gemma4" else { return nil }
        let dir = (ggufPath as NSString).deletingLastPathComponent
        let beside = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
            .filter { $0.lowercased().hasPrefix("mmproj") && $0.hasSuffix(".gguf") }
        for name in preferred(beside) {
            let p = (dir as NSString).appendingPathComponent(name)
            if projectorType(ofGGUFAt: p) == "gemma4uv" { return URL(fileURLWithPath: p) }
        }
        guard let hfRepo, let hfListing else { return nil }
        let mmproj = hfListing.files.filter {
            ($0.path as NSString).lastPathComponent.lowercased().hasPrefix("mmproj")
                && $0.path.hasSuffix(".gguf")
        }
        guard let file = preferred(mmproj.map(\.path)).lazy
            .compactMap({ p in mmproj.first { $0.path == p } }).first else { return nil }
        let root = HuggingFaceStore.hubRoot()
        if HuggingFaceStore.cached(file, repo: hfRepo, root: root) == nil {
            let head = try HuggingFaceStore.head(file, repo: hfRepo,
                                                 revision: hfListing.revision,
                                                 bytes: 4 << 20)
            let probe = FileManager.default.temporaryDirectory
                .appendingPathComponent("tempo9-mmproj-\(getpid()).gguf")
            try head.write(to: probe)
            defer { try? FileManager.default.removeItem(at: probe) }
            let type = projectorType(ofGGUFAt: probe.path)
            guard type == "gemma4uv" else {
                say("hf: \(file.path) holds a \(type ?? "unrecognised") projector,"
                    + " which this server does not run")
                return nil
            }
        }
        return try fetchFile(file, repo: hfRepo, revision: hfListing.revision)
    }

    /// F16 first: the projector path reads F16 and F32, not BF16.
    static func preferred(_ names: [String]) -> [String] {
        func rank(_ n: String) -> Int {
            let l = n.lowercased()
            if l.contains("bf16") { return 2 }
            if l.contains("f16") { return 0 }
            if l.contains("f32") { return 1 }
            return 3
        }
        return names.sorted { (rank($0), $0) < (rank($1), $1) }
    }

    static func projectorType(ofGGUFAt path: String) -> String? {
        (try? GGUFFile(path: path, readArrays: false))?
            .kv["clip.vision.projector_type"]?.stringValue
    }

    // MARK: - Fetching

    static func say(_ m: String) {
        FileHandle.standardError.write(Data("[tempo9] \(m)\n".utf8))
    }

    /// A Hugging Face repo holding a tower: a Core ML export (has
    /// tower_meta.json, fetched whole) or a GGUF repo (its mmproj only).
    static func fetch(_ spec: String) throws -> URL {
        guard let ref = HuggingFaceStore.parse(spec) else {
            throw TowerError("--tower: \(spec) is not owner/repo")
        }
        let root = HuggingFaceStore.hubRoot()
        let offline = ProcessInfo.processInfo.environment["HF_HUB_OFFLINE"] == "1"
        let listing: HuggingFaceStore.Listing
        do {
            if offline { throw TowerError("HF_HUB_OFFLINE=1") }
            listing = try HuggingFaceStore.listing(ref.repo)
        } catch {
            if let snap = HuggingFaceStore.cachedSnapshot(
                ref.repo, containing: "tower_meta.json", root: root) {
                say("hf: \(error); using the cached \(ref.repo)")
                return snap
            }
            throw TowerError(offline
                ? "hf: HF_HUB_OFFLINE=1 and \(ref.repo) is not in \(root.path)"
                : "hf: \(error)")
        }
        if listing.files.contains(where: { $0.path == "tower_meta.json" }) {
            // Documentation is not part of the tower.
            let files = listing.files.filter {
                !$0.path.hasPrefix(".") && !$0.path.lowercased().hasSuffix(".md")
            }
            let total = files.reduce(into: [String: Int64]()) { $0[$1.sha256] = $1.size }
                .values.reduce(0, +)
            let have = files.allSatisfy {
                HuggingFaceStore.cached($0, repo: ref.repo, root: root) != nil
            }
            say(String(format: "hf: %@ (%.1f GB)%@", ref.repo, Double(total) / 1e9,
                       have ? ", cached" : " -> \(root.path)"))
            return try withProgress(skip: have) { progress in
                try HuggingFaceStore.fetchSnapshot(files, repo: ref.repo,
                                                   revision: listing.revision,
                                                   root: root, progress: progress)
            }
        }
        let mmproj = listing.files.filter {
            ($0.path as NSString).lastPathComponent.lowercased().hasPrefix("mmproj")
                && $0.path.hasSuffix(".gguf")
        }
        let pick: HuggingFaceStore.RemoteFile?
        if let q = ref.quant?.lowercased() {
            pick = mmproj.first { ($0.path as NSString).lastPathComponent.lowercased()
                .contains(q) }
        } else {
            pick = preferred(mmproj.map(\.path)).first.flatMap { p in
                mmproj.first { $0.path == p } }
        }
        guard let pick else {
            throw TowerError("hf: \(ref.repo) has no tower_meta.json and no"
                + " mmproj .gguf\(ref.quant.map { " matching \($0)" } ?? "")")
        }
        return try fetchFile(pick, repo: ref.repo, revision: listing.revision)
    }

    static func fetchFile(_ file: HuggingFaceStore.RemoteFile, repo: String,
                          revision: String) throws -> URL {
        let root = HuggingFaceStore.hubRoot()
        let have = HuggingFaceStore.cached(file, repo: repo, root: root) != nil
        say(String(format: "hf: %@/%@ (%.1f GB)%@", repo, file.path,
                   Double(file.size) / 1e9, have ? ", cached" : " -> \(root.path)"))
        return try withProgress(skip: have) { progress in
            try HuggingFaceStore.fetch(file, repo: repo, revision: revision,
                                       root: root, progress: progress)
        }
    }

    /// The same one-line progress the model download prints.
    static func withProgress(
        skip: Bool,
        _ body: (@escaping @Sendable (Int64, Int64) -> Void) throws -> URL) throws -> URL {
        let tty = isatty(2) != 0
        let start = Date()
        nonisolated(unsafe) var lastPct = -1
        nonisolated(unsafe) var firstDone: Int64 = -1
        do {
            let url = try body { done, total in
                let pct = Int(Double(done) * 100 / Double(max(total, 1)))
                if firstDone < 0 { firstDone = done }
                let secs = max(Date().timeIntervalSince(start), 0.001)
                let line = String(format: "[tempo9] hf: %3d%%  %.2f / %.2f GB  %.0f MB/s",
                                  pct, Double(done) / 1e9, Double(total) / 1e9,
                                  Double(done - firstDone) / 1e6 / secs)
                if tty {
                    FileHandle.standardError.write(Data(("\r" + line).utf8))
                } else if pct / 10 != lastPct / 10 {
                    FileHandle.standardError.write(Data((line + "\n").utf8))
                }
                lastPct = pct
            }
            if tty && !skip { FileHandle.standardError.write(Data("\n".utf8)) }
            return url
        } catch {
            if tty && !skip { FileHandle.standardError.write(Data("\n".utf8)) }
            throw TowerError("hf: \(error)")
        }
    }

    // MARK: - Loading and pairing

    static func load(_ url: URL) throws -> VisionTower {
        url.pathExtension == "gguf"
            ? try VisionTower(gemma4Mmproj: url)
            : try VisionTower(directory: url)
    }

    /// Encode a blank image and compare what comes out with what the model
    /// takes in.  Also loads the encoder now rather than on the first
    /// request, which is where the user would otherwise wait for it.
    static func checkPairing(_ tower: VisionTower, model: ModelIdentity) throws {
        if let want = family[model.architecture], want != tower.layout.name {
            throw TowerError("this is a \(tower.layout.name) vision encoder and "
                + "\(model.displayName) is \(model.architecture); "
                + suggestion(for: model))
        }
        guard let hidden = model.hidden, hidden > 0 else { return }
        let side = 256
        guard let ctx = CGContext(data: nil, width: side, height: side,
                                  bitsPerComponent: 8, bytesPerRow: side * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return }
        ctx.setFillColor(gray: 0.5, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let image = ctx.makeImage() else { return }
        let e = try tower.encode(image: image)
        if e.outHidden != hidden {
            throw TowerError("the vision encoder emits \(e.outHidden)-wide embeddings"
                + " and \(model.displayName) takes \(hidden); "
                + suggestion(for: model))
        }
    }

    static func suggestion(for model: ModelIdentity) -> String {
        if let repo = publishedTower(for: model) {
            return "its encoder is --tower \(repo) (or --tower auto)"
        }
        return "use the encoder exported from this model"
    }
}
