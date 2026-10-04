// Copyright (c) 2026 Jiejing Zhang.
//
// The OpenAI-compatible server, with no app around it.
//
// Written for a benchmark that had drifted onto the wrong code path: the
// only way to reach this server was to launch the UI app, and when the app
// would not start for a given model the measurement quietly moved to the
// Python server -- a different implementation, measured and reported as if
// it were this one. A headless entry point removes that temptation, and is
// what CI should drive too.

import Foundation
import Tempo9
import Tempo9Engine
import VisionTowerKit
import CoreGraphics

nonisolated(unsafe) var keptAlive: OpenAIServer?

func arg(_ name: String, _ dflt: String? = nil) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: "--" + name), i + 1 < a.count { return a[i + 1] }
    return dflt
}

// Models Ollama already pulled, read in place.  The cost of trying another
// engine is the re-download of the weights, not of the engine; Ollama's
// blobs are bare GGUF, so that cost can simply be zero.  Read-only -- see
// OllamaStore.
if CommandLine.arguments.contains("--list-ollama") {
    let found = OllamaStore.list()
    if found.isEmpty {
        let msg = "[tempo9] no Ollama models under "
            + "\(OllamaStore.root.path)"
            + " (set OLLAMA_MODELS if the store moved)\n"
        FileHandle.standardError.write(Data(msg.utf8))
        exit(1)
    }
    for e in found {
        let gib = Double(e.sizeBytes) / 1_073_741_824
        print(String(format: "%-40s %6.2f GiB  %@", (e.name as NSString).utf8String!,
                     gib, e.ggufPath))
    }
    exit(0)
}

var ollamaEntry: OllamaStore.Entry?
var ollamaGguf: String?
if let ref = arg("ollama") {
    guard let e = OllamaStore.resolve(ref) else {
        // Say what IS there.  "not found" against a store the user knows
        // holds the model usually means a tag mismatch, and printing the
        // names answers that in one line instead of a support thread.
        let have = OllamaStore.list().map(\.name)
        var msg = "[tempo9] no Ollama model '\(ref)' under "
            + "\(OllamaStore.root.path)\n"
        if !have.isEmpty {
            msg += "  available: " + have.joined(separator: ", ") + "\n"
        }
        FileHandle.standardError.write(Data(msg.utf8))
        exit(2)
    }
    ollamaEntry = e
    // The blob is extensionless and the engine dispatches on the extension,
    // so serve it through a .gguf-named symlink in our own cache.
    ollamaGguf = try OllamaStore.linkedPath(for: e)
    FileHandle.standardError.write(Data(
        "[tempo9] ollama: \(e.name) -> \(e.ggufPath)\n".utf8))
    if e.projectorPath != nil {
        // Parsed but not loadable here: our vision path takes a Core ML
        // tower directory, not a GGUF mmproj.  Saying so beats letting a
        // user wonder why images do nothing.
        let note = "[tempo9] note: this model ships a GGUF vision"
            + " projector, which this path does not load; text only."
            + " Use --tower <dir>.\n"
        FileHandle.standardError.write(Data(note.utf8))
    }
}

// A GGUF from Hugging Face, fetched into Hugging Face's own cache (see
// HuggingFaceStore): `--hf owner/repo[:quant]`.  One command from an empty
// machine to a running server is the point; a file someone already pulled
// with huggingface_hub or llama.cpp is reused, not downloaded again.
var hfGguf: String?
var hfSize: Int64?
func fileSize(_ u: URL) -> Int64? {
    ((try? FileManager.default.attributesOfItem(
        atPath: u.resolvingSymlinksInPath().path))?[.size] as? NSNumber)?.int64Value
}
if let text = arg("hf") {
    func fail(_ m: String, _ code: Int32 = 1) -> Never {
        FileHandle.standardError.write(Data("[tempo9] \(m)\n".utf8))
        exit(code)
    }
    guard let ref = HuggingFaceStore.parse(text) else {
        fail("--hf takes owner/repo[:quant], e.g."
             + " unsloth/Qwen3.5-9B-GGUF:Q4_K_S", 2)
    }
    let root = HuggingFaceStore.hubRoot()
    let offline = ProcessInfo.processInfo.environment["HF_HUB_OFFLINE"] == "1"
    let listing: HuggingFaceStore.Listing?
    do {
        listing = offline ? nil : try HuggingFaceStore.listing(ref.repo)
    } catch {
        // Offline is a normal state for a laptop.  A copy already in the
        // cache is the right answer then; only say why it is not fresh.
        guard let c = HuggingFaceStore.cachedOffline(ref, root: root) else {
            fail("hf: \(error)")
        }
        FileHandle.standardError.write(Data(
            "[tempo9] hf: \(error); using the cached \(c.lastPathComponent)\n".utf8))
        listing = nil
        hfGguf = c.path
        hfSize = fileSize(c)
    }
    if offline {
        guard let c = HuggingFaceStore.cachedOffline(ref, root: root) else {
            fail("hf: HF_HUB_OFFLINE=1 and \(ref.repo) is not in \(root.path)")
        }
        FileHandle.standardError.write(Data(
            "[tempo9] hf: offline, cached \(c.lastPathComponent)\n".utf8))
        hfGguf = c.path
        hfSize = fileSize(c)
    }
    if let listing {
        let file: HuggingFaceStore.RemoteFile
        switch HuggingFaceStore.choose(listing.files, quant: ref.quant) {
        case .success(let f): file = f
        case .failure(.noGGUF): fail("hf: \(ref.repo) has no GGUF weights")
        case .failure(.noMatch(let q, let have)):
            fail("hf: \(ref.repo) has no \(q); it has "
                 + have.joined(separator: ", "), 2)
        case .failure(.ambiguous(let have)):
            fail("hf: \(ref.repo) has several GGUFs; pick one with"
                 + " \(ref.repo):<quant> -- " + have.joined(separator: ", "), 2)
        case .failure(.split(let p)):
            fail("hf: \(p) is split into parts, which this engine does not"
                 + " load; pick a single-file quant")
        }
        let gb = Double(file.size) / 1e9
        let have = HuggingFaceStore.cached(file, repo: ref.repo, root: root) != nil
        FileHandle.standardError.write(Data(String(format:
            "[tempo9] hf: %@/%@ (%.1f GB)%@\n", ref.repo, file.path, gb,
            have ? ", cached" : " -> \(root.path)").utf8))
        let tty = isatty(2) != 0
        let start = Date()
        nonisolated(unsafe) var lastPct = -1
        // Rate over THIS run's bytes: a resumed download starts with the
        // earlier run's bytes already counted, which read as 900 MB/s.
        nonisolated(unsafe) var firstDone: Int64 = -1
        do {
            let path = try HuggingFaceStore.fetch(
                file, repo: ref.repo, revision: listing.revision, root: root
            ) { done, total in
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
            if tty && !have { FileHandle.standardError.write(Data("\n".utf8)) }
            hfGguf = path.path
            hfSize = file.size
        } catch {
            if tty { FileHandle.standardError.write(Data("\n".utf8)) }
            fail("hf: \(error)")
        }
    }
}

// --graph is optional for a .gguf: the engine builds the graph itself.
guard let gguf = hfGguf ?? ollamaGguf ?? arg("gguf")
        ?? (CommandLine.arguments.count == 2
            && CommandLine.arguments[1].hasSuffix(".gguf")
            ? CommandLine.arguments[1] : nil) else {
    let usage = "usage: tempo9 (--gguf <.gguf> | --hf <owner/repo[:quant]>"
        + " | --ollama <model[:tag]>)"
        + " [--graph <.asgraph>] [--name <id>]\n"
        + "       [--port 11435] [--max-length 32768] [--max-batch 16]"
        + " [--speculation-k 0] [--tower <dir>]\n"
        + "       tempo9 --list-ollama        models Ollama already pulled\n"
    FileHandle.standardError.write(Data(usage.utf8))
    exit(2)
}
let graph = arg("graph") ?? ""
let port = UInt16(arg("port", "11435")!) ?? 11435
let maxLen = Int64(arg("max-length", "32768")!) ?? 32768
// Continuous batching is the engine's job; 1 would make the server serial
// end to end (the N=4 staircase).  16 matches the concurrency the P4c
// table demonstrated on this hardware.
let maxBatch = Int32(arg("max-batch", "16")!) ?? 16
// MTP speculation depth for every request this server serves (0 = off).
// The engine's controller varies k downward from here at runtime.
let specK = Int32(arg("speculation-k", "0")!) ?? 0
let name = arg("name", ollamaEntry?.name
                       ?? (gguf as NSString).lastPathComponent)!

// A refused model is an ANSWER, not a fault: print it and exit.  A bare
// top-level `try` turns the engine's refusal into Swift's "Fatal error:
// Error raised at top level" -- a SIGTRAP with a crash report, which reads
// as a bug in the CLI rather than a fact about the file.
let session: LocalSession
do {
    session = try LocalSession(modelName: "headless", graphPath: graph,
                               ggufPath: gguf, maxLength: maxLen,
                               maxBatch: maxBatch)
} catch {
    FileHandle.standardError.write(Data(
        "[tempo9] cannot load \(gguf): \(error)\n".utf8))
    exit(1)
}

// Optional vision. The tower is what makes the image path OURS rather than
// the engine's: encoding happens here, and its content-addressed cache is
// the thing an agent looping over one screenshot actually exercises.
var tower: VisionTower?
if let towerDir = arg("tower") {
    tower = try VisionTower(directory: URL(fileURLWithPath: towerDir))
    session.mediaLayout = tower?.layout
    FileHandle.standardError.write(Data("[tempo9] tower: \(towerDir)\n".utf8))
}

// Warm up INSIDE the Task and start the listener from there too.
//
// The first cut blocked the main thread on a semaphore while a Task did the
// warm-up. That deadlocks: the engine reported `Running: 0 Pending: 0` for
// eight minutes -- no request had been submitted at all -- because the
// concurrency runtime had nothing left to run the Task's continuation on.
// It looked exactly like a hung model and was a hung harness.
Task {
    await session.warmUp { phase in
        FileHandle.standardError.write(Data("[tempo9] \(phase)\n".utf8))
    }
    do {
        let server = OpenAIServer(
            session: session, modelName: name, port: port,
            encodeImage: tower.map { t in
                { (image: CGImage) async throws -> ImagePlacement? in
                    let e = try t.encode(image: image)
                    return ImagePlacement(embedding: e.embedding, tokens: e.tokens,
                                          hidden: e.outHidden, gridH: e.gridH,
                                          gridW: e.gridW, contentKey: e.contentKey)
                }
            })
        server.defaultSpeculationK = specK
        // What Ollama's discovery calls report.  Size is read off the
        // file, so it is exact.  Quantization is DERIVED FROM THE NAME and
        // only when the name actually says -- the truthful source is the
        // GGUF header, and a guessed "q4_K_M" on a file that is not one is
        // worse than a blank field a user can ignore.
        var info = Tempo9ModelInfo()
        if let n = hfSize {
            // A snapshot path is a symlink into blobs/; its own size is
            // the link's, not the model's.
            info.sizeBytes = n
        } else if let e = ollamaEntry {
            // The path we hand the engine is a symlink, and asking the file
            // system for its size answers with the length of the link's
            // target string.  The store already measured the real blob.
            info.sizeBytes = e.sizeBytes
        } else if let attrs = try? FileManager.default
            .attributesOfItem(atPath: gguf),
                  let size = attrs[.size] as? NSNumber {
            info.sizeBytes = size.int64Value
        }
        if let m = name.range(of: "[qQ][2-8]_[0-9KkSsMmLl_]+",
                              options: .regularExpression) {
            info.quantization = String(name[m]).uppercased()
        }
        server.modelInfo = info
        try server.start()
        FileHandle.standardError.write(Data(
            "[tempo9] http://127.0.0.1:\(port)/v1 as \(name)\n".utf8))
        // Held so the listener is not deallocated the moment this Task ends.
        keptAlive = server
    } catch {
        FileHandle.standardError.write(Data("[tempo9] listen failed: \(error)\n".utf8))
        exit(1)
    }
}
dispatchMain()
