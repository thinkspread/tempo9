// Copyright (c) 2026 Jiejing Zhang.
//
// The Ollama store is read against a SYNTHETIC one built here, because the
// machine that wrote this had the ollama binary installed and had never run
// it -- there was no real store to read.  A fixture exercises every branch
// that can be exercised without one; what it cannot prove is the exact
// mediaType strings Ollama emits, which is why the reader falls back to the
// largest layer instead of trusting a spelling.

import Testing
import Foundation
@testable import Tempo9

@Suite("Ollama store")
struct OllamaStoreTests {

    /// Builds <tmp>/models with the manifests and blobs described.
    /// `layers` is (mediaType, bytes) per blob; digests are synthesised.
    static func makeStore(
        _ models: [(host: String, ns: String, model: String, tag: String,
                    layers: [(String, Int)])],
        missingBlobs: Bool = false
    ) throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ollama-fixture-" + UUID().uuidString)
            .appendingPathComponent("models")
        let fm = FileManager.default
        var counter = 0
        for m in models {
            var entries: [[String: Any]] = []
            for (type, size) in m.layers {
                counter += 1
                let digest = String(format:
                    "sha256:%064x", counter)
                entries.append(["mediaType": type, "digest": digest,
                                "size": size])
                if !missingBlobs {
                    let blob = root.appendingPathComponent("blobs")
                        .appendingPathComponent(
                            digest.replacingOccurrences(of: ":", with: "-"))
                    try fm.createDirectory(
                        at: blob.deletingLastPathComponent(),
                        withIntermediateDirectories: true)
                    try Data(repeating: 0x41, count: size).write(to: blob)
                }
            }
            let manifest = root.appendingPathComponent("manifests")
                .appendingPathComponent(m.host)
                .appendingPathComponent(m.ns)
                .appendingPathComponent(m.model)
                .appendingPathComponent(m.tag)
            try fm.createDirectory(at: manifest.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            try JSONSerialization.data(
                withJSONObject: ["schemaVersion": 2, "layers": entries])
                .write(to: manifest)
        }
        return root
    }

    @Test("a pulled model is found, named the way ollama names it")
    func listsDefaultRegistry() throws {
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "qwen3",
             tag: "8b",
             layers: [("application/vnd.ollama.image.model", 4096),
                      ("application/vnd.ollama.image.template", 64)]),
        ])
        let all = OllamaStore.list(root: root)
        #expect(all.count == 1)
        // registry.ollama.ai/library is implicit in Ollama's own display.
        #expect(all.first?.name == "qwen3:8b")
        #expect(all.first?.sizeBytes == 4096,
                "the weights layer, not the template, is the model")
    }

    @Test("digest colon becomes a dash in the blob filename")
    func digestMapsToFilename() throws {
        // The one transformation easy to get wrong, and it fails as a
        // missing file rather than as anything that says why.
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "m",
             tag: "latest",
             layers: [("application/vnd.ollama.image.model", 32)]),
        ])
        let path = OllamaStore.list(root: root).first?.ggufPath ?? ""
        #expect(path.contains("/blobs/sha256-"))
        #expect(!path.contains("sha256:"))
        #expect(FileManager.default.isReadableFile(atPath: path))
    }

    @Test("a bare name resolves to :latest, like ollama run")
    func bareNameMeansLatest() throws {
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "llama3",
             tag: "latest",
             layers: [("application/vnd.ollama.image.model", 128)]),
        ])
        #expect(OllamaStore.resolve("llama3", root: root)?.name
                == "llama3:latest")
        #expect(OllamaStore.resolve("llama3:latest", root: root) != nil)
        #expect(OllamaStore.resolve("llama3:70b", root: root) == nil,
                "a tag that was never pulled must not silently match")
    }

    @Test("a non-default host or namespace keeps its full name")
    func qualifiedName() throws {
        let root = try Self.makeStore([
            (host: "hf.co", ns: "someone", model: "custom", tag: "q4",
             layers: [("application/vnd.ollama.image.model", 64)]),
        ])
        let e = OllamaStore.list(root: root).first
        #expect(e?.name == "hf.co/someone/custom:q4")
        // and the short form still resolves, so a user can type less
        #expect(OllamaStore.resolve("custom:q4", root: root)?.name
                == "hf.co/someone/custom:q4")
    }

    @Test("without a .model layer, the largest blob is the weights")
    func fallsBackToLargestLayer() throws {
        // Ollama's mediaType constants are theirs to change; "the weights
        // are the biggest blob" is a far more stable fact than a spelling.
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "odd",
             tag: "latest",
             layers: [("application/vnd.ollama.image.future", 8192),
                      ("application/vnd.ollama.image.template", 16)]),
        ])
        #expect(OllamaStore.list(root: root).first?.sizeBytes == 8192)
    }

    @Test("a manifest whose blob is gone is not offered")
    func danglingManifestSkipped() throws {
        // Interrupted pulls and manual cleanups leave these behind.  Listing
        // one turns a clear "not found" into a confusing load failure later.
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "ghost",
             tag: "latest",
             layers: [("application/vnd.ollama.image.model", 512)]),
        ], missingBlobs: true)
        #expect(OllamaStore.list(root: root).isEmpty)
    }

    @Test("size comes from the file, not the manifest's claim")
    func sizeTrustsDisk() throws {
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "trunc",
             tag: "latest",
             layers: [("application/vnd.ollama.image.model", 100)]),
        ])
        // Truncate the blob behind the manifest's back, as an interrupted
        // pull does: the manifest still advertises 100.
        let e = try #require(OllamaStore.list(root: root).first)
        try Data(repeating: 0x41, count: 7).write(
            to: URL(fileURLWithPath: e.ggufPath))
        #expect(OllamaStore.list(root: root).first?.sizeBytes == 7)
    }

    @Test("a projector is reported so the caller can say it is unused")
    func projectorSurfaced() throws {
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "vlm",
             tag: "latest",
             layers: [("application/vnd.ollama.image.model", 2048),
                      ("application/vnd.ollama.image.projector", 256)]),
        ])
        #expect(OllamaStore.list(root: root).first?.projectorPath != nil)
    }

    @Test("the blob is linked under a .gguf name, in OUR cache only")
    func linkedPathNamesTheModel() throws {
        // The engine dispatches GGUF-vs-asgraph on the extension, and
        // Ollama's blobs have none -- handed one directly it exits silently.
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "qwen3",
             tag: "0.6b",
             layers: [("application/vnd.ollama.image.model", 64)]),
        ])
        let cache = root.deletingLastPathComponent()
            .appendingPathComponent("cache")
        let e = try #require(OllamaStore.list(root: root).first)
        let link = try OllamaStore.linkedPath(for: e, in: cache)

        #expect(link.hasSuffix("qwen3-0.6b.gguf"),
                "':' is not a filename character, and the name should stay legible")
        #expect(link.hasPrefix(cache.path), "the link belongs in OUR cache")
        let dest = try FileManager.default
            .destinationOfSymbolicLink(atPath: link)
        #expect(dest == e.ggufPath)
        // The store itself must be untouched: read-only, always.
        #expect(try FileManager.default.contentsOfDirectory(
            atPath: root.appendingPathComponent("blobs").path).count == 1)
    }

    @Test("a link left pointing at an old blob is replaced, not reused")
    func staleLinkReplaced() throws {
        // Re-pulling a tag gives the same name different bytes.  A link kept
        // from last time would serve the old weights forever, and nothing
        // about the symptom would point here.
        let root = try Self.makeStore([
            (host: "registry.ollama.ai", ns: "library", model: "m",
             tag: "latest",
             layers: [("application/vnd.ollama.image.model", 64)]),
        ])
        let cache = root.deletingLastPathComponent()
            .appendingPathComponent("cache2")
        let e = try #require(OllamaStore.list(root: root).first)
        try FileManager.default.createDirectory(
            at: cache, withIntermediateDirectories: true)
        let link = cache.appendingPathComponent("m-latest.gguf")
        try FileManager.default.createSymbolicLink(
            atPath: link.path, withDestinationPath: "/somewhere/old.gguf")

        let got = try OllamaStore.linkedPath(for: e, in: cache)
        #expect(got == link.path)
        #expect(try FileManager.default
            .destinationOfSymbolicLink(atPath: got) == e.ggufPath,
                "the stale destination must be replaced")
    }

    @Test("no Ollama installed is empty, not an error")
    func absentStoreIsEmpty() {
        let nowhere = URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)")
        #expect(OllamaStore.list(root: nowhere).isEmpty)
        #expect(OllamaStore.resolve("anything", root: nowhere) == nil)
    }
}
