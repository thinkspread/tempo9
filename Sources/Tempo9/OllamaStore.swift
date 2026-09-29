// Copyright (c) 2026 Jiejing Zhang.
//
// Reading the models Ollama already pulled.
//
// WHY: the cost of trying another engine is not the download of the engine,
// it is the re-download of the weights.  Nobody re-fetches 40 GB to run a
// comparison, and every comparison we would win is downstream of someone
// bothering to try.  Ollama stores each pulled model as a BARE GGUF blob
// with a small JSON manifest pointing at it, and bare GGUF is exactly what
// this engine loads -- so the switching cost can be made zero by reading,
// with no download, no copy, and no conversion.
//
// STRICTLY READ-ONLY.  Nothing here creates, moves, or deletes a file under
// the Ollama root.  That store belongs to another program and a user's
// pulled models are expensive; corrupting one to save ourselves a download
// would be unforgivable, so the only calls made here are directory listing,
// file reads, and attribute lookups.
//
// Layout (verified against Ollama's published format):
//
//   <root>/manifests/<host>/<namespace>/<model>/<tag>   small JSON
//   <root>/blobs/sha256-<hex>                           the bytes
//
// The manifest is OCI-shaped: a `layers` array whose entries carry a
// mediaType, a digest, and a size.  The weights are the layer whose type
// ends in `.model`; a vision model also carries one ending in `.projector`.
// A digest "sha256:abc" names the file "sha256-abc" -- colon to dash, which
// is the one transformation easy to get wrong and it fails as a missing
// file rather than as anything informative.

import Foundation

public enum OllamaStore {

    public struct Entry: Sendable {
        /// Display name in Ollama's own convention: `qwen3:8b` for the
        /// default registry and namespace, `host/ns/model:tag` otherwise.
        public let name: String
        public let ggufPath: String
        public let sizeBytes: Int64
        /// A GGUF vision projector, when the model has one.
        ///
        /// Parsed because it is real information and free to collect, but
        /// NOT loadable by this CLI: our vision path takes a Core ML tower
        /// directory, not a GGUF mmproj.  Callers should say so rather than
        /// ignore it, so a user with a vision model is not left wondering
        /// why images do nothing.
        public let projectorPath: String?
    }

    /// Where Ollama keeps its models.  `OLLAMA_MODELS` overrides, which is
    /// how anyone who moved the store off their boot volume points at it --
    /// the most common reason our lookup would otherwise come up empty.
    public static var root: URL {
        if let custom = ProcessInfo.processInfo.environment["OLLAMA_MODELS"],
           !custom.isEmpty {
            return URL(fileURLWithPath: custom)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ollama/models")
    }

    /// Every model Ollama has pulled, sorted by name.
    ///
    /// An unreadable or absent store yields an empty list rather than an
    /// error: not having Ollama installed is the normal case, not a fault.
    public static func list(root: URL = OllamaStore.root) -> [Entry] {
        let manifests = root.appendingPathComponent("manifests")
        let fm = FileManager.default
        guard let walker = fm.enumerator(at: manifests,
                                         includingPropertiesForKeys: [
                                            .isRegularFileKey],
                                         options: [.skipsHiddenFiles])
        else { return [] }

        var out: [Entry] = []
        for case let url as URL in walker {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey])
                .isRegularFile) == true else { continue }
            // <manifests>/<host>/<ns>/<model>/<tag> -- anything shallower is
            // not a manifest, and skipping quietly beats guessing.
            let rel = url.path.dropFirst(manifests.path.count)
                .split(separator: "/").map(String.init)
            guard rel.count >= 4 else { continue }
            let tag = rel[rel.count - 1]
            let model = rel[rel.count - 2]
            let ns = rel[rel.count - 3]
            let host = rel[rel.count - 4]
            guard let e = entry(manifest: url, root: root, host: host,
                                namespace: ns, model: model, tag: tag)
            else { continue }
            out.append(e)
        }
        return out.sorted { $0.name < $1.name }
    }

    /// Resolve `qwen3:8b`, `qwen3`, or a fully qualified name to an entry.
    ///
    /// A bare name means `:latest`, matching `ollama run`.  Matching is done
    /// against the same display names `list()` reports, so whatever a user
    /// sees in `--list-ollama` is what they can pass.
    public static func resolve(_ ref: String,
                               root: URL = OllamaStore.root) -> Entry? {
        let want = ref.contains(":") ? ref : ref + ":latest"
        let all = list(root: root)
        if let hit = all.first(where: { $0.name == want }) { return hit }
        // Fully-qualified store entries still answer to their short name.
        return all.first { $0.name.hasSuffix("/" + want) }
    }

    // MARK: - naming a blob so the engine will load it

    /// A `.gguf`-named path for an entry's blob.
    ///
    /// Ollama's blobs are extensionless (`sha256-<hex>`), and the engine
    /// dispatches GGUF-vs-asgraph on the file extension: handed a blob path
    /// directly it refuses, and it refuses SILENTLY -- the process builds an
    /// engine, tears it straight back down, and exits 0 with nothing on
    /// stderr that names the cause.  (Fixing that dispatch to sniff the GGUF
    /// magic instead is the real repair and lives in the C++ engine.)
    ///
    /// The link is created in OUR cache, never in Ollama's store, which
    /// stays read-only.  Naming it after the model rather than the digest
    /// also gives the engine's graph cache a legible key -- the cache is
    /// keyed on basename plus size plus mtime, so `qwen3-0.6b.gguf`
    /// identifies itself in `~/.cache/tempo9/gguf_graphs/` where
    /// `sha256-aaaa...` would not.
    public static func linkedPath(for entry: Entry,
                                  in cache: URL? = nil) throws -> String {
        let fm = FileManager.default
        let dir = cache ?? fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/tempo9/ollama")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        // ':' and '/' are the two characters Ollama names carry that a
        // filename should not.
        let safe = entry.name
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: "/", with: "_")
        let link = dir.appendingPathComponent(safe + ".gguf")

        // A tag can be re-pulled to different bytes, so an existing link
        // that points somewhere else is stale, not a conflict.
        if let dest = try? fm.destinationOfSymbolicLink(atPath: link.path) {
            if dest == entry.ggufPath { return link.path }
            try fm.removeItem(at: link)
        } else if fm.fileExists(atPath: link.path) {
            try fm.removeItem(at: link)
        }
        try fm.createSymbolicLink(atPath: link.path,
                                  withDestinationPath: entry.ggufPath)
        return link.path
    }

    // MARK: - manifest

    static func entry(manifest url: URL, root: URL, host: String,
                      namespace ns: String, model: String,
                      tag: String) -> Entry? {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data)
                  as? [String: Any],
              let layers = obj["layers"] as? [[String: Any]]
        else { return nil }

        func blob(_ digest: String) -> String {
            root.appendingPathComponent("blobs")
                .appendingPathComponent(
                    digest.replacingOccurrences(of: ":", with: "-")).path
        }

        var weights: (path: String, size: Int64)?
        var projector: String?
        var largest: (path: String, size: Int64)?

        for l in layers {
            guard let digest = l["digest"] as? String else { continue }
            let type = l["mediaType"] as? String ?? ""
            let size = (l["size"] as? NSNumber)?.int64Value ?? 0
            let path = blob(digest)
            if type.hasSuffix(".model") { weights = (path, size) }
            if type.hasSuffix(".projector") { projector = path }
            if largest == nil || size > largest!.size {
                largest = (path, size)
            }
        }

        // Fall back to the biggest layer when no mediaType says `.model`.
        // Ollama's type strings are theirs to change, and the weights being
        // the largest blob is a far more stable fact than the spelling of a
        // constant; a wrong guess here fails loudly at model load, not
        // silently.
        guard let picked = weights ?? largest else { return nil }
        // A manifest can outlive its blobs (an interrupted pull, a manual
        // cleanup).  Listing a model we cannot open would turn a clear
        // "not found" into a confusing load failure later.
        guard FileManager.default.isReadableFile(atPath: picked.path) else {
            return nil
        }

        let short = "\(model):\(tag)"
        let name = (host == "registry.ollama.ai" && ns == "library")
            ? short : "\(host)/\(ns)/\(short)"
        // Trust the file over the manifest: a truncated pull leaves the
        // manifest's advertised size intact and the blob short.
        let onDisk = (try? FileManager.default
            .attributesOfItem(atPath: picked.path)[.size] as? NSNumber)?
            .int64Value
        return Entry(name: name, ggufPath: picked.path,
                     sizeBytes: onDisk ?? picked.size,
                     projectorPath: projector)
    }
}
