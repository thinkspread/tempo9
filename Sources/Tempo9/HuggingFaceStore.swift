// Copyright (c) 2026 Jiejing Zhang.
//
// Fetching a GGUF from Hugging Face into Hugging Face's own cache.
//
// WHY: "brew install, then find a model, download it, remember where it
// went, pass the path" is four steps before the first token, and the
// third one loses people.  `tempo9 --hf owner/repo:Q4_K_S` makes it one.
//
// THE CACHE IS HUGGING FACE'S, NOT OURS.  Files land in the standard hub
// layout (HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub):
//
//   models--<owner>--<repo>/blobs/<sha256>                 the bytes
//   models--<owner>--<repo>/snapshots/<revision>/<path>    -> ../../blobs/<sha256>
//   models--<owner>--<repo>/refs/main                      <revision>
//
// so a GGUF already pulled by huggingface_hub, llama.cpp's -hf, or anything
// else that speaks this layout is found and not downloaded again, and what
// we download is theirs to reuse.  A second copy of a 5-20 GB file is a
// disk-space incident, not a cache.  The partial file is
// blobs/<sha256>.incomplete, the name huggingface_hub resumes from.
//
// Package access, not public: the CLI and the tests use it, and it is not
// an SDK promise.

import CryptoKit
import Foundation

package enum HuggingFaceStore {

    // MARK: - Reference

    /// `owner/repo[:quant]`.  `hf.co/` and `https://huggingface.co/`
    /// prefixes are accepted because that is what people paste.
    package struct Ref: Equatable, Sendable {
        package let repo: String
        package let quant: String?
    }

    package static func parse(_ text: String) -> Ref? {
        var s = text.trimmingCharacters(in: .whitespaces)
        for prefix in ["https://", "http://"] where s.hasPrefix(prefix) {
            s.removeFirst(prefix.count)
        }
        for prefix in ["huggingface.co/", "hf.co/"] where s.hasPrefix(prefix) {
            s.removeFirst(prefix.count)
        }
        var quant: String?
        if let colon = s.lastIndex(of: ":") {
            quant = String(s[s.index(after: colon)...])
            s = String(s[..<colon])
            if quant!.isEmpty { quant = nil }
        }
        let parts = s.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty,
              !s.contains(" ") else { return nil }
        return Ref(repo: s, quant: quant)
    }

    // MARK: - Choosing the file

    package struct RemoteFile: Equatable, Sendable {
        package let path: String       // repo-relative, may contain '/'
        package let size: Int64
        package let sha256: String     // LFS oid; the blob's name

        package init(path: String, size: Int64, sha256: String) {
            self.path = path; self.size = size; self.sha256 = sha256
        }
    }

    package enum ChoiceError: Error, Equatable {
        case noGGUF
        /// The tag matched nothing; `available` lists the tags that exist.
        case noMatch(quant: String, available: [String])
        /// No tag given and no default present: the user has to pick.
        case ambiguous(available: [String])
        /// A multi-part GGUF.  The engine loads one file; saying so beats a
        /// download that ends in a load failure.
        case split(path: String)
    }

    /// When no tag is given.  Q4_K_S first: on Qwen3.5-9B it decodes ~10 %
    /// faster than Q4_K_M on this engine -- Q4_K_M carries q6_K tensors and
    /// q6_K has no fast kernel here.
    package static let defaultQuants = ["Q4_K_S", "Q4_K_M", "Q8_0"]

    /// Weights only: vision projectors and MTP heads are GGUFs too, and
    /// neither is a model this CLI can serve.
    static func isWeights(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent.lowercased()
        return name.hasSuffix(".gguf") && !name.hasPrefix("mmproj")
            && !name.hasPrefix("mtp") && !path.lowercased().hasPrefix("mtp/")
    }

    static let splitPattern = #"-\d{5}-of-\d{5}\.gguf$"#

    /// The quant tag a file name carries: what follows the last '-' (or
    /// '.') before `.gguf`, with a `UD-` prefix kept.  `Qwen3.5-9B-Q4_K_S`
    /// -> `Q4_K_S`; `gemma-4-E4B-it-UD-Q4_K_XL` -> `UD-Q4_K_XL`.
    static func tag(of path: String) -> String {
        var name = (path as NSString).lastPathComponent
        if let r = name.range(of: splitPattern, options: .regularExpression) {
            name = String(name[..<r.lowerBound]) + ".gguf"
        }
        name = String(name.dropLast(".gguf".count))
        let sep = name.lastIndex(where: { $0 == "-" || $0 == "." })
        guard let sep else { return name }
        let last = String(name[name.index(after: sep)...])
        let head = name[..<sep]
        if head.uppercased().hasSuffix("-UD") || head.uppercased().hasSuffix(".UD") {
            return "UD-" + last
        }
        return last
    }

    package static func choose(_ files: [RemoteFile], quant: String?)
        -> Result<RemoteFile, ChoiceError> {
        let weights = files.filter { isWeights($0.path) }
        guard !weights.isEmpty else { return .failure(.noGGUF) }
        var tags: [String] = []
        for f in weights where !tags.contains(tag(of: f.path)) {
            tags.append(tag(of: f.path))
        }
        func pick(_ q: String) -> RemoteFile? {
            let hits = weights.filter { tag(of: $0.path).uppercased() == q.uppercased() }
            // A file in the repo root wins over the same tag in a
            // subdirectory; further ties are rare and go to the first.
            return hits.first(where: { !$0.path.contains("/") }) ?? hits.first
        }
        let chosen: RemoteFile
        if let q = quant {
            guard let f = pick(q) else {
                return .failure(.noMatch(quant: q, available: tags))
            }
            chosen = f
        } else if weights.count == 1 {
            chosen = weights[0]
        } else if let f = defaultQuants.lazy.compactMap(pick).first {
            chosen = f
        } else {
            return .failure(.ambiguous(available: tags))
        }
        if chosen.path.range(of: splitPattern, options: .regularExpression) != nil {
            return .failure(.split(path: chosen.path))
        }
        return .success(chosen)
    }

    // MARK: - Cache layout

    package static func hubRoot(
        env: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let c = env["HF_HUB_CACHE"], !c.isEmpty {
            return URL(fileURLWithPath: (c as NSString).expandingTildeInPath)
        }
        if let h = env["HF_HOME"], !h.isEmpty {
            return URL(fileURLWithPath: (h as NSString).expandingTildeInPath)
                .appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    package static func repoDir(_ repo: String, root: URL) -> URL {
        root.appendingPathComponent(
            "models--" + repo.replacingOccurrences(of: "/", with: "--"))
    }

    /// A complete copy of `file` already in the cache, under any revision:
    /// the bytes are named by their hash, so the revision only matters for
    /// which snapshot links to them.
    package static func cached(_ file: RemoteFile, repo: String, root: URL) -> URL? {
        let blob = repoDir(repo, root: root).appendingPathComponent("blobs/\(file.sha256)")
        guard let a = try? FileManager.default.attributesOfItem(atPath: blob.path),
              (a[.size] as? NSNumber)?.int64Value == file.size else { return nil }
        return blob
    }

    /// Any cached snapshot of a file with this tag -- the offline path,
    /// when the listing cannot be fetched.  Newest revision by refs/main
    /// first, else whatever snapshot holds it.
    package static func cachedOffline(_ ref: Ref, root: URL) -> URL? {
        let dir = repoDir(ref.repo, root: root)
        let snaps = dir.appendingPathComponent("snapshots")
        var revs = (try? FileManager.default.contentsOfDirectory(atPath: snaps.path)) ?? []
        if let main = try? String(contentsOf: dir.appendingPathComponent("refs/main"),
                                  encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           let i = revs.firstIndex(of: main) {
            revs.swapAt(0, i)
        }
        for rev in revs {
            let base = snaps.appendingPathComponent(rev)
            guard let e = FileManager.default.enumerator(atPath: base.path) else { continue }
            let files: [RemoteFile] = e.compactMap { p in
                guard let p = p as? String, isWeights(p) else { return nil }
                let full = base.appendingPathComponent(p).resolvingSymlinksInPath()
                guard FileManager.default.fileExists(atPath: full.path) else { return nil }
                return RemoteFile(path: p, size: 0, sha256: "")
            }
            if case .success(let f) = choose(files, quant: ref.quant) {
                return base.appendingPathComponent(f.path)
            }
        }
        return nil
    }

    /// Link snapshots/<revision>/<path> to the blob and record refs/main.
    /// Returns the snapshot path, which ends in `.gguf` -- the engine
    /// dispatches on the extension, and a blob has none.
    @discardableResult
    package static func link(_ file: RemoteFile, repo: String, revision: String,
                             root: URL, isMain: Bool = true) throws -> URL {
        let dir = repoDir(repo, root: root)
        let fm = FileManager.default
        let snap = dir.appendingPathComponent("snapshots/\(revision)/\(file.path)")
        try fm.createDirectory(at: snap.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        let depth = file.path.split(separator: "/").count - 1
        let target = String(repeating: "../", count: 2 + depth) + "blobs/\(file.sha256)"
        if (try? fm.destinationOfSymbolicLink(atPath: snap.path)) != target {
            try? fm.removeItem(at: snap)
            try fm.createSymbolicLink(atPath: snap.path, withDestinationPath: target)
        }
        if isMain {
            let refs = dir.appendingPathComponent("refs")
            try fm.createDirectory(at: refs, withIntermediateDirectories: true)
            try Data(revision.utf8).write(to: refs.appendingPathComponent("main"))
        }
        return snap
    }

    // MARK: - Network

    package struct Listing: Sendable {
        package let revision: String
        package let files: [RemoteFile]
    }

    package enum FetchError: Error, CustomStringConvertible {
        case http(Int, String)
        case transport(String)
        case malformed(String)
        case diskFull(need: Int64, free: Int64)
        case checksum(expected: String, got: String)

        package var description: String {
            switch self {
            // The Hub answers 401 for a repo that does not exist, too (so
            // as not to reveal private names) -- a typo lands here more
            // often than a gated model does.
            case .http(401, let what), .http(403, let what):
                return "\(what): no such repo, or it is gated/private"
                    + " (then set HF_TOKEN)"
            case .http(404, let what):
                return "\(what): not found"
            case .http(let code, let what): return "\(what): HTTP \(code)"
            case .transport(let m): return m
            case .malformed(let m): return m
            case .diskFull(let need, let free):
                return String(format: "not enough disk space: need %.1f GB more, %.1f GB free",
                              Double(need) / 1e9, Double(free) / 1e9)
            case .checksum(let e, let g):
                return "downloaded file does not match its sha256 (expected \(e.prefix(12)), got \(g.prefix(12))); removed"
            }
        }
    }

    static let endpoint: String = {
        let e = ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? ""
        return e.isEmpty ? "https://huggingface.co" : e
    }()

    static func request(_ url: URL) -> URLRequest {
        var r = URLRequest(url: url)
        r.setValue("tempo9", forHTTPHeaderField: "User-Agent")
        if let t = ProcessInfo.processInfo.environment["HF_TOKEN"], !t.isEmpty {
            r.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization")
        }
        return r
    }

    /// The repo's files with their LFS hashes and sizes, and the revision
    /// they belong to: one call to the models API with `blobs=true`.
    package static func listing(_ repo: String) throws -> Listing {
        guard let url = URL(string: "\(endpoint)/api/models/\(repo)?blobs=true") else {
            throw FetchError.malformed("bad repo name '\(repo)'")
        }
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var out: (Data?, URLResponse?, Error?)
        URLSession.shared.dataTask(with: request(url)) { d, r, e in
            out = (d, r, e); sem.signal()
        }.resume()
        sem.wait()
        if let e = out.2 { throw FetchError.transport(e.localizedDescription) }
        let code = (out.1 as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200, let data = out.0 else { throw FetchError.http(code, repo) }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sha = obj["sha"] as? String,
              let sib = obj["siblings"] as? [[String: Any]] else {
            throw FetchError.malformed("unexpected listing for \(repo)")
        }
        let files: [RemoteFile] = sib.compactMap { s in
            guard let p = s["rfilename"] as? String,
                  let lfs = s["lfs"] as? [String: Any],
                  let oid = lfs["sha256"] as? String ?? lfs["oid"] as? String,
                  let size = (lfs["size"] as? NSNumber)?.int64Value else { return nil }
            return RemoteFile(path: p, size: size, sha256: oid)
        }
        return Listing(revision: sha, files: files)
    }

    /// Download into blobs/<sha256>, resuming a `.incomplete` left by an
    /// earlier run (ours or huggingface_hub's), verify the hash, and link
    /// the snapshot.  `progress(done, total)` is called from a background
    /// queue, at most a few times a second.
    package static func fetch(_ file: RemoteFile, repo: String, revision: String,
                              root: URL,
                              progress: @escaping @Sendable (Int64, Int64) -> Void)
        throws -> URL {
        if cached(file, repo: repo, root: root) != nil {
            return try link(file, repo: repo, revision: revision, root: root)
        }
        let fm = FileManager.default
        let blobs = repoDir(repo, root: root).appendingPathComponent("blobs")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        let blob = blobs.appendingPathComponent(file.sha256)
        let part = blobs.appendingPathComponent(file.sha256 + ".incomplete")

        var have = ((try? fm.attributesOfItem(atPath: part.path))?[.size]
                    as? NSNumber)?.int64Value ?? 0
        if have > file.size { try? fm.removeItem(at: part); have = 0 }
        let free = ((try? blobs.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage).map { Int64($0) } ?? .max
        if file.size - have > free {
            throw FetchError.diskFull(need: file.size - have, free: free)
        }
        if have < file.size {
            if !fm.fileExists(atPath: part.path) {
                fm.createFile(atPath: part.path, contents: nil)
            }
            let path = file.path.split(separator: "/")
                .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)! }
                .joined(separator: "/")
            guard let url = URL(string: "\(endpoint)/\(repo)/resolve/\(revision)/\(path)")
            else { throw FetchError.malformed("bad file path \(file.path)") }
            try Downloader(url: url, to: part, offset: have, total: file.size,
                           progress: progress).run()
        }

        let got = try sha256(of: part)
        guard got == file.sha256 else {
            try? fm.removeItem(at: part)
            throw FetchError.checksum(expected: file.sha256, got: got)
        }
        try? fm.removeItem(at: blob)
        try fm.moveItem(at: part, to: blob)
        return try link(file, repo: repo, revision: revision, root: root)
    }

    static func sha256(of url: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while let chunk = try h.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// One ranged GET streamed to the end of a file.  Delegate-based on its own
/// queue so the caller can simply block: the CLI has no run loop yet when
/// it downloads, and nothing here touches the main thread.
private final class Downloader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let url: URL, offset: Int64, total: Int64
    let out: FileHandle
    let progress: @Sendable (Int64, Int64) -> Void
    var done: Int64
    var lastTick = Date.distantPast
    var failure: Error?
    let finished = DispatchSemaphore(value: 0)

    init(url: URL, to file: URL, offset: Int64, total: Int64,
         progress: @escaping @Sendable (Int64, Int64) -> Void) throws {
        self.url = url; self.offset = offset; self.total = total
        self.progress = progress; self.done = offset
        out = try FileHandle(forWritingTo: file)
        try out.seekToEnd()
    }

    func run() throws {
        let q = OperationQueue(); q.maxConcurrentOperationCount = 1
        let s = URLSession(configuration: .default, delegate: self, delegateQueue: q)
        var r = HuggingFaceStore.request(url)
        if offset > 0 { r.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        s.dataTask(with: r).resume()
        finished.wait()
        s.finishTasksAndInvalidate()
        try? out.close()
        if let failure { throw failure }
        if done != total {
            throw HuggingFaceStore.FetchError.transport(
                "download ended at \(done) of \(total) bytes; run again to resume")
        }
    }

    func urlSession(_ s: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        // 206 continues the partial file; 200 to a ranged request means the
        // server ignored the range, so start the file over.
        if code == 200 && offset > 0 {
            try? out.truncate(atOffset: 0); done = 0
        } else if code != 200 && code != 206 {
            failure = HuggingFaceStore.FetchError.http(code, url.lastPathComponent)
            completionHandler(.cancel); return
        }
        completionHandler(.allow)
    }

    func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do { try out.write(contentsOf: data) } catch {
            failure = error; dataTask.cancel(); return
        }
        done += Int64(data.count)
        let now = Date()
        if now.timeIntervalSince(lastTick) >= 0.25 || done == total {
            lastTick = now; progress(done, total)
        }
    }

    func urlSession(_ s: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error, failure == nil {
            failure = HuggingFaceStore.FetchError.transport(error.localizedDescription)
        }
        finished.signal()
    }
}
