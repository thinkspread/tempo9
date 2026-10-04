// Copyright (c) 2026 Jiejing Zhang.
//
// `--hf` resolution against listings and caches built here: no network.
// The file lists are real ones (unsloth/Qwen3.5-9B-GGUF,
// unsloth/Qwen3.5-35B-A3B-GGUF, unsloth/gemma-4-E4B-it-GGUF as of
// 2026-10-02), trimmed, because the naming is the thing being tested.

import Testing
import Foundation
@testable import Tempo9

@Suite("Hugging Face store")
struct HuggingFaceStoreTests {
    typealias F = HuggingFaceStore.RemoteFile

    static func files(_ names: [String]) -> [F] {
        names.enumerated().map { F(path: $1, size: Int64($0 + 1), sha256: "h\($0)") }
    }

    static let qwen9 = files([
        "Qwen3.5-9B-BF16.gguf", "Qwen3.5-9B-IQ4_XS.gguf", "Qwen3.5-9B-Q4_K_M.gguf",
        "Qwen3.5-9B-Q4_K_S.gguf", "Qwen3.5-9B-Q8_0.gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "mmproj-BF16.gguf", "mmproj-F16.gguf",
    ])

    @Test func parsesReferences() {
        #expect(HuggingFaceStore.parse("unsloth/Qwen3.5-9B-GGUF:Q4_K_S")
                == .init(repo: "unsloth/Qwen3.5-9B-GGUF", quant: "Q4_K_S"))
        #expect(HuggingFaceStore.parse("hf.co/unsloth/Qwen3.5-9B-GGUF")
                == .init(repo: "unsloth/Qwen3.5-9B-GGUF", quant: nil))
        #expect(HuggingFaceStore.parse("https://huggingface.co/a/b:q8_0")
                == .init(repo: "a/b", quant: "q8_0"))
        #expect(HuggingFaceStore.parse("just-a-name") == nil)
        #expect(HuggingFaceStore.parse("a/b/c") == nil)
    }

    @Test func quantTagIsTheLastField() {
        #expect(HuggingFaceStore.tag(of: "Qwen3.5-9B-Q4_K_S.gguf") == "Q4_K_S")
        #expect(HuggingFaceStore.tag(of: "gemma-4-E4B-it-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL")
        #expect(HuggingFaceStore.tag(of: "BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf")
                == "BF16")
    }

    @Test func explicitTagMatchesExactlyCaseInsensitive() throws {
        let f = try HuggingFaceStore.choose(Self.qwen9, quant: "q4_k_s").get()
        #expect(f.path == "Qwen3.5-9B-Q4_K_S.gguf")
        // Q4_K_XL is the UD file's tag only with its prefix; a bare
        // "Q4_K_XL" must not silently pick it up as a different quant.
        let ud = try HuggingFaceStore.choose(Self.qwen9, quant: "UD-Q4_K_XL").get()
        #expect(ud.path == "Qwen3.5-9B-UD-Q4_K_XL.gguf")
        guard case .failure(.noMatch(_, let have)) =
            HuggingFaceStore.choose(Self.qwen9, quant: "Q4_K_XL") else {
            Issue.record("Q4_K_XL should not match UD-Q4_K_XL"); return
        }
        #expect(have.contains("UD-Q4_K_XL") && !have.contains { $0.hasPrefix("mmproj") })
    }

    @Test func noTagPrefersQ4KS() throws {
        #expect(try HuggingFaceStore.choose(Self.qwen9, quant: nil).get().path
                == "Qwen3.5-9B-Q4_K_S.gguf")
        let noKS = Self.qwen9.filter { !$0.path.contains("Q4_K_S") }
        #expect(try HuggingFaceStore.choose(noKS, quant: nil).get().path
                == "Qwen3.5-9B-Q4_K_M.gguf")
        let odd = Self.files(["m-IQ4_XS.gguf", "m-Q3_K_M.gguf", "mmproj-F16.gguf"])
        guard case .failure(.ambiguous(let have)) = HuggingFaceStore.choose(odd, quant: nil)
        else { Issue.record("expected ambiguous"); return }
        #expect(have == ["IQ4_XS", "Q3_K_M"])
    }

    @Test func projectorsAndMTPHeadsAreNotWeights() {
        let gemma = Self.files(["MTP/mtp-gemma-4-E4B-it-Q8_0.gguf",
                                "mtp-gemma-4-E4B-it.gguf", "mmproj-F16.gguf",
                                "gemma-4-E4B-it-Q8_0.gguf"])
        #expect((try? HuggingFaceStore.choose(gemma, quant: "Q8_0").get().path)
                == "gemma-4-E4B-it-Q8_0.gguf")
        #expect({ if case .failure(.noGGUF) =
                    HuggingFaceStore.choose(Self.files(["mmproj-F16.gguf"]), quant: nil)
                  { return true }; return false }())
    }

    @Test func splitFilesAreRefusedByName() {
        let moe = Self.files(["BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",
                              "BF16/Qwen3.5-35B-A3B-BF16-00002-of-00002.gguf",
                              "Qwen3.5-35B-A3B-Q3_K_M.gguf"])
        guard case .failure(.split(let p)) = HuggingFaceStore.choose(moe, quant: "BF16") else {
            Issue.record("a split quant must be refused before downloading"); return
        }
        #expect(p.hasSuffix("-00001-of-00002.gguf"))
        #expect((try? HuggingFaceStore.choose(moe, quant: "Q3_K_M").get().path)
                == "Qwen3.5-35B-A3B-Q3_K_M.gguf")
    }

    @Test func hubRootFollowsHuggingFaceEnvironment() {
        #expect(HuggingFaceStore.hubRoot(env: ["HF_HUB_CACHE": "/x/hub"]).path == "/x/hub")
        #expect(HuggingFaceStore.hubRoot(env: ["HF_HOME": "/y"]).path == "/y/hub")
        #expect(HuggingFaceStore.hubRoot(env: [:]).path.hasSuffix(".cache/huggingface/hub"))
    }

    /// The layout huggingface_hub writes: a snapshot entry is a RELATIVE
    /// symlink to blobs/<sha256>, so the cache survives being moved, and
    /// refs/main names the revision.
    @Test func linkWritesTheHubLayout() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-fixture-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = "o/r", rev = "abc123"
        let f = F(path: "sub/m-Q4_K_S.gguf", size: 5, sha256: "deadbeef")
        let blobs = HuggingFaceStore.repoDir(repo, root: root).appendingPathComponent("blobs")
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try Data("12345".utf8).write(to: blobs.appendingPathComponent("deadbeef"))

        #expect(HuggingFaceStore.cached(f, repo: repo, root: root) != nil)
        let wrongSize = F(path: f.path, size: 6, sha256: f.sha256)
        #expect(HuggingFaceStore.cached(wrongSize, repo: repo, root: root) == nil)

        let snap = try HuggingFaceStore.link(f, repo: repo, revision: rev, root: root)
        #expect(snap.path.hasSuffix("models--o--r/snapshots/abc123/sub/m-Q4_K_S.gguf"))
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: snap.path)
                == "../../../blobs/deadbeef")
        #expect(try String(contentsOf: snap, encoding: .utf8) == "12345")
        #expect(try String(contentsOf: HuggingFaceStore.repoDir(repo, root: root)
                .appendingPathComponent("refs/main"), encoding: .utf8) == rev)

        // Offline: the snapshot is found by tag with no listing at all.
        let off = HuggingFaceStore.cachedOffline(.init(repo: repo, quant: "Q4_K_S"), root: root)
        #expect(off?.lastPathComponent == "m-Q4_K_S.gguf")
        #expect(HuggingFaceStore.cachedOffline(.init(repo: repo, quant: "Q8_0"), root: root)
                == nil)
    }

    @Test func checksumIsSHA256OfTheBytes() throws {
        let u = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hf-sha-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: u) }
        try Data("abc".utf8).write(to: u)
        #expect(try HuggingFaceStore.sha256(of: u)
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }
}
