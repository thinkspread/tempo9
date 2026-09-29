// Copyright (c) 2026 Jiejing Zhang.
//
// L1 vision-embedding cache, content addressed.
//
// The key is not just a cache key: it is handed to the engine as
// `hash_input`, which substitutes it for the image placeholder tokens in the
// prefix-cache chain, so a second question about the same image reuses its
// prefill. That makes the key a wire format shared with
// python/pyhie/allspark/vlm/embedding_cache.py -- the two must produce the
// same number for the same image or the two clients quietly stop sharing
// engine-side cache entries.

import CryptoKit
import Foundation

public enum ContentKey {
    /// SHA-256, first 8 bytes little-endian.
    ///
    /// SHA-256 rather than something faster because both sides have to agree
    /// and it is the one strong hash present in both CryptoKit and Python's
    /// hashlib without a dependency.
    static func hash64(_ data: Data) -> UInt64 {
        var digest = SHA256()
        digest.update(data: data)
        let bytes = Array(digest.finalize())
        return bytes.prefix(8).reversed().reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// Key over the *preprocessed* patches, the grid, and the tower build.
    ///
    /// Patches are hashed as fp16 so a value that survives the fp16 cache
    /// round-trip cannot change the key.
    public static func make(patches: [Float], gridH: Int, gridW: Int,
                            fingerprint: String) -> Int64 {
        var half = [Float16](repeating: 0, count: patches.count)
        for i in 0..<patches.count { half[i] = Float16(patches[i]) }
        let patchHash = half.withUnsafeBufferPointer {
            hash64(Data(buffer: $0))
        }
        let meta = "\(gridH)x\(gridW)|\(fingerprint)"
        let metaHash = hash64(Data(meta.utf8))
        return Int64(bitPattern: (patchHash ^ metaHash) & 0x7FFF_FFFF_FFFF_FFFF)
    }
}

/// Process-level LRU with a byte budget.
public final class EmbeddingCache {
    private var entries: [Int64: [Float16]] = [:]
    private var order: [Int64] = []
    private var bytes = 0
    private let budget: Int
    private let lock = NSLock()

    public private(set) var hits = 0
    public private(set) var misses = 0

    /// Measurement affordance: with this false every `get` misses, so a
    /// loop over ONE image measures the tower N times instead of once.
    /// A zero byte budget cannot do this -- eviction deliberately keeps
    /// the last entry (`order.count > 1`), so a single-image loop hits
    /// every time and reports the tower as free.
    public var bypass = false

    public init(byteBudget: Int = 1 << 30) {
        self.budget = byteBudget
    }

    public func get(_ key: Int64) -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        guard !bypass, let stored = entries[key] else {
            misses += 1
            return nil
        }
        hits += 1
        touch(key)
        return stored.map(Float.init)
    }

    /// Store, and return the value *as a later hit will see it*.
    ///
    /// Handing back the caller's fp32 while storing fp16 would make the first
    /// answer about an image differ from every later one; the mismatch is
    /// small but at greedy decoding it is enough to change wording. Same
    /// contract as the Python cache.
    @discardableResult
    public func put(_ key: Int64, embedding: [Float]) -> [Float] {
        let half = embedding.map { Float16($0) }
        lock.lock()
        defer { lock.unlock() }
        if let existing = entries[key] {
            touch(key)
            return existing.map(Float.init)
        }
        entries[key] = half
        order.append(key)
        bytes += half.count * MemoryLayout<Float16>.size
        while bytes > budget, order.count > 1 {
            let evicted = order.removeFirst()
            if let gone = entries.removeValue(forKey: evicted) {
                bytes -= gone.count * MemoryLayout<Float16>.size
            }
        }
        return half.map(Float.init)
    }

    private func touch(_ key: Int64) {
        if let index = order.firstIndex(of: key) {
            order.remove(at: index)
            order.append(key)
        }
    }

    public var stats: (hits: Int, misses: Int, entries: Int, bytes: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (hits, misses, entries.count, bytes)
    }
}
