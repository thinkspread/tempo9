// Copyright (c) 2026 Jiejing Zhang.
// Portions derived from DashInfer/AllSpark (python/pyhie/allspark/gguf_utils.py),
// Copyright (c) Alibaba, Inc. and its affiliates, Apache-2.0. See NOTICE.
//
// GGUF container reader.
//
// A .gguf is self-contained in a way that matters for embedding this stack
// somewhere without Python: besides the weights and the hyperparameters it
// carries the tokenizer vocabulary, the BPE merges and the chat template.
// Those last three are the parts a host cannot precompute and cache -- a
// graph is built once per model, but tokenization happens on every request.
//
// Port of the reader in python/pyhie/allspark/gguf_utils.py, with one
// deliberate difference: that one skips the vocab and merge arrays (it only
// needs the header), and this one is here precisely to read them.

import Foundation

public enum GGUFError: LocalizedError {
    case notGGUF(String)
    case unsupportedVersion(UInt32)
    case truncated(String)
    case unknownValueType(UInt32)
    case missingKey(String)

    public var errorDescription: String? {
        switch self {
        case .notGGUF(let path): return "\(path): not a GGUF file"
        case .unsupportedVersion(let v): return "unsupported GGUF version \(v)"
        case .truncated(let what): return "GGUF truncated while reading \(what)"
        case .unknownValueType(let t): return "unknown GGUF KV type \(t)"
        case .missingKey(let k): return "GGUF has no key \(k)"
        }
    }
}

public enum GGUFValue {
    case uint(UInt64)
    case int(Int64)
    case double(Double)
    case bool(Bool)
    case string(String)
    case strings([String])
    case numbers([Double])

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }
    public var intValue: Int? {
        switch self {
        case .uint(let v): return Int(bitPattern: UInt(v))
        case .int(let v): return Int(v)
        case .double(let v): return Int(v)
        case .bool(let v): return v ? 1 : 0
        default: return nil
        }
    }
    public var doubleValue: Double? {
        switch self {
        case .uint(let v): return Double(v)
        case .int(let v): return Double(v)
        case .double(let v): return v
        default: return nil
        }
    }
    public var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }
    public var stringsValue: [String]? {
        if case .strings(let s) = self { return s }
        return nil
    }
    public var numbersValue: [Double]? {
        if case .numbers(let n) = self { return n }
        return nil
    }
}

public struct GGUFTensorInfo {
    public let name: String
    public let typeID: UInt32
    /// ggml order: ne[0] is innermost.
    public let ne: [UInt64]
    public let offset: UInt64
}

public final class GGUFFile {
    public let path: String
    public private(set) var kv: [String: GGUFValue] = [:]
    public private(set) var tensors: [String: GGUFTensorInfo] = [:]
    public private(set) var alignment: UInt64 = 32
    public private(set) var dataOffset: UInt64 = 0

    private static let magic: [UInt8] = [0x47, 0x47, 0x55, 0x46]  // "GGUF"

    /// `readArrays: false` skips vocab-sized arrays, which is much faster when
    /// only the hyperparameters are wanted.
    public init(path: String, readArrays: Bool = true) throws {
        self.path = path
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path),
                                   options: .mappedIfSafe) else {
            throw GGUFError.notGGUF(path)
        }
        var cursor = Cursor(data: data)

        guard let head = cursor.bytes(4), Array(head) == Self.magic else {
            throw GGUFError.notGGUF(path)
        }
        let version: UInt32 = try cursor.read()
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(version)
        }
        let tensorCount: UInt64 = try cursor.read()
        let kvCount: UInt64 = try cursor.read()

        for _ in 0..<kvCount {
            let key = try cursor.string()
            let type: UInt32 = try cursor.read()
            kv[key] = try cursor.value(type: type, readArrays: readArrays)
        }
        if let a = kv["general.alignment"]?.intValue, a > 0 {
            alignment = UInt64(a)
        }

        for _ in 0..<tensorCount {
            let name = try cursor.string()
            let dims: UInt32 = try cursor.read()
            var ne = [UInt64]()
            for _ in 0..<dims { ne.append(try cursor.read()) }
            let typeID: UInt32 = try cursor.read()
            let offset: UInt64 = try cursor.read()
            tensors[name] = GGUFTensorInfo(name: name, typeID: typeID,
                                           ne: ne, offset: offset)
        }
        let pos = UInt64(cursor.position)
        dataOffset = (pos + alignment - 1) / alignment * alignment
    }

    // MARK: - convenience

    public var architecture: String { kv["general.architecture"]?.stringValue ?? "" }
    public var chatTemplate: String? { kv["tokenizer.chat_template"]?.stringValue }

    public func string(_ key: String) throws -> String {
        guard let v = kv[key]?.stringValue else { throw GGUFError.missingKey(key) }
        return v
    }
    public func int(_ key: String) throws -> Int {
        guard let v = kv[key]?.intValue else { throw GGUFError.missingKey(key) }
        return v
    }
    public func optionalInt(_ key: String) -> Int? { kv[key]?.intValue }

    /// Architecture-prefixed lookup: `arch("block_count")` reads
    /// `qwen35.block_count`.
    public func arch(_ suffix: String) -> GGUFValue? {
        kv["\(architecture).\(suffix)"]
    }
}

// MARK: - byte cursor

private struct Cursor {
    let data: Data
    var position: Int = 0

    mutating func bytes(_ count: Int) -> Data? {
        guard position + count <= data.count else { return nil }
        defer { position += count }
        return data.subdata(in: position..<(position + count))
    }

    mutating func read<T>() throws -> T {
        let size = MemoryLayout<T>.size
        guard let raw = bytes(size) else {
            throw GGUFError.truncated("\(T.self)")
        }
        return raw.withUnsafeBytes { $0.loadUnaligned(as: T.self) }
    }

    mutating func skip(_ count: Int) throws {
        guard position + count <= data.count else {
            throw GGUFError.truncated("skip \(count)")
        }
        position += count
    }

    mutating func string() throws -> String {
        let length: UInt64 = try read()
        guard let raw = bytes(Int(length)) else {
            throw GGUFError.truncated("string of \(length) bytes")
        }
        // Vocabulary entries are not all valid UTF-8 on their own -- byte-level
        // BPE stores single bytes mapped into a private range -- so decoding
        // must not be allowed to fail.
        return String(decoding: raw, as: UTF8.self)
    }

    mutating func value(type: UInt32, readArrays: Bool) throws -> GGUFValue {
        switch type {
        case 0: return .uint(UInt64(try read() as UInt8))
        case 1: return .int(Int64(try read() as Int8))
        case 2: return .uint(UInt64(try read() as UInt16))
        case 3: return .int(Int64(try read() as Int16))
        case 4: return .uint(UInt64(try read() as UInt32))
        case 5: return .int(Int64(try read() as Int32))
        case 6: return .double(Double(try read() as Float))
        case 7: return .bool((try read() as UInt8) != 0)
        case 8: return .string(try string())
        case 10: return .uint(try read() as UInt64)
        case 11: return .int(try read() as Int64)
        case 12: return .double(try read() as Double)
        case 9:
            let elementType: UInt32 = try read()
            let count: UInt64 = try read()
            if elementType == 8 {
                if !readArrays && count > 512 {
                    for _ in 0..<count {
                        let n: UInt64 = try read()
                        try skip(Int(n))
                    }
                    return .strings([])
                }
                var out = [String]()
                out.reserveCapacity(Int(count))
                for _ in 0..<count { out.append(try string()) }
                return .strings(out)
            }
            let width = Self.scalarWidth(elementType)
            guard width > 0 else { throw GGUFError.unknownValueType(elementType) }
            if !readArrays && count > 4096 {
                try skip(width * Int(count))
                return .numbers([])
            }
            var out = [Double]()
            out.reserveCapacity(Int(count))
            for _ in 0..<count {
                out.append(try scalar(type: elementType))
            }
            return .numbers(out)
        default:
            throw GGUFError.unknownValueType(type)
        }
    }

    static func scalarWidth(_ type: UInt32) -> Int {
        switch type {
        case 0, 1, 7: return 1
        case 2, 3: return 2
        case 4, 5, 6: return 4
        case 10, 11, 12: return 8
        default: return 0
        }
    }

    mutating func scalar(type: UInt32) throws -> Double {
        switch type {
        case 0: return Double(try read() as UInt8)
        case 1: return Double(try read() as Int8)
        case 2: return Double(try read() as UInt16)
        case 3: return Double(try read() as Int16)
        case 4: return Double(try read() as UInt32)
        case 5: return Double(try read() as Int32)
        case 6: return Double(try read() as Float)
        case 7: return (try read() as UInt8) != 0 ? 1 : 0
        case 10: return Double(try read() as UInt64)
        case 11: return Double(try read() as Int64)
        case 12: return try read() as Double
        default: throw GGUFError.unknownValueType(type)
        }
    }
}
