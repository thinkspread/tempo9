// Copyright (c) 2026 Jiejing Zhang.
// `vitctl --mel <wav> <out.bin>`: dump the Swift mel for cross-checking
// against the Python reference. A front end that compiles is not a front
// end that matches.
import CoreGraphics
import Foundation
import ImageIO
import VisionTowerKit

enum MelProbe {
    static func run(wav: String, out: String) {
        let pcm = readWav(wav)
        let (mel, frames) = OmniMel.logMel(pcm)
        print("frames \(frames) mel \(mel.count)")
        mel.withUnsafeBufferPointer {
            FileManager.default.createFile(atPath: out, contents: Data(buffer: $0))
        }
    }

    /// Read the wav, run the Core ML tower, dump the embeddings.
    static func tower(dir: String, wav: String, out: String) {
        do {
            let t = try OmniAudioTower(directory: URL(fileURLWithPath: dir))
            let pcm = readWav(wav)
            let e = try t.encode(pcm)
            print(String(format: "%d tokens (%.1fs audio) in %.0f ms",
                         e.tokens, e.duration, e.seconds * 1000))
            e.embedding.withUnsafeBufferPointer {
                FileManager.default.createFile(atPath: out,
                                               contents: Data(buffer: $0))
            }
        } catch {
            FileHandle.standardError.write(Data("tower failed: \(error)\n".utf8))
            exit(1)
        }
    }

    static func readWav(_ path: String) -> [Float] {
        guard let d = FileManager.default.contents(atPath: path) else {
            FileHandle.standardError.write(Data("cannot read \(path)\n".utf8))
            exit(1)
        }
        let bytes = [UInt8](d)
        var i = 12
        var pcm: [Float] = []
        while i + 8 <= bytes.count {
            let id = String(bytes: bytes[i..<i+4], encoding: .ascii) ?? ""
            let raw = UInt32(bytes[i+4]) | UInt32(bytes[i+5]) << 8
                | UInt32(bytes[i+6]) << 16 | UInt32(bytes[i+7]) << 24
            let size = min(Int(raw), bytes.count - i - 8)
            if id == "data" {
                var j = i + 8
                let end = min(j + max(0, size), bytes.count)
                while j + 1 < end {
                    let v = Int16(bitPattern: UInt16(bytes[j]) | UInt16(bytes[j+1]) << 8)
                    pcm.append(Float(v) / 32768.0)
                    j += 2
                }
                break
            }
            i += 8 + size + (size & 1)
        }
        return pcm
    }

    static func vision(dir: String, image: String, out: String) {
        do {
            let t = try OmniVisionTower(directory: URL(fileURLWithPath: dir))
            guard let src = CGImageSourceCreateWithURL(
                      URL(fileURLWithPath: image) as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                FileHandle.standardError.write(Data("cannot read \(image)\n".utf8))
                exit(1)
            }
            let e = try t.encode(image: img)
            print(String(format: "%d tokens, grid %dx%d, bucket %d, %.0f ms",
                         e.tokens, e.gridH, e.gridW, e.bucketN,
                         e.towerSeconds * 1000))
            e.embedding.withUnsafeBufferPointer {
                FileManager.default.createFile(atPath: out,
                                               contents: Data(buffer: $0))
            }
        } catch {
            FileHandle.standardError.write(Data("vision failed: \(error)\n".utf8))
            exit(1)
        }
    }
}
