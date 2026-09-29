// Copyright (c) 2026 Jiejing Zhang.
//
// Splitting GPT-OSS "harmony" output into reasoning and answer, as it
// streams.
//
// Harmony is not an open/close tag pair, which is why ThinkSplitter cannot
// be taught it. A turn is a sequence of blocks:
//
//   <|channel|>analysis<|message|>...reasoning...<|end|>
//   <|start|>assistant<|channel|>final<|message|>...answer...<|return|>
//
// The channel NAME between <|channel|> and <|message|> decides where the
// block's body goes, and the text between <|start|> and the next marker is a
// role/recipient header ("assistant", "to=functions.x"), not content. Strip
// the markers without reading the names -- which is what the generic
// special-token stripping did -- and the names concatenate into the answer:
// "analysisThe user asks...assistantfinalParis". That literal string is what
// this file exists to prevent.
//
// The body of a block whose header mentions "final" is the answer;
// everything else (analysis, commentary, tool-call headers) is filed as
// reasoning. Header text itself also goes to reasoning rather than being
// dropped -- same rule as the Gemma 4 channel name: never silently eat
// bytes, put them where nobody reads them.
//
// Same partial-marker discipline as ThinkSplitter: the markers arrive as
// atomic preserved tokens, but ordinary text is free to end in "<|", so any
// tail that could still become a marker is held back until the next chunk
// decides it.

import Foundation

public final class HarmonySplitter: StreamSplitter {
    private enum State {
        case body       // streaming a block's body to `target`
        case header     // between <|channel|>/<|start|> and <|message|>
    }

    static let markers = ["<|channel|>", "<|message|>", "<|start|>",
                          "<|end|>", "<|call|>", "<|return|>"]
    private static let headerOpeners: Set<String> = ["<|channel|>", "<|start|>"]

    private var state: State
    private var toContent = false   // where the current body goes
    private var header = ""         // accumulated header text
    private var buffer = ""

    /// Generation resumes after the template's trailing `<|start|>assistant`,
    /// so the model's first token is normally `<|channel|>`. Starting in
    /// `.body` (target: answer) makes that first marker a clean transition
    /// AND does the right thing for a raw, template-less continuation, which
    /// never emits a marker at all.
    public init() {
        state = .body
        toContent = true
    }

    public func feed(_ text: String) -> ThinkSplit {
        buffer += text
        var out = ThinkSplit()

        while !buffer.isEmpty {
            guard let (range, marker) = earliestMarker(in: buffer) else {
                // No complete marker. Hold back a tail that could still
                // become one; the rest is settled text.
                if let keep = partialMarkerSuffixLength(of: buffer) {
                    let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
                    emit(String(buffer[buffer.startIndex..<cut]), into: &out)
                    buffer = String(buffer[cut...])
                } else {
                    emit(buffer, into: &out)
                    buffer = ""
                }
                return out
            }

            emit(String(buffer[buffer.startIndex..<range.lowerBound]),
                 into: &out)
            buffer = String(buffer[range.upperBound...])

            switch state {
            case .body:
                if Self.headerOpeners.contains(marker) {
                    state = .header
                    header = ""
                }
                // <|end|>/<|call|>/<|return|> in a body: the block just
                // closed; whatever follows opens with its own marker. A
                // stray <|message|> is dropped the same way.
            case .header:
                if marker == "<|message|>" {
                    // The header decides the body's destination. "final" is
                    // the answer channel; analysis, commentary, and tool
                    // recipients are reasoning.
                    toContent = header.contains("final")
                    state = .body
                } else if Self.headerOpeners.contains(marker) {
                    header = ""     // a fresh header opener restarts it
                }
                // <|end|>/<|call|> inside a header is malformed; stay in
                // header and let the next <|message|> resolve it.
            }
        }
        return out
    }

    public func flush() -> ThinkSplit {
        defer { buffer = ""; header = "" }
        var out = ThinkSplit()
        if !buffer.isEmpty { emit(buffer, into: &out) }
        return out
    }

    private func emit(_ text: String, into out: inout ThinkSplit) {
        guard !text.isEmpty else { return }
        if state == .header {
            // Header text ("analysis", "assistant", "to=functions.x") is
            // routing, not content -- but it is also bytes the model wrote,
            // so it goes to reasoning, and to the routing decision.
            header += text
            out.reasoning += text
        } else if toContent {
            out.content += text
        } else {
            out.reasoning += text
        }
    }

    private func earliestMarker(in s: String)
        -> (Range<String.Index>, String)? {
        var best: (Range<String.Index>, String)?
        for m in Self.markers {
            if let r = s.range(of: m),
               best.map({ r.lowerBound < $0.0.lowerBound }) ?? true {
                best = (r, m)
            }
        }
        return best
    }

    /// Longest tail of `s` that is a proper prefix of ANY marker.
    private func partialMarkerSuffixLength(of s: String) -> Int? {
        let maxLen = min(s.count,
                         (Self.markers.map(\.count).max() ?? 1) - 1)
        guard maxLen > 0 else { return nil }
        for len in stride(from: maxLen, through: 1, by: -1) {
            let tail = String(s.suffix(len))
            if Self.markers.contains(where: { $0.hasPrefix(tail) }) {
                return len
            }
        }
        return nil
    }
}
