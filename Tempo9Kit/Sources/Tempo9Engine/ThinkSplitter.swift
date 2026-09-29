// Copyright (c) 2026 Jiejing Zhang.
// Portions derived from DashInfer/AllSpark (pyhie/serving/parsers/think_parser.py),
// Copyright (c) Alibaba, Inc. and its affiliates, Apache-2.0. See NOTICE.
//
// Separating the model's reasoning from its answer, as it streams.
//
// A port of the server's ThinkTagParser (pyhie/serving/parsers/think_parser.py)
// so the in-process path shows the same thing the HTTP path does. Both sides
// have to agree or the same model appears to behave differently depending on
// how it was reached, which is the kind of difference that gets blamed on the
// engine.
//
// The subtle part is partial tags. Tokens do not respect tag boundaries: a
// chunk can end in "<thi", and emitting that verbatim puts stray markup on
// screen for one frame. So any tail that could still become a tag is held
// back until the next chunk decides it.

import Foundation

public struct ThinkSplit: Sendable {
    public var content = ""
    public var reasoning = ""
    public var isEmpty: Bool { content.isEmpty && reasoning.isEmpty }
}

/// What LocalSession streams through: ThinkSplitter for tag-pair dialects,
/// HarmonySplitter for GPT-OSS. Same contract either way -- feed returns
/// what is SETTLED, flush returns what was held back at end of stream.
public protocol StreamSplitter: AnyObject {
    func feed(_ text: String) -> ThinkSplit
    func flush() -> ThinkSplit
}

public final class ThinkSplitter: StreamSplitter {
    private enum State { case initial, thinking, content }

    private let openTag: String
    private let closeTag: String
    private var state: State
    private var buffer = ""

    /// `forceReasoning` is for models that begin reasoning without ever
    /// emitting an open tag (DeepSeek-R1 does this). Qwen3.5 emits the tag,
    /// so the default is false.
    public init(openTag: String = "<think>",
                closeTag: String = "</think>",
                forceReasoning: Bool = false) {
        self.openTag = openTag
        self.closeTag = closeTag
        self.state = forceReasoning ? .thinking : .initial
    }

    public func feed(_ text: String) -> ThinkSplit {
        buffer += text
        var out = ThinkSplit()

        while !buffer.isEmpty {
            switch state {
            case .initial:
                // A CLOSE before any open means the block was already open
                // when generation started -- which is the normal shape for a
                // template that appends a pre-closed thought and a model that
                // keeps writing in it anyway. Everything up to the close is
                // reasoning; without this it flowed into the answer, and a
                // bare "<channel|>" appeared in front of every reply.
                if let c = buffer.range(of: closeTag),
                   buffer.range(of: openTag).map({ c.lowerBound < $0.lowerBound })
                       ?? true {
                    out.reasoning += String(buffer[buffer.startIndex..<c.lowerBound])
                    buffer = String(buffer[c.upperBound...])
                    state = .content
                } else if let r = buffer.range(of: openTag) {
                    // Anything before the tag is ordinary content.
                    out.content += String(buffer[buffer.startIndex..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    state = .thinking
                } else if let keep = partialTagSuffixLength(of: buffer,
                                                           tag: openTag) {
                    let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
                    out.content += String(buffer[buffer.startIndex..<cut])
                    buffer = String(buffer[cut...])
                    return out          // wait for the rest of the tag
                } else {
                    out.content += buffer
                    buffer = ""
                }

            case .thinking:
                if let r = buffer.range(of: closeTag) {
                    out.reasoning += String(buffer[buffer.startIndex..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    state = .content
                } else if let keep = partialTagSuffixLength(of: buffer,
                                                           tag: closeTag) {
                    let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
                    out.reasoning += String(buffer[buffer.startIndex..<cut])
                    buffer = String(buffer[cut...])
                    return out
                } else {
                    out.reasoning += buffer
                    buffer = ""
                }

            case .content:
                // A stray close in the middle of an answer: the model opened
                // a thought with a marker this dialect did not match, or
                // repeated the close. Dropping it is right either way — it is
                // a control marker, never something the reader asked for.
                if let c = buffer.range(of: closeTag),
                   buffer.range(of: openTag).map({ c.lowerBound < $0.lowerBound })
                       ?? true {
                    out.content += String(buffer[buffer.startIndex..<c.lowerBound])
                    buffer = String(buffer[c.upperBound...])
                    continue
                }
                // A second <think> block is legal; the model may alternate.
                if let r = buffer.range(of: openTag) {
                    out.content += String(buffer[buffer.startIndex..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    state = .thinking
                } else if let keep = partialTagSuffixLength(of: buffer,
                                                           tag: openTag) {
                    let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
                    out.content += String(buffer[buffer.startIndex..<cut])
                    buffer = String(buffer[cut...])
                    return out
                } else {
                    out.content += buffer
                    buffer = ""
                }
            }
        }
        return out
    }

    /// Anything still held back at end of stream.
    ///
    /// A tail that looked like the start of a tag and never became one is
    /// real text, so it is emitted rather than dropped — silently eating a
    /// trailing "<" would be worse than showing it.
    public func flush() -> ThinkSplit {
        defer { buffer = ""; }
        guard !buffer.isEmpty else { return ThinkSplit() }
        var out = ThinkSplit()
        if state == .thinking { out.reasoning = buffer } else { out.content = buffer }
        return out
    }

    /// Length of the trailing run of `s` that is a proper prefix of `tag`,
    /// or nil when the tail cannot become one.
    private func partialTagSuffixLength(of s: String, tag: String) -> Int? {
        let maxLen = min(s.count, tag.count - 1)
        guard maxLen > 0 else { return nil }
        for len in stride(from: maxLen, through: 1, by: -1) {
            let tail = String(s.suffix(len))
            if tag.hasPrefix(tail) { return len }
        }
        return nil
    }
}
