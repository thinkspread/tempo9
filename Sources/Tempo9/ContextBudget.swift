// Copyright (c) 2026 Jiejing Zhang.
//
// The engine's max length is a TOTAL budget per request, prompt included
// (te9_request_start hands it prompt + max_tokens, and the engine refuses
// the request when that exceeds --max-length).  What the server does when
// a request does not fit is decided here, as a pure function of three
// numbers, so the decision can be tested without an engine.
//
// The shape of the problem, from running Claude Code 2.1.239: it sends
// max_tokens 32000 on every request, and its baseline prompt -- system
// text plus 29 tool schemas -- is ~32k tokens.  Refusing on prompt +
// max_tokens would refuse every turn on a 32768 engine, including the
// ones that would have fit with room to spare.  So:
//
//   * the PROMPT alone is what is refused on: if it leaves no room for a
//     single new token, nothing can run, and the 400 says so with the
//     numbers and both ways out;
//   * max_tokens is clamped to the room the prompt leaves, and the clamp
//     is reported (one log line, a `warnings` entry in the reply), because
//     a reply cut at 1116 tokens when 32000 were asked for should not
//     look like the model chose to stop.

import Foundation

enum ContextBudget {
    enum Outcome: Equatable {
        /// prompt + max_tokens fits; run as asked.
        case fits
        /// The prompt fits but max_tokens does not; run with `maxTokens`
        /// and tell the caller with `warning`.
        case clamped(maxTokens: Int, warning: String)
        /// The prompt alone leaves no room for a single new token.
        case refused(promptTokens: Int, maxTokens: Int, maxLength: Int)
    }

    /// `maxLength <= 0` means the engine reported no limit: nothing to
    /// enforce here, the engine's own check (if any) still applies.
    static func apply(promptTokens: Int, maxTokens: Int,
                      maxLength: Int) -> Outcome {
        guard maxLength > 0 else { return .fits }
        let room = maxLength - promptTokens
        guard room >= 1 else {
            return .refused(promptTokens: promptTokens, maxTokens: maxTokens,
                            maxLength: maxLength)
        }
        guard maxTokens > room else { return .fits }
        return .clamped(maxTokens: room,
                        warning: clampWarning(promptTokens: promptTokens,
                                              maxTokens: maxTokens,
                                              clampedTo: room,
                                              maxLength: maxLength))
    }

    static func clampWarning(promptTokens: Int, maxTokens: Int,
                             clampedTo: Int, maxLength: Int) -> String {
        "max_tokens \(maxTokens) was clamped to \(clampedTo): the prompt is "
        + "\(promptTokens) tokens and this server's max length is "
        + "\(maxLength) (start tempo9 with a larger --max-length for longer "
        + "replies)"
    }

    static func refusalMessage(promptTokens: Int, maxTokens: Int,
                               maxLength: Int) -> String {
        "prompt is \(promptTokens) tokens and max_tokens is \(maxTokens): "
        + "\(promptTokens + maxTokens) exceeds this server's max length "
        + "\(maxLength), and the prompt alone leaves no room for a reply "
        + "(start tempo9 with a larger --max-length, or shorten the prompt)"
    }
}
