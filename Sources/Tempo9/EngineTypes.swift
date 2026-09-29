// Copyright (c) 2026 Jiejing Zhang.
//
// The three engine types that Tempo9's OWN public API names, re-exposed so a
// caller can name them too -- and, at the bottom, the two facts about the
// linked engine that an app shows its user.
//
// Without this, three public methods are uncallable from outside the package:
// `stream(config:)` takes a SamplingConfig, `stream(images:)` takes an
// ImageEmbeddings, and `stats()` returns an EngineStats -- all defined in the
// engine module, which is not a product. A consumer that writes `import
// Tempo9` gets "cannot find 'SamplingConfig' in scope" and cannot construct
// the argument. Verified with a two-file consumer package before and after.
//
// Aliases rather than `@_exported import Tempo9Engine`: the promise should be
// exactly these three names -- the ones already in the public signatures --
// not every symbol in the engine module. That is the distinction Package.swift
// argues for, and a blanket re-export would give it up to save two lines.
//
// Worth knowing while reading that argument: the boundary is not currently
// enforced either way. SPM makes a transitively-built module importable, so an
// external package can write `import Tempo9Engine` today and reach everything
// (measured). What the boundary actually stops right now is the caller who
// stays on the public API -- which is backwards. Closing it for real means
// narrowing the engine module's own symbols and forwarding explicitly; these
// aliases only fix the caller who is holding it correctly.

import Tempo9Engine

public typealias SamplingConfig = Tempo9Engine.SamplingConfig
public typealias EngineStats = Tempo9Engine.EngineStats
public typealias ImageEmbeddings = Tempo9Engine.ImageEmbeddings

/// Which engine build is linked, and which GEMM path it is running.
///
/// Forwarded one property at a time, for the reason above: an About panel
/// and a bug report need these two strings, not the engine module.
public enum EngineInfo {
    /// Engine build string, for an About panel and bug reports.
    public static var version: String { Engine.version }

    /// "metal" or "cpu": the GEMM path actually running, as the engine
    /// reports it. A silent CPU fallback is several times slower and
    /// otherwise invisible, which is why an app should show this rather
    /// than a label it assumes.
    ///
    /// Asking initialises the engine's Metal context, so this selects the
    /// backend first (LocalSession.prepareEnvironment, idempotent): a badge
    /// that asked before the environment was set once caused the very CPU
    /// fallback it existed to reveal.
    public static var gemmBackend: String {
        LocalSession.prepareEnvironment()
        return Engine.gemmBackend
    }
}
