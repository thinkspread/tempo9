// Copyright (c) 2026 Jiejing Zhang.
//
// The three engine types that Tempo9's OWN public API names, re-exposed so a
// caller can name them too.
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
