// Copyright (c) 2026 Jiejing Zhang.
//
// How a model family splices a picture into a prompt.
//
// Two families ship, and they disagree about nearly everything except that
// the chat template emits ONE placeholder token which somebody has to expand:
//
//                       Qwen3.5                 Gemma 4
//   placeholder id      248056                  258880
//   wrapping            none                    255999 … 258882
//   positions           M-RoPE, [3, seq]        ordinary sequential
//   2-D structure from  the M-RoPE table        intra-block bidirectional
//                                               attention (the graph's
//                                               `bidi_token_ids` attribute)
//
// None of these disagreements fails loudly if you get one wrong. A wrong
// placeholder id splices the embedding somewhere nothing reads and the model
// answers from the text alone; a missing M-RoPE table gives the image tokens
// text positions; missing wrap tokens put the picture somewhere the model was
// never trained to look for one. The symptom in every case is a fluent answer
// about an image nobody encoded — which is why this is one value with all of
// it in one place, rather than four constants read from four files.

import Foundation

public struct MediaLayout: Sendable, Equatable {
    /// The token the chat template emits, and the KEY the engine stores the
    /// embedding under (`MultiMediaInfo` keys on `str(image_token_id)`).
    public var imageTokenID: Int
    /// Emitted once before the expanded run, and once after. Gemma 4 was
    /// trained with them; Qwen3.5 has no such pair.
    public var openTokenID: Int?
    public var closeTokenID: Int?
    /// How this family positions its media tokens -- and therefore what
    /// table, if any, the caller has to build.
    ///
    /// It was a Bool while two families shipped. Three do now, and the third
    /// is not "some M-RoPE": Qwen2.5-Omni's audio is one-dimensional in
    /// time, so its three axes advance together, while Qwen3.5's image axes
    /// come from the patch grid. Same table shape, different contents, and
    /// building one where the other belongs is silent.
    ///
    /// What is NOT here: interleaved versus contiguous-section M-RoPE. That
    /// is a property of the model's exported graph, the engine reads it from
    /// there, and a request has no business restating it.
    public enum PositionScheme: Sendable, Equatable {
        /// No table. Ordinary sequential positions, and the media block gets
        /// its structure some other way -- Gemma 4 uses intra-block
        /// bidirectional attention.
        case sequential
        /// t/h/w from the image's patch grid (Qwen3.5).
        case mropeImageGrid
        /// All three axes are the sequence index: one-dimensional content in
        /// a three-axis table (Qwen2.5-Omni audio).
        case mropeTimeline
    }
    public var positions: PositionScheme

    /// Does a request in this layout carry a position table at all? The one
    /// bit the C ABI needs; everything else about M-RoPE lives in the graph.
    public var usesMRoPE: Bool { positions != .sequential }
    /// The audio soft token, and its wrapping, when the family can hear.
    ///
    /// Deliberately absent from the graph's `bidi_token_ids`: an image block
    /// attends bidirectionally inside itself, audio stays CAUSAL. Sound
    /// arrives in time order and the model was trained to read it that way.
    public var audioTokenID: Int?
    public var audioOpenTokenID: Int?
    public var audioCloseTokenID: Int?
    public var hearsAudio: Bool { audioTokenID != nil }

    /// Does this family need a Core ML tower exported and compiled?
    ///
    /// Qwen3.5's tower is a real ViT: 1.7 GB of .mlpackage buckets, exported
    /// from the checkpoint. Gemma 4 is encoder-free — its whole visual front
    /// end is eleven tensors in an mmproj .gguf beside the weights, and there
    /// is no export, no bucket and no compile.
    ///
    /// Anything that CHECKS for a tower has to ask this first. The app's
    /// preflight blocked Gemma with "Vision tower has no compiled buckets",
    /// which was true and irrelevant: a model that was completely ready was
    /// declared not ready, with a fix-it command pointing at a Qwen export
    /// script.
    public var usesCoreMLTower: Bool
    /// For diagnostics only — so a log line can say which convention ran.
    public var name: String

    public init(imageTokenID: Int, openTokenID: Int?, closeTokenID: Int?,
                positions: PositionScheme, usesCoreMLTower: Bool = true,
                audioTokenID: Int? = nil, audioOpenTokenID: Int? = nil,
                audioCloseTokenID: Int? = nil,
                name: String) {
        self.imageTokenID = imageTokenID
        self.openTokenID = openTokenID
        self.closeTokenID = closeTokenID
        self.positions = positions
        self.usesCoreMLTower = usesCoreMLTower
        self.audioTokenID = audioTokenID
        self.audioOpenTokenID = audioOpenTokenID
        self.audioCloseTokenID = audioCloseTokenID
        self.name = name
    }

    public static let qwen35 = MediaLayout(
        imageTokenID: 248056, openTokenID: nil, closeTokenID: nil,
        positions: .mropeImageGrid, usesCoreMLTower: true,
        name: "qwen3.5")

    /// Qwen3-VL: the qwen2 vocab's vision triple --
    /// `<|vision_start|>`(151652) `<|image_pad|>`(151655) x N
    /// `<|vision_end|>`(151653) -- with the same interleaved-mrope grid
    /// positions the LLM side shares with Qwen3.5.
    public static let qwen3vl = MediaLayout(
        imageTokenID: 151655, openTokenID: 151652, closeTokenID: 151653,
        positions: .mropeImageGrid, usesCoreMLTower: true,
        name: "qwen3vl")

    /// `<|image>`(255999) `<|image|>`(258880) × N `<image|>`(258882).
    ///
    /// 258880 is also one of the two ids in the graph's `bidi_token_ids`
    /// (the other is 258884, video) — the engine finds the media block by
    /// scanning the input ids for a contiguous run of them, so getting this
    /// id right is what turns the bidirectional attention on at all. Audio
    /// (258881) is deliberately NOT in that list: audio stays causal.
    public static let gemma4 = MediaLayout(
        imageTokenID: 258880, openTokenID: 255999, closeTokenID: 258882,
        positions: .sequential, usesCoreMLTower: false,
        // `<|audio>`(256000) `<|audio|>`(258881) x N `<audio|>`(258883)
        audioTokenID: 258881, audioOpenTokenID: 256000,
        audioCloseTokenID: 258883,
        name: "gemma4")

    /// Qwen2.5-Omni: one model for text, images and audio.
    ///
    /// `<|audio_bos|>`(151647) `<|AUDIO|>`(151646) x N `<|audio_eos|>`(151648)
    /// and the vision pair alongside. Its Thinker is a Qwen2-VL, so the
    /// positions are M-RoPE -- but audio is one-dimensional, so the three
    /// axes advance together rather than tracking a grid.
    public static let qwen25omni = MediaLayout(
        imageTokenID: 151655, openTokenID: 151652, closeTokenID: 151653,
        positions: .mropeTimeline, usesCoreMLTower: true,
        audioTokenID: 151646, audioOpenTokenID: 151647,
        audioCloseTokenID: 151648,
        name: "qwen2.5-omni")

    /// Which convention a .gguf follows, by file name.
    ///
    /// By name rather than by metadata because the app already selects models
    /// by file name everywhere else (graphs are bundled under the basename),
    /// and a name that matches neither is a model this build cannot serve
    /// images for — better nil than a guess that decodes as noise.
    public static func forModelFile(_ name: String) -> MediaLayout? {
        let n = name.lowercased()
        if n.contains("omni") { return .qwen25omni }
        if n.contains("qwen3vl") { return .qwen3vl }
        if n.contains("gemma4") || n.contains("gemma-4") { return .gemma4 }
        if n.contains("qwen35") || n.contains("qwen3.5") { return .qwen35 }
        return nil
    }
}
