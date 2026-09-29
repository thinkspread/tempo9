// Copyright (c) 2026 Jiejing Zhang.
//
// One prompt through the in-process pipeline, so it can be compared against
// the same prompt through the HTTP server. Until the two agree the app keeps
// using HTTP: a second inference path that has not been checked against the
// first is a way to introduce differences and then blame the engine.
//
//   localctl <model.asgraph> <model.gguf> "<question>" [--think]
//                                          [--image <file> --tower <dir>
//                                           --image-token <id>]

import Foundation
import CoreGraphics
import ImageIO
import Tempo9
import Tempo9Engine
import VisionTowerKit

let args = CommandLine.arguments
guard args.count >= 4 else {
    FileHandle.standardError.write(Data(
        "usage: localctl <asgraph> <gguf> <question> [--think]\n".utf8))
    exit(2)
}
let think = args.contains("--think")

/// Arguments as JSON on one line.
///
/// The first version printed `name(k=v, k=v)`, which is ambiguous the moment
/// a value contains ", " — and BFCL is full of addresses like
/// "2020 Addison Street, Berkeley, CA, USA". The scorer split on the comma
/// and reported a perfectly good call as no call at all. Function names
/// also carry dots (`uber.ride`), so the name is printed unquoted and
/// unparsed up to the first space.
func jsonArgs(_ args: [String: String]) -> String {
    let data = try? JSONSerialization.data(
        withJSONObject: args, options: [.sortedKeys])
    return data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
}

func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
let imagePath = flag("--image")
/// gpu | gpuOnly | ane | cpu -- see makeTower.
let tierFlag = flag("--tier")
/// Several frames, cycled one per run — the app's actual pattern.
///
/// A repeat with one image measures the easy case (nothing changed). What
/// matters is whether a system prompt still gets reused when the picture
/// after it does not, and that cannot be measured across processes: the
/// cache lives in the engine, and each invocation loads a fresh one.
let imagePaths: [String] = flag("--images")?
    .split(separator: ",").map(String.init) ?? []
let towerDir = flag("--tower")
let imageToken = flag("--image-token").flatMap(Int.init)
/// 16 kHz mono .wav. The point of testing it here is the TWO-BLOCK path:
/// an image and an utterance in one request, which is what holding the key
/// with the camera on actually sends.
let audioPath = flag("--audio")
/// Previous turns, replayed exactly the way the app replays them:
/// [{"question": "...", "answer": "..."}]. The app's own history is two
/// turns of plain text with no images, and "does it remember" cannot be
/// answered by a tool that only ever sends one turn.
let historyPath = flag("--history")

/// Read a 16-bit PCM .wav as float in [-1, 1]. Deliberately minimal: this is
/// a test entry point, and a real decoder would be a dependency to carry for
/// one flag. It refuses anything that is not what it expects rather than
/// reinterpreting it — audio at the wrong rate does not fail, it just makes
/// every token cover the wrong span of time.
func loadWav16k(_ path: String) throws -> [Float] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    func u32(_ o: Int) -> UInt32 {
        d.subdata(in: o..<(o + 4)).withUnsafeBytes {
            $0.loadUnaligned(as: UInt32.self)
        }
    }
    func u16(_ o: Int) -> UInt16 {
        d.subdata(in: o..<(o + 2)).withUnsafeBytes {
            $0.loadUnaligned(as: UInt16.self)
        }
    }
    guard d.count > 44, u32(0) == 0x46464952 else {   // "RIFF"
        throw NSError(domain: "localctl", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "\(path) is not a RIFF wav"])
    }
    var o = 12, rate: UInt32 = 0, channels: UInt16 = 0, bits: UInt16 = 0
    var samples = [Float]()
    while o + 8 <= d.count {
        let id = u32(o), size = Int(u32(o + 4))
        let body = o + 8
        if id == 0x20746d66 {                          // "fmt "
            channels = u16(body + 2); rate = u32(body + 4)
            bits = u16(body + 14)
        } else if id == 0x61746164 {                   // "data"
            let end = min(body + size, d.count)
            var i = body
            while i + 1 < end {
                samples.append(Float(Int16(bitPattern: u16(i))) / 32768)
                i += 2
            }
        }
        o = body + size + (size & 1)
    }
    guard rate == 16000, channels == 1, bits == 16 else {
        throw NSError(domain: "localctl", code: 3, userInfo: [
            NSLocalizedDescriptionKey:
                "need 16 kHz mono 16-bit, got \(rate) Hz \(channels)ch "
                + "\(bits)-bit"])
    }
    return samples
}

/// The tower runs here too, so this is the whole VLM path in one process:
/// pixels -> Core ML -> embeddings -> placeholder expansion -> M-RoPE ->
/// engine. Nothing in it is Python.
func encodeImage(_ path: String? = nil) throws -> Encoding? {
    guard let imagePath = path ?? imagePath, let towerDir else { return nil }
    guard let src = CGImageSourceCreateWithURL(
              URL(fileURLWithPath: imagePath) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw NSError(domain: "localctl", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "cannot read image \(imagePath)"])
    }
    // A .gguf as --tower is Gemma 4's projector, not a Core ML directory:
    // that family has no export to point at, its whole front end is eleven
    // tensors in the mmproj.
    let towerURL = URL(fileURLWithPath: towerDir)
    let tower = try makeTower(towerURL)
    towerLayout = tower.layout
    return try tower.encode(image: img)
}
/// Set by encodeImage, read when the session is configured. The tower knows
/// the convention; nothing else in this tool does.
var towerLayout: MediaLayout?

/// Which of the three front ends a --tower path names.
///
/// By what is THERE rather than by a flag: a .gguf is Gemma's projector, a
/// directory holding coreml-audio/ is Omni, anything else is the Qwen3.5
/// Core ML export. Asking the caller to also declare it would be one more
/// thing to get wrong about a path they already typed.
func makeTower(_ url: URL,
               cache: EmbeddingCache = EmbeddingCache()) throws -> VisionTower {
    // --tier pins the Core ML compute tier.  The library's default is
    // .auto (GPU while the LLM is idle, ANE while a generation is in
    // flight), which is the right SERVING policy and the wrong
    // MEASUREMENT one: a J/image or latency comparison between tiers
    // needs the tier held still, and .auto would silently answer with
    // whichever one it felt like per call.  An unknown name is fatal
    // rather than a fallback -- a benchmark that quietly measures a
    // different tier than its label is worse than one that refuses.
    var policy: TierPolicy = .auto
    if let t = tierFlag {
        guard let tier = ComputeTier(rawValue: t) else {
            throw NSError(domain: "localctl", code: 2, userInfo: [
                NSLocalizedDescriptionKey:
                    "--tier \(t) unknown; expected one of "
                    + ComputeTier.allCases.map(\.rawValue).joined(separator: ", ")])
        }
        policy = .fixed(tier)
    }
    if url.pathExtension == "gguf" {
        // Gemma 4's projector is Metal, not Core ML: it has no tier to pin.
        if tierFlag != nil {
            FileHandle.standardError.write(Data(
                "[localctl] --tier ignored: a .gguf projector has no Core ML tier\n".utf8))
        }
        return try VisionTower(gemma4Mmproj: url, cache: cache)
    }
    if FileManager.default.fileExists(atPath: url
        .appendingPathComponent("coreml-audio/tower_meta.json").path) {
        return try VisionTower(omniDirectory: url, policy: policy, cache: cache)
    }
    return try VisionTower(directory: url, policy: policy, cache: cache)
}

// --encode-loop N: run ONLY the vision tower, N times, and exit.
//
// For a J/image measurement the LLM must not be in the window at all --
// otherwise every image carries a model load (or a decode) and the number
// stops being the tower's.  The cache is given a zero byte budget so every
// iteration is real work; a warm cache would make iterations 2..N free and
// report the tower as costing nothing.
if let loopN = flag("--encode-loop").flatMap(Int.init) {
    guard let imagePath, let towerDir else {
        FileHandle.standardError.write(Data(
            "--encode-loop needs --image and --tower\n".utf8))
        exit(2)
    }
    guard let src = CGImageSourceCreateWithURL(
              URL(fileURLWithPath: imagePath) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write(Data(
            "cannot read image \(imagePath)\n".utf8))
        exit(2)
    }
    // Cache bypassed: every iteration is real tower work.  A warm cache
    // would make iterations 2..N free and report the tower as costing
    // nothing at all -- and a zero byte budget does NOT achieve this,
    // because eviction keeps the last entry on purpose.
    let bench = EmbeddingCache()
    bench.bypass = true
    let tower = try makeTower(URL(fileURLWithPath: towerDir), cache: bench)
    // One warm-up encode outside the count: the first predict pays shader
    // compilation and, on the ANE, model planning.
    _ = try tower.encode(image: img)
    let t0 = Date()
    var hits = 0
    for _ in 0..<loopN {
        let e = try tower.encode(image: img)
        if e.cacheHit { hits += 1 }
    }
    let dt = Date().timeIntervalSince(t0)
    print("ENCODES \(loopN) tier=\(tower.backendLabel) seconds=\(String(format: "%.3f", dt)) "
          + "per_image_ms=\(String(format: "%.1f", dt / Double(loopN) * 1000)) cache_hits=\(hits)")
    exit(hits == 0 ? 0 : 3)   // a cache hit means the measurement is void
}

do {
    let session = try LocalSession(modelName: "localctl",
                                   graphPath: args[1],
                                   ggufPath: args[2],
                                   // Settable, because the context ceiling
                                   // used to decide first-token latency
                                   // through the attention workspace, and
                                   // "is that still true" cannot be answered
                                   // without building the engine both ways.
                                   maxLength: Int64(
                                       flag("--max-length").flatMap(Int.init)
                                       ?? 2048))
    var cfg = SamplingConfig()
    cfg.maxTokens = flag("--max-tokens").flatMap(Int.init) ?? 64
    // Greedy: this is a comparison, and two sampled runs would differ for
    // reasons that have nothing to do with the pipeline being tested.
    cfg.doSample = false
    cfg.topK = 1
    cfg.temperature = 0
    // MTP draft depth. A *request* field, not a model one -- setting it on
    // the model config changes a default the engine never reads per request,
    // which is how MTP once shipped compiled in, loaded, and never used.
    if let k = flag("--speculation-k").flatMap(Int32.init) { cfg.speculationK = k }
    // Sampling changes what "same answer" means, so a quality comparison
    // stays greedy unless asked otherwise.
    if let t = flag("--temperature").flatMap(Float.init), t > 0 {
        cfg.temperature = t; cfg.doSample = true; cfg.topK = 20; cfg.topP = 0.95
    }

    print("cfg: speculationK=\(cfg.speculationK) doSample=\(cfg.doSample) "
          + "temp=\(cfg.temperature) maxTokens=\(cfg.maxTokens)")
    let encoding = try encodeImage()
    var utterance: AudioPlacement?
    if let audioPath {
        // The tower has to exist for audio too — the projector carries both
        // front ends, and there is nothing else to load it from.
        guard let towerDir else {
            throw NSError(domain: "localctl", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "--audio needs --tower"])
        }
        let pcm = try loadWav16k(audioPath)
        let tower = try makeTower(URL(fileURLWithPath: towerDir))
        towerLayout = tower.layout
        if let enc = try tower.encodeAudio(pcm) {
            print(String(format: "audio: %d tokens (%.1fs) in %.0fms",
                         enc.tokens, enc.duration, enc.seconds * 1e3))
            utterance = AudioPlacement(embedding: enc.embedding,
                                       tokens: enc.tokens,
                                       hidden: enc.outHidden,
                                       contentKey: Int64(pcm.count))
        } else {
            print("audio: this tower cannot hear")
        }
    }
    // Set from whichever front end ran — audio-only is a real case, and
    // hanging the layout off the image branch made it fail with "no
    // image_token_id set" for a request that had no image in it.
    // --image-token still overrides, for probing a model whose id this
    // build does not know; otherwise the tower's own layout wins.
    if let imageToken {
        session.imageTokenID = imageToken
    } else if towerLayout != nil {
        session.mediaLayout = towerLayout
    }
    if let encoding {
        print("tower: \(encoding.tokens) tokens, grid \(encoding.gridH)x"
              + "\(encoding.gridW), \(encoding.tier.rawValue), "
              + String(format: "%.0fms", encoding.towerSeconds * 1e3)
              + ", layout \(session.mediaLayout?.name ?? "none")")
    }
    // The template needs a part per block so it emits each placeholder, in
    // the order the runs will appear.
    var parts: [Any] = []
    if encoding != nil { parts.append(["type": "image"]) }
    if utterance != nil { parts.append(["type": "audio"]) }
    let content: Any = parts.isEmpty
        ? args[3]
        : (parts + [["type": "text", "text": args[3]]]) as [Any]

    // Same shape as LocalBackend.messages: plain user/assistant turns before
    // the current one, no images on the past turns.
    var priorTurns: [[String: Any]] = []
    if let historyPath {
        let d = try Data(contentsOf: URL(fileURLWithPath: historyPath))
        let turns = ((try JSONSerialization.jsonObject(with: d))
                     as? [[String: String]]) ?? []
        for t in turns {
            priorTurns.append(["role": "user", "content": t["question"] ?? ""])
            priorTurns.append(["role": "assistant",
                               "content": t["answer"] ?? ""])
        }
        print("history: \(turns.count) prior turn(s)")
    }

    var placement = encoding.map {
        ImagePlacement(embedding: $0.embedding, tokens: $0.tokens,
                       hidden: $0.outHidden, gridH: $0.gridH,
                       gridW: $0.gridW, contentKey: $0.contentKey,
                       deepstack: $0.deepstack,
                       numDeepstackLayers: $0.numDeepstackLayers)
    }
    // A quality comparison asks several questions of one model. Loading the
    // model is minutes and asking is seconds, so the questions come to the
    // model rather than the model being reloaded per question -- 3 loads
    // instead of 15 for a 5-question, 3-model comparison.
    //
    // Each line is its OWN conversation: no history, so answer N cannot be
    // influenced by answer N-1, and the prefix cache is the only thing
    // shared between them.
    // Tools go through the template, so the model sees them the way it was
    // trained to. --tools takes the same JSON array an OpenAI-style API
    // would carry.
    var toolSpecs: [[String: Any]]? = nil
    if let tp = flag("--tools") {
        let data = try Data(contentsOf: URL(fileURLWithPath: tp))
        toolSpecs = (try JSONSerialization.jsonObject(with: data))
            as? [[String: Any]]
        print("tools: \(toolSpecs?.count ?? 0) 个")
    }

    // One case per line: {"id", "tools":[...], "question": "..."}. Written
    // for BFCL, which is a few hundred cases each with its OWN tool list —
    // so the tools cannot be a single process-wide flag.
    if let path = flag("--cases") {
        let lines = (try String(contentsOfFile: path, encoding: .utf8))
            .split(separator: "\n").map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        await session.warmUp()
        for line in lines {
            guard let d = line.data(using: .utf8),
                  let o = (try? JSONSerialization.jsonObject(with: d))
                          as? [String: Any],
                  let id = o["id"] as? String,
                  let q = o["question"] as? String else { continue }
            print("### CASE \(id)")
            let tools = o["tools"] as? [[String: Any]]
            do {
                let r = try await session.stream(
                    messages: [["role": "user", "content": q]],
                    config: cfg, enableThinking: think, tools: tools,
                    onDelta: { _ in })
                let calls = ToolCallParser.parse(r.reasoning + r.text)
                for c in calls {
                    print("TOOLCALL \(c.name) \(jsonArgs(c.arguments))")
                }
                // A case that yields no call has to show what it DID say.
                // Printing only the parsed calls makes "the model refused"
                // and "the model called it in a dialect this parser does not
                // read" look identical — and the second is the failure that
                // actually happens when a new model family arrives.
                if calls.isEmpty {
                    print("NOCALL raw: \(r.text.prefix(400))")
                }
            } catch {
                print("ERROR \(error)")
            }
        }
        exit(0)
    }

    // A system prompt shared by every question, which is what the app does
    // and what makes the prefix cache worth having: the mode prompt is
    // ~1500 tokens and identical across turns, so it should be reused whole.
    let systemPrompt = flag("--system").map { text -> [String: Any] in
        ["role": "system", "content": text]
    }

    if let path = flag("--ask-file") {
        let lines = (try String(contentsOfFile: path, encoding: .utf8))
            .split(separator: "\n").map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        await session.warmUp()
        for (i, q) in lines.enumerated() {
            let t0 = Date()
            let msgs: [[String: Any]] = (systemPrompt.map { [$0] } ?? [])
                + [["role": "user", "content": q]]
            let r = try await session.stream(
                messages: msgs,
                config: cfg, enableThinking: think, tools: toolSpecs,
                onDelta: { _ in })
            let secs = Date().timeIntervalSince(t0)
            print("### Q\(i + 1): \(q)")
            if !r.reasoning.isEmpty {
                print("[think \(r.reasoning.count) 字]")
            }
            print(r.text)
            // Parsed separately from the text: what matters for a tool
            // question is whether a CALL came out, not whether the raw
            // string happens to contain the right words.
            for c in ToolCallParser.parse(r.reasoning + r.text) {
                print("TOOLCALL \(c.name) \(jsonArgs(c.arguments))")
            }
            print(String(format: "[%d tok, %.1fs, %.1f tok/s]\n",
                         r.completionTokens, secs,
                         Double(r.completionTokens) / max(secs, 1e-6)))
        }
        exit(0)
    }

    // Repeat in ONE process, because the interesting question is which costs
    // are one-time. A cold first request pays for Metal pipeline compilation,
    // the first graph build and an empty prefix cache; a per-process harness
    // charges all of that to every number it prints and cannot tell you that
    // run 2 was four times faster.
    // --warm reproduces what the app now does at load time, so run 1 here
    // measures what the user's first question will actually cost.
    if args.contains("--warm") {
        // Printed because a skipped half is silent: the image warm-up needs
        // an image token and a hidden size, and without them it returns
        // having done nothing, which looks exactly like having done it.
        let t = Date()
        await session.warmUp(onPhase: { print("warm: \($0)") })
        print(String(format: "warm: done in %.1fs", Date().timeIntervalSince(t)))
    }
    let repeats = flag("--repeat").flatMap(Int.init) ?? 1
    // --repeat-delay S: sleep between runs.  A watch workload is paced —
    // one check every N seconds, idle in between — and its power draw is
    // a duty cycle, not a throughput.  Back-to-back repeats measure
    // continuous inference; this measures watching.
    let repeatDelay = flag("--repeat-delay").flatMap(Double.init) ?? 0
    var reply: LocalReply!
    for run in 1...repeats {
        if run > 1 && repeatDelay > 0 {
            try await Task.sleep(for: .seconds(repeatDelay))
        }
        let t0 = Date()
        var firstToken: Double = 0
        if !imagePaths.isEmpty {
            let path = imagePaths[(run - 1) % imagePaths.count]
            placement = try encodeImage(path).map {
                ImagePlacement(embedding: $0.embedding, tokens: $0.tokens,
                               hidden: $0.outHidden, gridH: $0.gridH,
                               gridW: $0.gridW, contentKey: $0.contentKey,
                               deepstack: $0.deepstack,
                               numDeepstackLayers: $0.numDeepstackLayers)
            }
        }
        // --system applies to the single-question path too. It did not,
        // which made every prefix-cache measurement taken with this tool
        // meaningless: the app's ~1500-token mode prompt is the thing that
        // is supposed to be reused, and the tool was sending an 80-token
        // request with no system prompt at all while reporting on caching.
        reply = try await session.stream(
            messages: (systemPrompt.map { [$0] } ?? []) + priorTurns
                + [["role": "user", "content": content]],
            config: cfg, enableThinking: think, image: placement,
            audio: utterance,
            onDelta: { _ in
                if firstToken == 0 { firstToken = Date().timeIntervalSince(t0) }
            })
        let total = Date().timeIntervalSince(t0)
        // Decode rate excludes TTFT on purpose: mixing prefill into tok/s
        // makes a prefill regression look like a decode regression.
        let decode = max(total - firstToken, 1e-6)
        let rate = Double(max(reply.completionTokens - 1, 1)) / decode
        print(String(format: "run %d: TTFT %.0f ms, decode %.1f tok/s "
                     + "(%d prompt, %d completion, %.1fs total)",
                     run, firstToken * 1e3, rate, reply.promptTokens,
                     reply.completionTokens, total))
        // The ENGINE's own split, when it gave one.  The line above is this
        // process's stopwatch and carries tokenisation, templating and
        // scheduling; the line below is the forward.  Printing both, always,
        // is what stops the two being quoted interchangeably -- our
        // published prefill number was the stopwatch while llama-bench and
        // mlx_lm report the forward.
        if let e = reply.engine {
            print(String(format: "        engine: prefill %.0f ms (%lld tok"
                         + ", %lld cached) / decode %.0f ms (%lld tok)"
                         + " -> pp %.0f tok/s, tg %.1f tok/s",
                         e.prefillMs, e.promptTokens, e.prefixCacheHitTokens,
                         e.decodeMs, e.generatedTokens,
                         e.prefillMs > 0
                            ? Double(e.promptTokens - e.prefixCacheHitTokens)
                              / (e.prefillMs / 1000) : 0,
                         e.decodeMs > 0
                            ? Double(max(e.generatedTokens - 1, 1))
                              / (e.decodeMs / 1000) : 0))
        }
    }
    if args.contains("--raw") {
        // Diagnostic: what the splitter was given, before it split. The
        // difference between "the model never closed the tag" and "the
        // splitter lost the transition" is invisible from the outside.
        print("--- raw (reasoning+content, unsplit) ---")
        print(reply.reasoning + reply.text)
        print("--- contains </think>: "
              + String((reply.reasoning + reply.text).contains("</think>")))
    }
    print("--- reasoning ---\n\(reply.reasoning)")
    print("--- answer ---\n\(reply.text)")
    if let st = session.stats() {
        let seen = st.prefixCacheHitTokens + st.prefixCacheMissTokens
        print(String(format: "--- engine: KV %lld/%lld tokens, prefix %.0f%% "
                     + "of %lld tokens, %lld prefill / %lld generated",
                     st.totalTokens - st.freeTokens, st.totalTokens,
                     Double(st.prefixCacheHitRate) * 100, seen,
                     st.prefillTokens, st.generatedTokens))
    }
    print("--- prompt=\(reply.promptTokens) completion=\(reply.completionTokens) "
          + String(format: "%.1fs", reply.seconds))
} catch {
    FileHandle.standardError.write(Data("FAILED: \(error)\n".utf8))
    exit(1)
}
