# Why a real agent framework crashed this engine on turn two

Copyright (c) 2026 Jiejing Zhang.

`curl` was green for months. The first request from OpenClaw was green.
The second killed the process. This is what it was, and the two wrong
answers that came first — both of which looked convincing.

## The symptom

    SIGBUS / EXC_BAD_ACCESS, KERN_PROTECTION_FAILURE
    "Could not determine thread index for stack guard region"
    thread: com.apple.root.user-initiated-qos.cooperative

A stack overflow, on Swift's cooperative pool — whose threads have a
fraction of the main thread's stack, and which is where every request is
served.

## The cause

`BPETokenizer.encodeSegments` recursed on the remainder after each
special token:

```swift
encodeSegments(left, into: &out)          // left has no specials by construction
out.append(hit.id)
encodeSegments(text[after...], into: &out)  // ← one frame per MARKER
```

One stack frame per marker in the text, each holding a fresh `String`
copy of the rest, each re-scanning that rest against every special.

**Measured threshold** (probe on the cooperative pool, Qwen3.5-35B
tokenizer, `<|im_start|>` repeated):

| build | markers |
|---|---|
| recursive | **2000 ok / 2100 SIGBUS** |
| iterative | 10 000 ok, no limit — a loop adds no frames |

2000 is not a comfortable margin. It is roughly what one OpenClaw prompt
carrying 34 tool schemas renders to. Hence: turn one survives, turn two
adds history and tool results, and the process dies.

## Two wrong answers, recorded because they were convincing

**"Memory pressure — the OS reclaimed it."** There *was* a real memory
bug next door (below), and fixing it took the resident set from 15.4 GB
to 6.1 GB. **It still crashed.** The user's own memory panel from the
incident settled it: pressure green, 11.41 of 24 GB used, 902 MB swap.
The machine was not short of memory. A clean exit with no crash report
had been read as a jetsam kill; the report existed, under a filename the
first glob missed.

**"The prompt is too large."** A 200 000-character prompt with no tools
survives. Size alone does nothing — it takes *markers*, and tool schemas
are what multiply them.

## The memory bug found on the way, which is real but separate

`LocalEngineHost` deduplicated model loading with a comment claiming the
actor serialised it. Actors serialise only between suspension points:

```swift
let fresh = try LocalSession(...)   // synchronous, protected
await fresh.warmUp(...)             // ← suspends; actor re-enters here
session = fresh                     // only now is it non-nil
```

A second caller arriving during warm-up sees `session == nil` and builds
another engine. The log said so and was read as noise: two
`engine load: start`, a second `loading the model` *while the first was
warming*, two `ready` lines 160 ms apart. Fixed by holding the in-flight
`Task` and having later callers await it. 15.4 GB → 6.1 GB.

## What this says about the OpenAI-compatible layer

It had never been driven by a real agent harness. Simple clients — one
short prompt, no tools — exercise none of this. The integration itself
cost one config file and no code:

```json5
models: { providers: { tempo9: {
  baseUrl: "http://127.0.0.1:11435/v1",
  apiKey: "local",
  models: [ { id: "…", name: "…" } ] } } }
```

Everything that broke was behind that, and only the second turn reached
it.

## Reproducing

The captured request body kills an unfixed build 100 % of the time. Keep
one: it is worth more than any synthetic test, because the shape that
matters — many markers, many tools, real history — is hard to guess and
trivial to record.

The depth probe must run its tokenizer call inside a `Task` — on the
main thread the 8 MB stack hides the bug entirely.
