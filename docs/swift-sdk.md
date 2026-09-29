# The Swift SDK

Swift packages for running Tempo9 on
Apple platforms — enough to take a `.gguf` and an image or a sound and get
tokens back, with no Python server in the middle.

The engine is C++. This is its Swift side, and the Apple Silicon
half of it — Metal, the Neural Engine towers, and the host.

## The products are separable on purpose

| product | what it does | Apple-only? |
|---|---|---|
| `Tempo9` (library) | the one public module: engine session + OpenAI/Anthropic/Ollama-shaped server | yes (links a macOS engine build) |
| `tempo9` (executable) | **the Kit's demo**: `tempo9 --gguf model.gguf` — one file, one command, four API dialects | yes |

### The CLI is a demo, and that is the point

`tempo9` holds no inference logic of its own. Everything it does — loading a
GGUF, serving OpenAI, Anthropic, Responses and Ollama dialects, tool calls,
vision, KV offload — is `Tempo9` library API, and `Sources/tempo9-cli/main.swift`
is under two hundred lines of argument parsing around it.

That is deliberate on both sides. It keeps the CLI honest (anything it can do,
your app can do, because it is the same code path), and it makes the file the
shortest complete answer to "how do I use this Kit" — shorter than any tutorial
we would write, and it cannot drift, because it is compiled and shipped.

Read it first. Then [`examples/`](../examples/) for requests against a running
server, and [`manual/`](../manual/) for what the flags mean.

Internally that splits into targets (`GGUFKit` gguf+tokenizers, `ChatTemplateKit`
Jinja, `VisionTowerKit` vision/audio towers, `Tempo9Engine` the C-ABI binding),
but only `Tempo9` is exported — an internal target layout is not an API promise.
`ggufctl` / `vitctl` / `localctl` are development tools and are not shipped.

A host that only needs to read a model file and tokenize should not have to
pull in a template engine, let alone Core ML — so `GGUFKit` depends on
Foundation and nothing else.

## Use it

```swift
.package(url: "https://github.com/thinkspread/tempo9", branch: "main")
```

(Pin a version once the first release is tagged; that release also carries
the engine an app links. Until then this compiles, and linking an app
fails for want of the engine.)

then take the product:

```swift
.product(name: "Tempo9", package: "tempo9")
```

```swift
import Tempo9

let session = try LocalSession(modelName: "app", graphPath: "",
                               ggufPath: "/path/model.gguf",
                               maxLength: 32768, maxBatch: 16)
let server = OpenAIServer(session: session, modelName: "local", port: 11435)
try server.start()
```

Those five lines are what `tempo9` does; see `Sources/tempo9-cli/main.swift`
for the rest of it, which is argument parsing.

`Tempo9` is the only library product. The internal targets named above are
not exported, so `.product(name: "GGUFKit", ...)` does not resolve — that
example was here until 2026-08-31 and never worked after the products were
collapsed.

To run a server instead of linking one, see [`manual/`](../manual/); for
five working requests spanning the OpenAI, Anthropic and Ollama
dialects, [`examples/`](../examples/).

## Building anything that touches the engine

`Tempo9` links the engine, which is **not vendored here** —
`Tempo9Kit/Sources/CTempo9Engine` is a header shim over its C ABI
(`tempo9_engine.h`, `te9_*` symbols). Build the engine first:

```bash
AS_PLATFORM=macos AS_ENABLE_METAL=ON AS_BUILD_PYTHON=ON bash build.sh
```

Write that line in full. `AS_PLATFORM` defaults to `cuda`, and on a Mac that
default does not fail loudly — it reconfigures the build directory with CUDA
settings and dies on a missing header, leaving the cache poisoned so the next
build fails the same way and looks like a pre-existing breakage.

Then point the linker at it:

```bash
swift build -Xlinker -L$TEMPO9/build/csrc
```

Two traps that are silent rather than loud:

- The Metal kernels are compiled **from source at runtime**. A shipped
  binary compiles the sources **embedded in it at build time**, so this is
  a dev-only concern — but in a dev tree, a binary linked against one
  engine build while compiling another checkout's kernels measures a
  chimera. Keep the archives and the `.metal` sources from the same build
  together and set `AS_METAL_KERNEL_DIR` to that directory; an explicitly
  set dir that is unreadable fails loudly rather than silently substituting
  the embedded copy.
- `swift build` does **not** relink when only the engine archives change:
  SwiftPM does not treat them as inputs, so it exits 0 having done nothing
  and whatever you test next is the previous engine. Remove the binary to
  force it.

## What is actually portable

`GGUFKit` imports Foundation and nothing else, and `ChatTemplateKit` adds
only cross-platform Swift packages. Both would run on a Linux host unchanged.
CI asserts this rather than trusting the sentence: an earlier draft of this
file claimed the serving module was portable too, which is false — it decodes
images through CoreGraphics and ImageIO.
