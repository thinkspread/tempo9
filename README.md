# Tempo9

**Server-grade LLM inference on Apple Silicon.** Continuous batching, paged
KV cache, GGUF direct load, GPU + Neural Engine execution — one binary, one
command:

```bash
tempo9 --gguf your-model.gguf --port 11435
```

That serves OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`),
and OpenAI Responses (`/v1/responses`) APIs on localhost — Cursor, Claude
Code, Codex CLI and anything OpenAI-compatible can point at it directly.

## Install

> **Not yet.** These three need a published release carrying a signed
> binary, and the first one is not out. Until it is, the Swift package below
> compiles and its tests run, but an app cannot link the engine. Run the
> installer anyway and it says so — "no published release for
> thinkspread/tempo9 yet" — rather than failing in a way you have to decode.

```bash
curl -fsSL https://raw.githubusercontent.com/thinkspread/tempo9/main/install.sh | bash
```

or with Homebrew:

```bash
brew install thinkspread/tap/tempo9
```

or grab the tarball from [Releases](https://github.com/thinkspread/tempo9/releases).

## Why

Tempo9 is built for several agents at once, not one person typing, and it
is measured that way. Numbers from the 2026-08-30 benchmark write-up: Apple
M5 Pro, 24 GB, default macOS settings; llama.cpp b10307 built from source
(Metal + BLAS); both engines reading the same Qwen3.5-35B-A3B q3km GGUF.
Paired figures are two runs.

- **2.8× aggregate decode throughput at 8 concurrent streams**
  (105.5 / 105.2 vs 38.1 / 32.2 tok/s) — continuous batching, not queueing
- **128K declared context at flat memory** (547–557 MB vs 3,625–3,657 MB,
  where llama.cpp ran out of memory in both runs) — paged KV allocates for
  what you use, not the worst case
- **27% lower energy per token** at 8 concurrent streams (0.268 vs 0.368
  J/token, one sampling window)
- **1.14–1.20× single-stream decode** (68.2 / 71.9 vs 60.0 / 60.1 tok/s)

Where it does not win: one stream's prefill is 0.90× llama.cpp's, and a
single stream at 7K context is 0.64× end to end.

## Models

Feed it a standard GGUF straight from Hugging Face — no conversion step.
Verified: Qwen3.5 (dense + MoE + VL), Qwen3, Llama 3.1/3.2, Mistral,
Ministral, Gemma 4, GPT-OSS, TinyLlama — 16 model/quant combinations in
the support matrix, including 10K-context and vision tests.

## Build apps on it

Two ways, and they are different products rather than two flavours of one:

- **Over HTTP.** Point any OpenAI-compatible client at the running server.
  Nothing to link, any language. Claude Code and Codex speak their own
  protocols and both are served — see [manual/claude-code.md](manual/claude-code.md)
  for what WebSearch does on a local model.
- **In-process, on Apple platforms.** The Swift SDK in this repository — one
  library, `import Tempo9` — holds the engine inside your app: no server, no
  localhost, one copy of the weights. See [docs/swift-sdk.md](docs/swift-sdk.md).

```swift
.package(url: "https://github.com/thinkspread/tempo9", branch: "main")
```

Pin a version once the first release is tagged; that release also carries
the engine an app links.

Application-level guidance for both — streaming, tool calling, guided
decoding, agent loops, multimodal — is collected as skills in a companion
repository, tempo9-skills, which is not public yet.

## Notes

- Apple Silicon (arm64) only; the engine's Metal kernels ship embedded in
  the binary — no SDK, no Xcode, no kernel files to configure.
- The binary is signed (Developer ID). If you download via a browser and
  Gatekeeper objects, right-click → Open once; `curl` and `brew` installs
  are unaffected.
- Built by the founding author of DashInfer / AllSpark; this is an
  independent continuation for local AI. See NOTICE and LICENSES in the
  tarball for provenance and third-party licenses.

## Contributing

Pull requests are welcome. Every commit needs a `Signed-off-by` line — the
[Developer Certificate of Origin](DCO) — which `git commit -s` adds.
[CONTRIBUTING.md](CONTRIBUTING.md) covers building and testing without the
engine, and what this repository can and cannot take. Security issues:
[SECURITY.md](SECURITY.md).

## License

The source in this repository — the Swift SDK, the `tempo9` CLI, the tools,
the manual and the examples — is licensed under the **Apache License 2.0**;
see [LICENSE](LICENSE).

The engine is not in this repository and is not covered by that licence. It
is distributed only as a binary, under the licence that ships with it.

[NOTICE](NOTICE) carries the attribution this project owes: what the source
here derives from (ports from DashInfer/AllSpark, HuggingFace transformers,
vLLM and tiktoken, each named beside the file it applies to), the Swift
packages it depends on, and the components linked into the distributed
binary. Full licence texts are in [LICENSES/](LICENSES/), and ship in the
release tarball too.

Source lineage: DashInfer/AllSpark (Apache-2.0), independently developed
since.
