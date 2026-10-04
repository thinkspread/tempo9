# Tempo9

**Server-grade LLM inference on Apple Silicon.** Continuous batching, paged
KV cache, GGUF direct load, GPU + Neural Engine execution — one binary, one
command:

```bash
brew install thinkspread/tap/tempo9
tempo9 --hf unsloth/Qwen3.5-9B-GGUF:Q4_K_S
```

`--hf owner/repo[:quant]` downloads a GGUF into the standard Hugging Face
cache, where a copy that huggingface_hub or llama.cpp already pulled is
reused instead of downloaded again. Any local file works too:
`tempo9 --gguf your-model.gguf`. So does a model that Ollama already
pulled: `tempo9 --ollama qwen3:8b`.

That serves OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`),
and OpenAI Responses (`/v1/responses`) APIs on localhost — Cursor, Claude
Code, Codex CLI and anything OpenAI-compatible can point at it directly.
Claude Code, for example:

```bash
tempo9 --hf unsloth/Qwen3.5-9B-GGUF:Q4_K_S --max-length 65536
ANTHROPIC_BASE_URL=http://127.0.0.1:11435 ANTHROPIC_API_KEY=local claude
```

Claude Code's system prompt alone is about 32k tokens, so give it the
larger context. Its first request on a fresh server prefills about 40k
tokens: roughly 50 s with Qwen3.5-9B and 25 s with Gemma-4-E4B on an
M5 Pro. After that the prefix cache carries it, so each tool-call turn
prefills in under 1.5 s, and a later Qwen session on the same server
took 10-23 s for a whole question. Qwen3.5-9B is the more reliable
choice for multi-step work; Gemma-4-E4B is faster.
[manual/claude-code.md](manual/claude-code.md#what-to-expect-time-per-question-and-which-model)
has the measurements.

## Install

Apple Silicon and macOS 26 or later. Any of these installs the signed,
notarized `tempo9` binary:

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
is measured that way. [The 2026-10 benchmark](docs/benchmark-2026-10-m5pro.md)
used the SiliconBench pipeline on an Apple M5 Pro with 24 GB and default
macOS settings, and compared 7 models on four engines:

- **At 16 concurrent requests, 1.4–2× the throughput of omlx and
  llama.cpp**, and the first token arrives 6–24× sooner, on Qwen3.5
  0.8B–9B and Gemma-4-E4B. The one exception is Qwen3.5-0.8B agent
  against omlx, where the lead is 1.07×.
- **Ahead of vllm-metal on every Qwen3.5 throughput cell** (+4…+33 %),
  and at parity on Gemma-4-E4B.
- **Qwen3.5-35B-A3B on 24 GB.** vllm-metal and omlx (MLX 4-bit) run out of
  GPU memory. On the same GGUF as llama.cpp, Tempo9 is 15–59 % faster,
  with a 1.6 s median first token at 16 concurrent chats against 18 s.
- **Quality on par.** On the Qwen3.5 and Gemma models, GMRID F1 is
  within run-to-run noise of the other engines.

Where it does not win: under heavy concurrency vllm-metal's first token
still comes sooner (Qwen3.5-9B agent at 16 concurrent: 7.5 s vs 5.4 s),
and a single stream is close everywhere. omlx is faster on the
smallest models (Qwen3.5-0.8B chat: 211 vs 177 tok/s).

Earlier, against llama.cpp only (2026-08-30, Qwen3.5-35B-A3B q3km):
**128K declared context at flat memory**, 547–557 MB against 3,625–3,657 MB,
where llama.cpp ran out of memory. Paged KV allocates for what you use,
not for the worst case. Energy per token was **27 % lower** at 8
concurrent streams (0.268 vs 0.368 J/token, one sampling window).

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
- **In-process, on Apple platforms.** The Swift SDK in this repository —
  `import Tempo9`, plus `VisionTowerKit` for image input and `GGUFKit` for
  reading model files — holds the engine inside your app: no server, no
  localhost, one copy of the weights. See [docs/swift-sdk.md](docs/swift-sdk.md).

```swift
.package(url: "https://github.com/thinkspread/tempo9", from: "1.1.2")
```

SwiftPM downloads the engine (`Tempo9Engine.xcframework`) from the release;
an app declares `platforms: [.macOS("26.0")]` (see
[docs/swift-sdk.md](docs/swift-sdk.md) for why).

Application-level guidance for both — streaming, tool calling, guided
decoding, agent loops, multimodal — is collected as skills in
[thinkspread/tempo9-skills](https://github.com/thinkspread/tempo9-skills).

## Notes

- Apple Silicon (arm64) and macOS 26 or later only; the engine's Metal
  kernels ship embedded in the binary — no SDK, no Xcode, no kernel files to
  configure.
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
is distributed only as a binary, under the Tempo9 Engine License
([ENGINE-LICENSE](ENGINE-LICENSE)): free for individuals and for
organisations with fewer than 100 employees and under US$1M in annual
revenue; above that, a commercial licence is required.

[NOTICE](NOTICE) carries the attribution this project owes: what the source
here derives from (ports from DashInfer/AllSpark, HuggingFace transformers,
vLLM and tiktoken, each named beside the file it applies to), the Swift
packages it depends on, and the components linked into the distributed
binary. Full licence texts are in [LICENSES/](LICENSES/), and ship in the
release tarball too.

Source lineage: DashInfer/AllSpark (Apache-2.0), independently developed
since.
