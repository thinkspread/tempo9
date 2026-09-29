# Models and quantizations

The question before downloading anything is "will my file run". Two axes
decide it, and neither is visible from the other: the **quantization** and
the **architecture**. A third, the tokenizer, hides inside the second.

Nothing here is inferred from source. Every `verified` below means a model
in that format answered prompts on this machine.

## Quantizations

GGUF blocks are used **as they are**. Nothing is dequantized on load, so a
Q4_K file costs Q4_K memory rather than expanding to FP16.

### Verified — a real model in this format was served

| GGUF type | Block | Seen in |
|---|---|---|
| `Q8_0` | 32 / 34 B | Qwen3 0.6B, Qwen3.5 0.8B |
| `Q4_K` (`Q4_K_S`, `Q4_K_M`, `UD-Q4_K_XL`) | 256 / 144 B | most of the table below |
| `Q5_K` | 256 / 176 B | as the companion type inside `Q4_K_M` files |
| `Q6_K` | 256 / 210 B | Qwen3.5 4B Q6_K |
| `Q3_K` (`Q3_K_M`) | 256 / 110 B | Qwen3.5 35B-A3B |
| `MXFP4` | 32 / 17 B | GPT-OSS 20B |
| `F32` | — | MoE routing gates, zero-copy |

`Q4_K_M` and `Q5_K_M` files are **mixed**: some layers promote `attn_v` or
`ffn_down` to `Q6_K` while the rest stay `Q4_K`. That is handled per segment,
which is why `Q5_K` and `Q6_K` appear as companions above without a
single-format file of their own.

### Supported, but no model in this format has been run here

| GGUF type | Note |
|---|---|
| `Q4_0` | The ARM edge tier. Code path present on all sides. |
| `F16` / `BF16` | Converted to FP32 at load. Fine for the KB–MB tensors that use it; a whole model in F16 is not the intended path. |

### Refused, with an error that says why

There is **no silent fallback**. A file in one of these formats fails at
load with an actionable message rather than running slowly or badly:

`IQ1` – `IQ4` · `Q2_K` · `Q4_1` · `Q5_0` · `Q5_1` ·
`Q4_0_4_4` / `4_8` / `8_8` (removed from GGUF upstream) · `TQ*` ·
`NVFP4` · `Q8_K` (an activation-only type — in a weight table it means the
file is corrupt)

If your file is one of these, the usual fix is a `Q4_K_M` or `Q8_0` build of
the same model.

> **On picking a quantization for serving**, `Q4_K_S` is generally preferred
> over `Q4_K_M` here: no faster `Q6_K` kernel exists, so the promoted tensors
> in a `_M` file cost more than they return.

## Architectures

GGUF direct load accepts seven: `qwen35`, `qwen35moe`, `gemma4`, `qwen2vl`,
`qwen3`, `qwen3moe`, and `llama` — the last covering the Llama line and most
of Mistral.

**The architecture field does not tell you whether a file will run**, because
the tokenizer is a second axis and invisible from there. Mistral 7B v0.3 and
Llama 3.2 are both `architecture=llama`; one writes a SentencePiece
tokenizer and the other writes `gpt2`.

| Family | Arch | Tokenizer |
|---|---|---|
| Llama 3.1 / 3.2 / 3.3 | `llama` | gpt2 |
| Mistral v0.1 – v0.3 | `llama` | SentencePiece |
| Llama 2, CodeLlama, Vicuna | `llama` | SentencePiece |
| Qwen3.5 / Qwen3 / Gemma 4 | own | gpt2 / gemma4 |

Both sides of that split work. A dialect the engine does not implement is
**refused by name** — never guessed — because a vocabulary read with the
wrong algorithm still produces ids, and those ids decode into fluent text
from the wrong distribution. A wrong answer that reads well is worse than an
error.

## Verified models

Produced by serving real requests to each model, not by reading a registry.
The generator runs in the engine's own test tooling and is not part of this
repository; the table below is its output.

| Model | Load | Text | 10k context | Thinking | Image |
|---|---|---|---|---|---|
| gemma4-12b-it-q4ks | ok | PASS | PASS | — | PASS |
| gptoss-20b-mxfp4 | ok | PASS | PASS | PASS | — |
| llama31-8b-q4km | ok | PASS | PASS | — | — |
| llama32-1b-q4km | ok | PASS | PASS | — | — |
| ministral3-3b-q4km | ok | PASS | PASS | — | — |
| mistral7b-v03-q4km | ok | PASS | PASS | — | — |
| qwen3-0.6b-q8_0 | ok | PASS | PASS | PASS | — |
| qwen35-0.8b-q8_0 | ok | PASS | PASS | PASS | — |
| qwen35-35b-a3b-q3km | ok | PASS | PASS | PASS | PASS |
| qwen35-4b-q4ks | ok | PASS | PASS | PASS | — |
| qwen35-4b-q6k | ok | PASS | PASS | PASS | — |
| qwen35-9b-imat-q4ks | ok | PASS | PASS | PASS | PASS |
| qwen38-27b-q4ks | ok | PASS | PASS | — | PASS |
| qwen3vl-4b-q4km | ok | PASS | PASS | — | PASS |
| qwen3vlmoe-30b-a3b-q4km | ok | PASS | PASS | — | PASS |
| tinyllama-1.1b-q4km | ok | PASS | N/A | — | — |

A blank cell means **nothing was run** — no vision front end paired with that
model, or no separate thinking mode to test. `N/A` means the model never
claimed it: TinyLlama declares a 2048-token context, so a 10k prompt is
outside what it offers. Neither is a failure, and both are printed rather
than omitted, because the gap between "not tested" and "does not work" is
the entire reason to publish a table instead of a claim.

The `10k context` column exists because every check here was once a short
prompt, and the table was green for two models that produced word salad past
about 8k.

## What the model registry is not

The engine's C++ registry lists roughly thirty model families — LLaMA v1–v4,
ChatGLM, DeepSeek, GLM-5, MiniMax-M2, Baichuan, the Qwen line. Those reach
the engine through a Python conversion path, and **none of them is verified
on Metal**. Reading that registry as a support list would overstate this
page by an order of magnitude, which is why the page is generated by running
models instead.

Commonly recommended for a 24 GB Mac, and where they stand:

| Model | Status |
|---|---|
| Llama 3.3 8B | yes — verified at 3.1 8B |
| Mistral 7B v0.3 | yes |
| GPT-OSS 20B | yes — MXFP4 |
| Mistral Small 3.2 24B | architecture yes, this size untested |
| Qwen3.6-27B | untested |
| Gemma 4 31B | architecture yes, this size untested |
| Phi-4 | no |
