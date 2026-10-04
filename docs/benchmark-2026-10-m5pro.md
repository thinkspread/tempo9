Copyright (c) 2026 Jiejing Zhang.

# Benchmark: Tempo9 vs vllm-metal, omlx and llama.cpp (M5 Pro 24 GB, 2026-10)

Seven models and four engines, measured on one machine with the
[SiliconBench](https://github.com/WindChimeRan/SiliconBench) pipeline. The
Tempo9 figures are for the engine and Swift code that ship in **1.1.2**.

**Who ran this.** We did. SiliconBench is a third-party benchmark, but the
Tempo9 adapter that drives it is ours and is not upstream. So read these
numbers as a self-report that used someone else's harness, and not as an
independent result.

## Setup

| | |
|---|---|
| Machine | Apple M5 Pro, 24 GB unified memory, on mains power |
| OS | macOS 26.6.2 (25G83); `iogpu.wired_limit_mb` left at the system default |
| Prompts | fixed at Hub `windchimeran/SiliconBench@3e9fa20c` |
| Tempo9 | 1.1.2 (engine 3.0.0-rc3); every server start was checked for Metal + tensor units on |
| vllm-metal | 0.30.0.dev20260929 |
| omlx | 0.7.0rc1 (mlx 0.32.2) |
| llama.cpp | `19e28a277` (2026-09-29) |

**Method.** This is the stock SiliconBench pipeline:
- The server restarts at each concurrency level (1, 8, 16).
- Each level gets 3 warm-up requests, then 100 measured requests, then a 60 s cool-down.
- Speculative decoding is off, and every engine runs with its default settings.
- Before each cell starts, the machine must be idle: no other inference, compile or CI jobs, and CPU at least 85 % idle.

**Weights.** We compare 8-bit against 8-bit:
- Tempo9 and llama.cpp load GGUF Q8_0.
- vllm-metal and omlx load MLX 8-bit.
- The exceptions are Qwen3-0.6B BF16, which is the paper's reference arm, and the 35B. The 35B uses GGUF Q3_K_M against MLX 4-bit, because those are the builds that fit in 24 GB, and the MLX one did not fit anyway.

**Deviations from stock, all recorded:**
- Tempo9 gets a fresh KV-offload directory on each start, so its disk cache cannot carry hits from one level into the next.
- omlx binds to 127.0.0.1, because it refuses to bind 0.0.0.0 without an API key.
- llama.cpp gets `--ctx-size 65536` on the chat and agent runs, and on the 9B and Gemma quality runs. Without it, it sizes the KV cache to the model's training context and runs out of GPU memory. This change favours llama.cpp.

## Throughput (output tok/s at concurrency 1 / 8 / 16)

The numbers in brackets are median time-to-first-token in ms, at concurrency 1 / 16.

**chat**

| model | Tempo9 | vllm-metal | omlx | llama.cpp |
|---|---|---|---|---|
| Qwen3.5-0.8B | **177 / 461 / 535** (50 / 127) | 143 / 369 / 401 (70 / 194) | 211 / 358 / 359 (61 / 2867) | 169 / 325 / 325 (58 / 3000) |
| Qwen3.5-4B | **44.3 / 117 / 141** (201 / 513) | 39.1 / 111 / 119 (258 / 805) | 45.5 / 86.8 / 87.8 (247 / 11060) | 40.3 / 80.7 / 84.8 (319 / 11370) |
| Qwen3.5-9B | **26.3 / 75.6 / 93.1** (352 / 1007) | 23.9 / 72.8 / 80.5 (416 / 902) | 26.2 / 56.4 / 56.6 (401 / 17072) | 24.2 / 51.1 / 51.0 (591 / 21368) |
| Gemma-4-E4B | **41.4 / 127 / 167** (144 / 634) | 42.5 / 147 / 171 (153 / 459) | 42.2 / 103 / 102 (185 / 8711) | 37.8 / 82.0 / 81.6 (247 / 10693) |
| Qwen3.5-35B-A3B | **52.0 / 92.2 / 78.4** (368 / 1634) | out of memory | out of memory | 45.1 / 65.7 / 62.5 (680 / 17957) |
| Qwen3-0.6B (8-bit) | **161 / 306 / 329** (37 / 153) | 185 / 417 / 405 (37 / 156) | 176 / 228 / 225 (72 / 2538) | 160 / 264 / 267 (56 / 2414) |

**agent** (long prompts with tool calls)

| model | Tempo9 | vllm-metal | omlx | llama.cpp |
|---|---|---|---|---|
| Qwen3.5-0.8B | **94.6 / 142 / 143** (280 / 1053) | 86.5 / 124 / 117 (218 / 1017) | 123 / 137 / 134 (233 / 6049) | 88.5 / 114 / 106 (262 / 7277) |
| Qwen3.5-4B | **24.1 / 36.1 / 37.6** (1234 / 4778) | 22.1 / 32.4 / 32.6 (1072 / 3931) | 25.7 / 25.8 / 24.1 (1136 / 35476) | 20.4 / 26.9 / 25.5 (1533 / 33976) |
| Qwen3.5-9B | **16.7 / 28.3 / 27.7** (2003 / 7492) | 16.0 / 25.0 / 25.3 (2033 / 5426) | 15.8 / 20.0 / 20.2 (2376 / 61025) | 13.9 / 20.7 / 19.6 (2651 / 54133) |
| Gemma-4-E4B | **31.8 / 69.0 / 79.3** (628 / 1875) | 33.0 / 71.9 / 74.6 (553 / 1148) | 31.0 / 46.2 / 47.2 (824 / 22678) | 25.8 / 38.8 / 39.0 (1136 / 30823) |
| Qwen3.5-35B-A3B | **26.8 / 35.1 / 31.2** (1611 / 6522) | out of memory | out of memory | 20.0 / 22.1 / 23.3 (2653 / 54287) |
| Qwen3-0.6B (8-bit) | **88.3 / 127 / 129** (203 / 726) | 110 / 156 / 147 (146 / 594) | 92.4 / 56.6 / 53.9 (317 / 11700) | 56.3 / 68.1 / 61.9 (552 / 12925) |

On Gemma-4-E4B agent, vllm-metal and omlx completed 94 of 100 requests. Six
very long prompts came back with 0 tokens, and the pipeline counts them as
failures. Those requests also spared them six of the heaviest prefills, so
their agent throughput is slightly flattered. Tempo9 and llama.cpp
completed 100 of 100.

## Quality (GMRID weighted F1, 0-shot / 5-shot, 1146 items each)

| model | Tempo9 | vllm-metal | omlx | llama.cpp |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.699 / 0.596 | 0.693 / 0.606 | 0.692 / 0.592 | 0.697 / 0.599 |
| Qwen3.5-4B | 0.885 / 0.935 | 0.891 / 0.936 | 0.888 / 0.939 | 0.886 / 0.936 |
| Qwen3.5-9B | 0.928 / 0.945 | 0.928 / 0.946 | 0.927 / 0.946 | 0.927 / 0.944 |
| Gemma-4-E4B | 0.848 / 0.908 | 0.855 / 0.912 | 0.854 / 0.910 | 0.850 / 0.908 |
| Qwen3.5-35B-A3B | 0.934 / 0.939 | out of memory | out of memory | 0.934 / 0.941 |
| Qwen3-0.6B (8-bit) | 0.382 / 0.735 | 0.390 / 0.732 | 0.396 / 0.737 | 0.401 / 0.737 |

Rerunning one configuration gives a spread of about ±0.008 (standard
deviation, 0-shot), because the quality runs use 8 concurrent requests and
the batches differ from run to run. Every gap in this table is within that
noise except Qwen3-0.6B 0-shot, where Tempo9 is 0.008–0.019 lower.

## What it says

- **Under concurrency, Tempo9 is ahead of omlx and llama.cpp by a wide margin.**
  - On Qwen3.5 0.8B–9B and Gemma-4-E4B at 16 concurrent requests, its throughput is 1.4–2× theirs. The one exception is Qwen3.5-0.8B agent against omlx, where the lead is 1.07×.
  - Its first token arrives 6–24× sooner.
- **Against vllm-metal it is close.**
  - Every Qwen3.5 throughput cell is ahead, by 4–33 %.
  - Gemma-4-E4B is at parity: ahead at agent c16, 2–14 % behind elsewhere.
  - Qwen3-0.6B is behind.
- **Where vllm-metal still wins:** time-to-first-token under heavy concurrency. For example, Qwen3.5-9B agent at c16 takes 7.5 s on Tempo9 and 5.4 s on vllm-metal. Tempo9's prefill admits one request at a time, and that is the next thing to fix.
- **35B on 24 GB:** both MLX engines run out of GPU memory. Tempo9 is 15–59 % faster than llama.cpp on the same GGUF.
- **Single stream is close**, because decoding one stream is limited by memory bandwidth. omlx is faster on the smallest models; for example, Qwen3.5-0.8B chat runs at 211 tok/s on omlx and 177 on Tempo9.
- **Memory:** on 0.8B and 4B, Tempo9's peak memory is 4–8 GB lower than vllm-metal's. vllm-metal sat at 22.3–22.6 GB in every cell.
