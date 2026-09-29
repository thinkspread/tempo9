# Per-model examples

The scripts in [`../curl/`](../curl/) are model-agnostic. These show what
changes when the model does — and, for the 35B, what the architecture is
actually for.

**None of these start a server.** Each begins with the `tempo9` command to
run in another terminal, so an example never takes the GPU without you
deciding to. All of them honour `$TEMPO9` (default
`http://127.0.0.1:11435`).

| Script | Model | What it shows |
|---|---|---|
| [`qwen35-4b.sh`](qwen35-4b.sh) | Qwen3.5 4B | The everyday case: fast, small, tool calling |
| [`qwen35-35b-a3b.sh`](qwen35-35b-a3b.sh) | Qwen3.5 35B-A3B | **Concurrency** — eight streams at once, which is the point |
| [`qwen38-27b.sh`](qwen38-27b.sh) | Qwen3.8 27B | Dense decode, 10k context, and vision |
| [`qwen3vl-4b.sh`](qwen3vl-4b.sh) | Qwen3-VL 4B | Images, in both the OpenAI and Ollama shapes |
| [`gemma4-12b.sh`](gemma4-12b.sh) | Gemma 4 12B | A second architecture, and an honest gap |

## Dense and MoE, at the same size

Run `qwen38-27b.sh` and `qwen35-35b-a3b.sh` back to back. The two models take
almost the same space on disk — 14.74 GiB against 15.99 GiB — and decode
completely differently:

| | Weights on disk | Touched per token |
|---|---|---|
| Qwen3.8 27B | 14.74 GiB | all 27B — dense |
| Qwen3.5 35B-A3B | 15.99 GiB | ~3B active — MoE |

So the model with the *smaller* parameter count is usually the slower one per
token, which is not what the name suggests. Both scripts measure it rather
than asserting it.

On a 24 GB machine, run them **one at a time**. Two 15 GiB models resident
at once drags the system into swap and every number after that is noise.

## Vision needs a Core ML tower, not an mmproj

`--tower` takes a **directory containing `tower_meta.json`**, which is the
`coreml/` subdirectory of an exported tower:

```
--tower ~/models/qwen3vl_tower/coreml        # correct
--tower ~/models/qwen3vl_tower               # missingAsset: .../tower_meta.json
--tower ~/models/…/qwen3vl-4b-mmproj.gguf    # same error — a GGUF projector is not a tower
```

A GGUF mmproj beside the model is **not** usable here. The engine's vision
front end is Core ML; the mmproj is what a tower is exported *from*, offline.
This is also why an Ollama model carrying a GGUF projector runs text-only —
Tempo9 reports the projector and tells you it is not loading it.
