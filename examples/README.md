# Examples

Runnable against a local Tempo9. Start one first:

```
tempo9 --gguf ~/models/qwen3-8b-q4ks.gguf
# or, if you already use Ollama:
tempo9 --ollama qwen3:8b
```

Everything here defaults to `http://127.0.0.1:11435` and honours `$TEMPO9`:

```
TEMPO9=http://127.0.0.1:11440 examples/curl/openai-chat.sh
```

| File | Shows |
|---|---|
| [`curl/openai-chat.sh`](curl/openai-chat.sh) | The plainest request there is |
| [`curl/openai-stream.sh`](curl/openai-stream.sh) | SSE streaming |
| [`curl/openai-tools.sh`](curl/openai-tools.sh) | Tool calling, and that arguments are typed |
| [`curl/anthropic-messages.sh`](curl/anthropic-messages.sh) | The dialect Claude Code speaks |
| [`curl/ollama-chat.sh`](curl/ollama-chat.sh) | Ollama's native routes — NDJSON, not SSE |
| [`python/no_deps.py`](python/no_deps.py) | Standard library only, nothing to install |
| [`python/openai_sdk.py`](python/openai_sdk.py) | The official SDK, pointed at a local base_url |

## Per-model

[`models/`](models/) shows what changes when the model does — and, for the
35B, what the concurrency is for. Those scripts **never start a server**;
each carries the `tempo9` command to run yourself.

| File | Model |
|---|---|
| [`models/qwen35-4b.sh`](models/qwen35-4b.sh) | Qwen3.5 4B — the everyday case |
| [`models/qwen35-35b-a3b.sh`](models/qwen35-35b-a3b.sh) | Qwen3.5 35B-A3B — eight streams at once |
| [`models/qwen38-27b.sh`](models/qwen38-27b.sh) | Qwen3.8 27B — dense vs MoE, 10k context |
| [`models/qwen3vl-4b.sh`](models/qwen3vl-4b.sh) | Qwen3-VL 4B — images, both dialects |
| [`models/gemma4-12b.sh`](models/gemma4-12b.sh) | Gemma 4 12B — and an honest gap |

Docs are in [`../manual/`](../manual/).
