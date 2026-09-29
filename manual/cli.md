# CLI reference

```
tempo9 (--gguf <.gguf> | --ollama <model[:tag]>) [options]
tempo9 --list-ollama
```

## Choosing a model

| Flag | Default | Meaning |
|---|---|---|
| `--gguf <path>` | — | Model file. A bare `.gguf` path as the only argument works too. |
| `--ollama <model[:tag]>` | — | A model Ollama already pulled. A bare name means `:latest`. See [Ollama models](ollama-models.md). |
| `--list-ollama` | — | Print what Ollama has pulled, then exit. Exits `1` if the store is empty or absent. |
| `--graph <path>` | built from the GGUF | Precomputed graph. Rarely needed — the engine builds and caches one. |
| `--name <string>` | the file's basename, or the Ollama name | What `/v1/models` and `/api/tags` report. |

## Serving

| Flag | Default | Meaning |
|---|---|---|
| `--port <n>` | `11435` | Loopback port. |
| `--max-length <n>` | `32768` | Context length the engine is built for, prompt and reply together. Claude Code's baseline prompt alone is ~32k tokens; see [Context length](claude-code.md#context-length). |
| `--max-batch <n>` | `16` | Concurrent requests the engine schedules. `1` makes the server serial end to end. |
| `--speculation-k <n>` | `0` | MTP speculation depth (0 = off). The engine varies the effective k downward per request. |
| `--tower <dir>` | — | Core ML vision tower. Required for image input. |

There is **no `--token` flag**: nothing is authenticated. See
[Limits](limits.md#no-authentication).

## Environment

| Variable | Effect |
|---|---|
| `OLLAMA_MODELS` | Where to look for Ollama's store. Defaults to `~/.ollama/models`. |
