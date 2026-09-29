# Ollama models

If you already pulled models with Ollama, Tempo9 can run them **where they
sit** — no download, no copy, no conversion.

```
tempo9 --list-ollama
```

```
qwen3:8b            4.68 GiB  /Users/you/.ollama/models/blobs/sha256-1f3d...
llama3.2:latest     1.88 GiB  /Users/you/.ollama/models/blobs/sha256-a41c...
```

```
tempo9 --ollama qwen3:8b
```

A bare name means `:latest`, the same as `ollama run`. If you moved the
store, `OLLAMA_MODELS` is respected.

**Ollama does not need to be running.** This is not a protocol; nothing
talks to the Ollama process. It is a read of its files.

## How it works

Ollama has no model format of its own. It stores plain GGUF wrapped in an
OCI-style content-addressed layout for deduplication and versioning:

```
~/.ollama/models/
├── manifests/registry.ollama.ai/library/qwen3/8b   small JSON — the path IS the name
└── blobs/sha256-1f3d...                            bare GGUF — the weights
```

The manifest lists layers:

```json
{"schemaVersion":2,"layers":[
  {"mediaType":"application/vnd.ollama.image.model",   "digest":"sha256:1f3d…","size":4831838208},
  {"mediaType":"application/vnd.ollama.image.template","digest":"sha256:a7c1…","size":1482}
]}
```

The weights are the `.model` layer. Its digest `sha256:1f3d…` names the file
`blobs/sha256-1f3d…` — **colon becomes dash**. That blob is byte-identical to
the GGUF you would download yourself, and Tempo9 loads GGUF, so it simply
opens it.

`registry.ollama.ai/library` is implicit in Ollama's display convention, so
that manifest shows as `qwen3:8b`. Models from elsewhere keep their full
`host/namespace/model:tag` name, and still answer to the short form.

## What is deliberately not used

The manifest also carries `template` and `params` layers. Tempo9 ignores
them and uses the **chat template embedded in the GGUF** — the one the model
author shipped.

Ollama's template layer is written in Go's `text/template`; ours is Jinja.
Honouring theirs would mean implementing a second template engine to obtain
a transcription of something we already have in the original.

## Why a symlink

Tempo9 creates one file, in its own cache, never in Ollama's store:

```
~/.cache/tempo9/ollama/qwen3-8b.gguf -> ~/.ollama/models/blobs/sha256-1f3d...
```

The engine currently decides GGUF-vs-graph by the **file extension**, and
Ollama's blobs have none. Handed a blob path directly it refuses, and it
refuses silently. The link works around that, and gives the engine's graph
cache a legible key besides — that cache is keyed on basename, size and
mtime, and `qwen3-8b.gguf` identifies itself where a digest would not.

A link left pointing at a blob that a re-pull replaced is detected and
rewritten, so a re-pulled tag never serves last week's weights.

## What it will not do

- **Never writes to Ollama's store.** Read-only: directory listing, file
  reads, attribute lookups. Nothing else.
- **Skips a model whose blob is missing.** Interrupted pulls leave manifests
  behind; listing one would turn a clean "not found" into a confusing load
  failure later.
- **Trusts the file over the manifest for size**, because a truncated pull
  leaves the advertised size intact and the bytes short.
- **Does not load a GGUF vision projector.** It is detected and reported;
  image input needs `--tower`. See [Limits](limits.md#vision-needs-a-core-ml-tower).
