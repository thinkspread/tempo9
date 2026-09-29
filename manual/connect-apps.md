# Connect an app

Tempo9 speaks the protocols your tools already speak. In most
cases connecting one is a change of address, not a change of code — and if
you have already pulled models with Ollama, there is nothing to download
either.

## Start the server

```
tempo9 --gguf ~/models/qwen3-8b-q4ks.gguf
```

It listens on `http://127.0.0.1:11435`, **loopback only** — the wildcard is
never bound, so nothing off this machine can reach it. Change the port with
`--port`.

## Your app has a "custom OpenAI endpoint" field

Most local-AI apps do. Fill it in:

| Field | Value |
|---|---|
| Base URL | `http://127.0.0.1:11435/v1` |
| API key | anything, or leave blank |
| Model | whatever `/v1/models` reports |

No key is required and none is checked, which is safe here only because the
server is loopback-bound. Do not put it behind a tunnel or a reverse proxy
and leave it that way.

Apps in this category include Cherry Studio, AnythingLLM, and anything else
built on an OpenAI client — we have not tested each one, so treat this as
"the field exists and this is what goes in it" rather than a certification.

## Your app only knows how to talk to Ollama

Some apps do not offer a generic endpoint field at all; they have an Ollama
integration and nothing else. Those work too. Point them at:

```
http://127.0.0.1:11435
```

Tempo9 serves Ollama's native routes alongside its own: `/api/tags`,
`/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`.
Streaming is newline-delimited JSON and tool-call arguments are objects,
exactly as those clients expect.

## You already downloaded models with Ollama

> Full detail, including what is never written: [Ollama models](ollama-models.md).

Then you do not need to download them again. Ollama stores each model as a
plain GGUF file, which is what Tempo9 loads, so it can read them where they
sit:

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

This is **read-only**. Nothing is copied, converted, moved, or written into
Ollama's store — your models stay exactly as Ollama left them, and Ollama
keeps working. If you moved the store, `OLLAMA_MODELS` is respected.

A bare name means `:latest`, the same as `ollama run`.

## Agent tools

Claude Code, Codex and similar clients do not use the OpenAI chat protocol —
they each want their own. Tempo9 serves all three from one process:

| Protocol | Route | Spoken by |
|---|---|---|
| OpenAI Chat Completions | `/v1/chat/completions` | most apps and SDKs |
| Anthropic Messages | `/v1/messages` | Claude Code — see [Claude Code](claude-code.md) |
| OpenAI Responses | `/v1/responses` | Codex |
| Ollama native | `/api/chat` | apps with an Ollama integration |

Agent workloads are the case Tempo9 is built for: many requests at once, long
and growing context, heavy tool calling. That is where the architecture shows
up rather than in a single stream — see the benchmarks.

## What it does not do

> Also: [Limits](limits.md).

Worth knowing before you wire something up:

- **No embeddings.** `/api/embed` and `/v1/embeddings` are not served;
  the Ollama route answers `501` and says so rather than pretending.
- **One model per process.** `/v1/models` and `/api/tags` report the one
  that is loaded. Run a second server on another port for a second model.
- **Vision needs a tower.** Image input works with `--tower <dir>`. A GGUF
  vision projector inside an Ollama model is detected but not loaded; the
  server tells you when it sees one.
- **No authentication.** By design, and only defensible because it is
  loopback-bound.
- **No server-side tools.** Claude Code's WebSearch and WebFetch, code
  execution, and the OpenAI hosted tools run on the vendor's servers; a
  request that carries one is refused with `400` naming it, never
  quietly answered without it. `GET /v1/capabilities` lists what is
  served. See [Claude Code](claude-code.md).
