# API reference

One process serves four dialects on one port. Nothing needs to be enabled;
they are all live.

## OpenAI

| Method | Route | Notes |
|---|---|---|
| `GET` | `/v1/models` | Reports the one loaded model. |
| `GET` | `/v1/models/{id}` | Single-model lookup. Answers for any id. |
| `POST` | `/v1/chat/completions` | Chat. `stream` defaults to **false**. |
| `POST` | `/v1/responses` | The Responses API, which Codex speaks. |
| `GET` | `/v1/capabilities` | What is served, refused (and why), and stripped, per protocol. See [Claude Code](claude-code.md). |

## Anthropic

| Method | Route | Notes |
|---|---|---|
| `POST` | `/v1/messages` | What Claude Code speaks. |
| `POST` | `/v1/messages/count_tokens` | The real tokenizer over the prompt `/v1/messages` would prefill. Claude Code budgets its context window from it. |

## Ollama native

For apps that integrate Ollama and offer no generic endpoint field.

| Method | Route | Notes |
|---|---|---|
| `GET` | `/api/tags` | Model list. This is the discovery call. |
| `GET` | `/api/version` | Reports `0.12.0-tempo9` — semver-parseable, and tagged so it never claims to be an Ollama release. |
| `GET` | `/api/ps` | Loaded models. One, always resident. |
| `POST` | `/api/show` | Model detail. `capabilities` reflects this build: `tools` always, `vision` only with `--tower`. |
| `POST` | `/api/chat` | Chat. |
| `POST` | `/api/generate` | Bare-prompt completion. Replies use `response`, not `message`. |
| `POST` | `/api/embed`, `/api/embeddings` | **501.** We serve no embeddings, and say so in the caller's dialect rather than 404ing. |

Three things differ from the OpenAI dialect, and each fails silently against
a client that assumes otherwise:

1. **Streaming is newline-delimited JSON.** No `data: ` prefix, no `[DONE]`
   sentinel; the last object carries `done: true`.
2. **`stream` defaults to `true`.** The opposite of OpenAI.
3. **Tool-call arguments are a JSON object**, not a JSON string.

And one place where this server deliberately does *not* behave like
Ollama. Ollama silently truncates a prompt that does not fit the context
(`truncate` defaults to `true` there). Here a prompt that alone fills
`--max-length` is **refused with `400`** in Ollama's error shape,
`{"error": "prompt is 33652 tokens and max_tokens is 48: 33700 exceeds this server's max length 32768, and the prompt alone leaves no room for a reply (start tempo9 with a larger --max-length, or shorten the prompt)"}`,
before any byte of a stream is out — the same rule and the same numbers
as the other three protocols ([Context length](claude-code.md#context-length)).
When the prompt fits but `options.num_predict` (Ollama's `max_tokens`)
does not, `num_predict` is clamped to the room left; the server logs one
line and the reply carries it in a top-level `warnings` array on the
final object (the `done: true` line when streaming — Ollama's response
has no field for it, and its clients ignore fields they do not know),
and a reply that hits the clamped budget reports `done_reason:
"length"`. The warning names the value as `max_tokens`, the engine-side
name for `num_predict`.

## Tools

Custom / function tools pass through on every protocol. Anthropic's
client-executed built-ins (`bash_*`, `text_editor_*`, `computer_*`,
`memory_*`) pass through with their published schema. Server-executed tools
(`web_search_*`, `web_fetch_*`, `code_execution_*`; OpenAI `web_search`,
`file_search`, `code_interpreter`) are refused with `400` naming
`tools[i].type`, or stripped with a `warnings` array when the server runs
with `TEMPO9_STRIP_UNSUPPORTED_TOOLS=1`. Full rules and the error shape:
[Claude Code](claude-code.md).

## Authentication

None. See [Limits](limits.md#no-authentication).
