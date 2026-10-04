# Claude Code

Claude Code speaks the Anthropic Messages protocol, which Tempo9 serves at
`/v1/messages`. Point it at the local server and it talks to the model you
loaded:

```
tempo9 --gguf ~/models/qwen3-8b-q4ks.gguf
```

```
ANTHROPIC_BASE_URL=http://127.0.0.1:11435 ANTHROPIC_API_KEY=local claude
```

The key is not checked (the server is loopback-only; see
[Limits](limits.md#no-authentication)), but Claude Code refuses to start
without one. If your model's context is smaller than Claude's, cap what
Claude Code assumes with `CLAUDE_CODE_MAX_CONTEXT_TOKENS`.

`/v1/messages/count_tokens`, which Claude Code calls on startup and per
turn, is answered by the real tokenizer over exactly the prompt
`/v1/messages` would prefill — not an estimate.

## Context length

The engine's `--max-length` (default `32768`) is a **total** budget per
request: prompt and reply together. Claude Code's baseline prompt --
its system text plus the schemas of its 29 tools -- is about **32k
tokens before you have typed anything**, and it sends `max_tokens:
32000` on every request. So on a default server every turn is over
budget. Start with room to spare:

```
tempo9 --gguf ~/models/qwen3-8b-q4ks.gguf --max-length 65536
```

and use a model whose context is at least that large. What the server
does with a request that does not fit:

- **`max_tokens` is clamped, never refused on its own.** When the
  prompt fits but `prompt + max_tokens` does not, the reply runs with
  `max_tokens` cut to the room the prompt leaves. The server logs one
  line and the reply carries it in a top-level `warnings` array (on
  `message_start` when streaming):
  `max_tokens 32000 was clamped to 1116: the prompt is 31652 tokens and this server's max length is 32768 (start tempo9 with a larger --max-length for longer replies)`.
  A reply that hits the clamped budget reports `stop_reason:
  "max_tokens"`, as it would for any other budget.
- **A prompt that alone fills the context is refused with 400**, in the
  Anthropic error shape, naming the numbers and both ways out:
  `prompt is 33652 tokens and max_tokens is 48: 33700 exceeds this server's max length 32768, and the prompt alone leaves no room for a reply (start tempo9 with a larger --max-length, or shorten the prompt)`.
  This is a `400`, not a `500`: nothing on the server failed, and the
  same request will fail the same way until the prompt or the server
  changes.

To see how large a prompt is before sending it, ask the server -- the
answer is the real tokenizer over exactly the prompt `/v1/messages`
would prefill:

```
curl -s http://127.0.0.1:11435/v1/messages/count_tokens \
  -H 'content-type: application/json' \
  -d '{"model":"local","system":"...","messages":[{"role":"user","content":"..."}],"tools":[...]}'
```

```json
{"input_tokens": 31652}
```

Tell Claude Code the real window with `CLAUDE_CODE_MAX_CONTEXT_TOKENS`
when the model's context is smaller than Claude's; it compacts against
that number instead of running into the 400.

## `role: "system"` inside `messages`

Claude Code sends a second system message *inside* `messages`, after the
first user turn (`Available agent types for the Agent tool: ...`), on
top of the top-level `system`. Chat templates accept exactly one system
message, at the start: Qwen's throws `System message must be at the
beginning` on the second one, and every turn was a 500.

The server folds every system-role message in `messages`, in order, onto
the end of the top-level `system` text (blank-line separated), so the
template sees one system message at position 0 and the rest of the
conversation unchanged. The same folding applies to `system` and
`developer` turns on `/v1/chat/completions` and `/v1/responses`.
`count_tokens` counts the folded prompt, so its number is the one
`/v1/messages` prefills.

## Prompt caching: `cache_control`

Claude Code marks `cache_control` breakpoints (`ttl: "1h"`) on its system
blocks and on the last message of every request. The server forwards
them to the engine's prefix cache as pinned cache points, so the
prefix a breakpoint closes stays resumable for its TTL (5 min or 1 h)
even after later turns change everything behind it. On a hybrid model
(Qwen3.5) that also keeps the recurrent-state snapshot there, without
which a cached prefix cannot be resumed at all.

Every breakpoint is located by rendering the prefix it closes with the
model's own chat template and checking it is a byte-prefix of the real
prompt; one that cannot be placed exactly is dropped, never guessed.
Cost: 25-40 ms per request on Claude Code's ~43k-token prompt.

Beyond the breakpoints the client wrote, the server also pins **the end
of the tool list**. Anthropic caches in the order tools, system,
messages, so that boundary is part of every breakpoint's prefix, and it
is where Claude Code's prompt actually changes: its claude.ai MCP
connectors finish connecting after the first request, and turn 2
arrives with ~6k more tokens of tools appended (the Qwen3.5 template
renders tools first). Measured on Qwen3.5-9B, M5 Pro, `claude -p`
asking one question, interleaved fresh-server arms:

| | turn-2 cached tokens | wall |
|---|---|---|
| neither | 0 of 43,500 | 98.0 s |
| engine snapshot ladder only (`TEMPO9_CACHE_CONTROL=0`) | 28,672 | 69.6 s |
| `cache_control` only (`AS_CPU_PREFIX_SNAPSHOT_LADDER=0`) | 31,872 | 63.9 s |
| both (default) | 30,464 of 42,176 | 63.0 s |

The end pin does not cover every race: in some runs turn 1 also
carries a transient built-in tool (`WaitForMcpServers`) that turn 2
drops from the *middle* of the list, ~2.2k tokens before its end. So the
server also pins the tool boundaries ~1k, 2k, 4k and 8k tokens before
the end of the list: log-spaced by distance rather than "the last N
tools", because where a transient tool sorts decides its index and
nothing decides its distance. A change x tokens before the end then
resumes from the first rung at or beyond x. Measured, two interleaved
arms each, turn-2 prefill:

| turn 1 differs from turn 2 by | rungs off | rungs on |
|---|---|---|
| `WaitForMcpServers`, 2.2k from the end | 28,672 cached, 22.15 s | 29,952 cached, 20.59 s |
| a transient tool ~6k from the end | 20,480 cached, 32.12 s | 23,808 cached, 28.55 s |
| tools appended only | 32,000 cached | same, +0.36 s per session |

The last row is the price: four more snapshots captured on turns 1 and
2 (none after), ~20 ms of rendering per request, 4 x 25 MB pinned for
the breakpoint's TTL. `TEMPO9_TOOL_LADDER=0` keeps only the end pin.

`TEMPO9_CACHE_CONTROL=0` turns the forwarding off;
`TEMPO9_CACHE_POINT_DEBUG=1` logs each resolved point with the tokens on
both sides of it (`[cache-points] ... tools@131574B->tok 32003 ttl 3600
["function\"}\n"|"</tools>"]`).

## What to expect: time per question, and which model

Measured on an M5 Pro (24 GB) with Q4_K_S weights, `--max-length 65536`,
and `claude -p` asked one question per session. Every session started its
own fresh server unless the row says otherwise. Two runs per cell.

| | Qwen3.5-9B | Gemma-4-E4B |
|---|---|---|
| first request: ~40k tokens from nothing | 51 s (~850 tok/s) | 24 s (~1,680 tok/s) |
| each later tool-call turn (cache hit) | 0.3-1.4 s prefill | 0.5-1 s prefill |
| "count the .txt files" (1 tool call), whole session | 58 / 65 s | 31 / 45 s |
| "find where `digest` is defined" (Grep + Read) | 114 / 62 s, both right | 40 s **gave up** / 63 s right |
| a later session on the **same** server, whole session | 10-23 s (first turn hits 28k-40k) | 26-27 s (first turn hits < 2.5k) |

- **The first question of a new server is the slow one.** Claude Code's
  system prompt and tool schemas come to about 40k tokens before you
  have typed anything, and they have to be prefilled once.
- **Pick Qwen3.5-9B for multi-step work, Gemma-4-E4B for speed.** In this
  small test Qwen answered 4 of 4 correctly. Gemma answered 3 of 4; on the
  fourth it searched eight times, never opened the file, and asked the
  user instead.
- **Why Gemma does not reuse the cache across sessions.** Gemma 4's chat
  template puts the system prompt *before* the tool declarations, and
  Claude Code's system prompt carries per-session text (the working
  directory, git status, a session-specific scratch path) about 2k tokens
  in. So every new session differs from the last one ahead of ~30k tokens
  of tools. Qwen3.5's template puts tools first, which is the order
  Anthropic's own cache uses, so a new session reuses the tools.
  Reordering Gemma's template would change what the model sees, so it is
  not done.
- **When the MCP connectors finish connecting late, turn 2 costs one more
  partial prefill.** Claude Code then adds a "MCP Server Instructions"
  section to the system prompt and appends the MCP tools. On Gemma that
  section lands ahead of the tool list: 22.7 s once per session. On Qwen
  it lands after the tools: 18-21 s once per session.

## Streaming errors are events, not silence

With `stream: true`, anything that can fail before generation -- the
template render above all -- now happens before a byte is on the wire,
so a failure is an ordinary `4xx`/`5xx` with a JSON body. A failure
*after* `message_start` (the engine interrupting a request, say) ends
the stream with Anthropic's own `error` event:

```
event: error
data: {"type":"error","error":{"type":"api_error","message":"tempo9 error 5: wait: ..."}}
```

which the SDKs raise as an exception. Before this, both cases were a
`200`, `message_start`, `content_block_start`, and a closed connection --
339 bytes that Claude Code read as an empty reply and retried.

## What works

- **Claude Code's own tools** (Read, Edit, Bash, Grep, …). They are sent
  as *custom* tools with an `input_schema`, the model calls them, Claude
  Code runs them. Nothing about the local model changes that loop.
- **Anthropic's client-executed built-ins** — `bash_*`, `text_editor_*`
  (`str_replace_editor` / `str_replace_based_edit_tool`), `computer_*`,
  `memory_*`. These carry no schema on the wire, because Anthropic bakes
  it into Claude. Tempo9 carries the published schema for each version it
  knows (`bash_20241022`, `bash_20250124`; `text_editor_20241022`,
  `text_editor_20250124`, `text_editor_20250429`, `text_editor_20250728`;
  `computer_20250124`, `computer_20251124`; `memory_20250818`), shows it
  to the local model, and the reply's `tool_use` block carries the name
  the client sent. The client executes it as it always did. An unknown
  version of one of these families is refused with 400 naming the
  versions the server does know. One difference for `memory`: Anthropic's
  API adds a memory-protocol instruction to the *system prompt* when the
  tool is present ("view your memory directory before doing anything
  else"); this server injects no system text for tools, so that
  instruction lives in the tool's description instead.
- **Thinking**, streaming with incremental `tool_use` blocks, and
  `count_tokens` via the tokenizer.

## What is unavailable in local mode: server-executed tools

Anthropic's **server-executed** tools — `web_search_*`, `web_fetch_*`,
`code_execution_*`, `tool_search_tool_*` — run on Anthropic's API server
between generations: the model emits `server_tool_use`, the server
answers with a `web_search_tool_result` block, and the model continues,
all inside one request. There is no server here to do that.

A request that carries such a tool is **refused as a whole with 400** —
never silently degraded to a reply without the tool. The exact shape:

```
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "tools[2] is a \"web_search_20260209\" tool; this server does not execute server-side tools -- remove it from tools. Supported tool types: custom (with input_schema), bash_*, text_editor_*, computer_*, memory_* (client-executed)."
  }
}
```

The index in `tools[2]` is the position in the request's `tools` array,
so a client can find and drop the offending entry. The same rule, with
the OpenAI error shape (`error.param` = `tools[2].type`), applies to
`web_search`, `file_search` and `code_interpreter` on `/v1/responses`
and `/v1/chat/completions`.

Why refuse rather than drop: a request that asked for web search and got
a reply from a model that never searched is a wrong answer that looks
like a right one.

### What this means for Claude Code: nothing, as measured

Claude Code does **not** send these types to a custom `ANTHROPIC_BASE_URL`.
Captured from Claude Code 2.1.239 against this server (2026-09-06): all
29 tools in its request are `custom` tools with an `input_schema` —
including **`WebSearch` and `WebFetch`**, which arrive as ordinary
client-side tools (`query` required). The model calls them like any other
tool; Claude Code runs them itself. Whether Claude Code's own WebSearch
implementation works without Anthropic's API behind it is a Claude Code
question, not a server one — in the captured run it answered its own
call with "Claude requested permissions to use WebSearch, but you haven't
granted it yet" (a `-p` run without `--allowedTools WebSearch`).

So the 400 above is for other Anthropic-SDK clients that declare
`web_search_20260209` and friends. A Claude Code turn is never refused
for this reason.

Two things the same run showed that are worth knowing before blaming the
server:

- **Empty tool calls loop.** A 9B model occasionally emits
  `WebSearch` with `input: {}`; Claude Code answers "required parameter
  `query` is missing", the model tries again, and a `-p` run without
  `--max-turns` will do this indefinitely (69 rounds observed before it
  was killed). The server streamed a correct `input_json_delta` in every
  replay of that request — it is sampling variance in the model, so cap
  turns and prefer a larger model for tool-heavy work.
- **The tools cost ~32k tokens before you type anything.** Claude Code's
  29 tool schemas plus its system prompt measured 31,839 tokens; see
  [Context length](#context-length) for what that means for
  `--max-length`.

### An explicit escape hatch: `TEMPO9_STRIP_UNSUPPORTED_TOOLS=1`

Set this on the **server** to strip server-executed tools from a request
instead of refusing it. It is off by default; an environment variable set
on purpose is not a silent downgrade. When it is on:

- each stripped tool prints one line to the server log:
  `[tempo9] warning: tools[2] (web_search_20260209) was stripped from tools: this server does not execute server-side tools (TEMPO9_STRIP_UNSUPPORTED_TOOLS=1)`
- the reply carries the same lines in a top-level `warnings` array — on
  the message object for a non-streaming `/v1/messages`, on the
  `message_start` event when streaming, on the final usage chunk for chat
  completions, and on the response object for `/v1/responses`.

It strips **only** server-executed tools. An unknown version of a
client-executed family, or a client *toolset* (`computer_toolset_*`,
`browser_toolset_*` — one entry declaring a fixed set of member tools,
a different shape from a single built-in, and not tabled here), is still
a 400 — there is nothing known to strip there.

## Check what the server supports: `GET /v1/capabilities`

```
curl -s http://127.0.0.1:11435/v1/capabilities | jq .
```

```json
{
  "object": "capabilities",
  "model": "qwen3-8b-q4ks",
  "protocols": [
    {
      "id": "anthropic_messages",
      "routes": ["POST /v1/messages", "POST /v1/messages/count_tokens"],
      "tool_types": ["custom", "bash_20241022", "bash_20250124", "computer_20250124", "computer_20251124", "memory_20250818", "text_editor_20241022", "…"],
      "refused": [
        {"param": "tool_choice", "reason": "only \"auto\" or absent: …"},
        {"param": "stop_sequences", "reason": "the engine stops on token ids only; …"},
        {"param": "tools[].type", "reason": "server-executed tools (web_search_*, web_fetch_*, code_execution_*, tool_search_*) are not executed here; …"},
        {"param": "messages[].content[].type", "reason": "only text, tool_use and tool_result blocks; …"}
      ],
      "implemented": [
        {"feature": "count_tokens", "how": "the real tokenizer over the prompt /v1/messages would prefill"},
        {"feature": "…"}
      ]
    },
    {"id": "chat_completions", "…": "…"},
    {"id": "responses", "…": "…"},
    {"id": "ollama", "…": "…"}
  ],
  "unsupported_tools": {
    "policy": "a request carrying a tool this server cannot run is refused as a whole with 400 naming tools[i].type; nothing is silently dropped",
    "server_executed": ["advisor_*", "code_execution_*", "mcp_toolset_*", "tool_search_tool_bm25_*", "tool_search_tool_regex_*", "web_fetch_*", "web_search_*", "code_interpreter", "file_search", "…"],
    "client_executed_passthrough": ["bash_20241022", "bash_20250124", "…"],
    "strip_env": "TEMPO9_STRIP_UNSUPPORTED_TOOLS",
    "strip_enabled": false
  }
}
```

The document is a hand-maintained struct that lives beside the refusal
code, and a test throws every refusal the parsers know and checks it
against the document in both directions, so it cannot drift silently.
`GET /v1/models` (and `/v1/models/{id}`) carries a pointer to it under
`tempo9.capabilities_url`, for clients that only read that.

## Not implemented, by decision (2026-09-06)

Server-side tool execution is a real feature that Tempo9 deliberately
does not build now. Refusing is honest and cheap; a half-built version
that answers 200 without searching is the failure this page exists to
prevent.

If it is ever built, this is the shape it takes, so that the decision
can be revisited rather than rediscovered:

1. **A search backend the user brings (BYOK).** The local server does
   not reach the network on the user's behalf today, and must not start
   doing so implicitly. Web search would run against a search API the
   user configures with their own key, off by default; web fetch would
   fetch only URLs already present in the conversation, as Anthropic's
   does, and against the same allow/deny lists.
2. **Synthesised blocks in Anthropic's shape.** When the model calls the
   tool, the server runs the search, then emits a `server_tool_use`
   block followed by a `web_search_tool_result` block (a list of
   `web_search_result` objects with `url`, `title`, `encrypted_content`
   replaced by the page text, and `page_age`), exactly as the Anthropic
   API does, so Claude Code needs no change.
3. **A second generation.** The result blocks are appended to the
   assistant turn and the model continues in the same request — one
   client request, two (or more) model generations, all under the same
   `max_tokens` and `max_uses` accounting, with `pause_turn` when the
   budget runs out mid-loop.

Code execution is the same pattern with a sandbox instead of a search
backend, and is further out: a sandbox on the user's machine is a
security decision, not a serving feature.

Until then: `GET /v1/capabilities` says what is served, and a 400 says
what is not.
