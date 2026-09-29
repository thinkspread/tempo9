# Limits

Things worth knowing before you design around them.

## No embeddings

`/v1/embeddings` is not served and `/api/embed` answers `501` with a message
saying so. If your pipeline needs embeddings, get them elsewhere.

## One model per process

`/v1/models` and `/api/tags` report the single loaded model. To serve a
second model, run a second `tempo9` on another port.

## Vision needs a Core ML tower

Image input works when the server is started with `--tower <dir>`, where
*dir* is the directory containing `tower_meta.json` — the `coreml/`
**subdirectory** of an exported tower, not its parent:

```
--tower ~/models/qwen3vl_tower/coreml     # correct
--tower ~/models/qwen3vl_tower            # missingAsset: .../tower_meta.json
--tower ~/models/…/model-mmproj.gguf      # same error — a GGUF projector is not a tower
```

A GGUF mmproj is what a tower is exported *from*, offline; it is not a
drop-in. A GGUF
vision projector inside an Ollama model **is detected but not loaded** — the
server prints a line saying so rather than leaving you to wonder why images
do nothing.

## No server-side tools

WebSearch, WebFetch and code execution are tools the vendor's servers run
between generations. Nothing here runs them, so a request carrying one
is refused as a whole with `400` naming the tool — not answered by a
model that never searched. Client-executed built-ins (`bash_*`,
`text_editor_*`, `computer_*`, `memory_*`) work. `GET /v1/capabilities` lists both.
Details, the exact error, and the opt-in strip switch:
[Claude Code](claude-code.md).

## No authentication

Nothing checks a key, and there is no flag to make it. This is defensible
only because the listener binds `127.0.0.1` explicitly and never the
wildcard, so the server is unreachable from other machines.

**Do not put it behind a tunnel, a reverse proxy, or a port forward and
leave it that way.** That removes the only thing protecting it.

## Ollama's store is read, never written

Using `--ollama` opens Ollama's blobs in place. Nothing is copied,
converted, moved, or written into their store. The one file Tempo9 creates
is a symlink in **its own** cache — see
[Ollama models](ollama-models.md#why-a-symlink).
