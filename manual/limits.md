# Limits

Things worth knowing before you design around them.

## No embeddings

`/v1/embeddings` is not served and `/api/embed` answers `501` with a message
saying so. If your pipeline needs embeddings, get them elsewhere.

## One model per process

`/v1/models` and `/api/tags` report the single loaded model. To serve a
second model, run a second `tempo9` on another port.

## Images need a vision encoder

A GGUF holds the language model. Answering about an image also takes the
model's vision encoder, and where that comes from depends on the family:

- **Gemma 4 12B**: the mmproj `.gguf` *is* the encoder. It is picked up
  without a flag when it sits beside the model or in the `--hf` repo:
  `tempo9 --hf thinkspread/gemma-4-12b-it-Q4_K_S-GGUF`. Other Gemma 4
  projectors, E4B's for one, contain a full vision encoder this server does
  not run; it reads the projector's header, says so, and serves text
  without downloading the rest of the file.
- **Qwen3.5, Qwen3.6, Qwen3.8, Qwen3-VL**: the encoder runs as an exported
  Core ML tower. These are published, and `--tower auto` picks the right one
  from the GGUF's own header, whatever its quantization:

| model | `--tower` |
|---|---|
| Qwen3.5-9B | `thinkspread/Qwen3.5-9B-Vision-CoreML-fp16` |
| Qwen3.5-35B-A3B | `thinkspread/Qwen3.5-35B-A3B-Vision-CoreML-fp16` |
| Qwen3.6-35B-A3B | `thinkspread/Qwen3.6-35B-A3B-Vision-CoreML-fp16` |
| Qwen3.8-27B | `thinkspread/Qwen3.8-27B-Vision-CoreML-fp16` |
| Qwen3-VL-4B-Instruct | `thinkspread/Qwen3-VL-4B-Instruct-Vision-CoreML-fp16` |
| Qwen3-VL-30B-A3B-Instruct | `thinkspread/Qwen3-VL-30B-A3B-Instruct-Vision-CoreML-fp16` |

```
tempo9 --hf unsloth/Qwen3.5-9B-GGUF:Q4_K_S --tower auto
```

A local directory works too: the one containing `tower_meta.json`, which
for an export is its `coreml/` **subdirectory**, not the parent.

The encoder is checked against the model before the server listens: its
output width against the model's embedding width, and its family against
the GGUF architecture (a Qwen3.5 encoder and a Qwen3-VL one are both 2048
wide for the 35B and 30B models). A mismatch stops the server and names
the right encoder. Two encoders of the same family and width cannot be
told apart this way: the Qwen3.5-35B-A3B encoder loads with Qwen3.6-35B-A3B,
whose own is in the table.

A Qwen mmproj `.gguf` is not usable here: it is what a tower is exported
*from*, offline. A GGUF vision projector inside an Ollama model **is
detected but not loaded** — the server prints a line saying so rather than
leaving you to wonder why images do nothing.

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
