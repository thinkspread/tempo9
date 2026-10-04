# Getting started

## Run a model

```
tempo9 --hf unsloth/Qwen3.5-9B-GGUF:Q4_K_S
```

The first run downloads the 5.4 GB file into the Hugging Face cache, and
later runs start from that copy. A GGUF you already have works the same way:
`tempo9 --gguf ~/models/qwen3-8b-q4ks.gguf`.

That is the whole setup. There is no configuration file, no daemon to
install, and no model conversion step — Tempo9 loads GGUF directly.

The server listens on `http://127.0.0.1:11435`, **loopback only**. The
wildcard address is never bound, so nothing off this machine can reach it.

## First request

```
curl http://127.0.0.1:11435/v1/chat/completions \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}]}'
```

No API key is required and none is checked. That is safe only because of the
loopback bind — see [Limits](limits.md#no-authentication) before you put it
behind a tunnel or a reverse proxy.

## If you already use Ollama

You do not need to download anything:

```
tempo9 --list-ollama
tempo9 --ollama qwen3:8b
```

See [Ollama models](ollama-models.md) for how that works and what it does
not touch.

## Where to go next

- Wiring up an app you already use: [Connect an app](connect-apps.md)
- Every flag: [CLI reference](cli.md)
- Every route: [API reference](api.md)
