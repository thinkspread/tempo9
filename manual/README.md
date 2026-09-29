# Tempo9 manual

User documentation. If you are running Tempo9 rather than building it, this
is the directory you want.

| Page | Read it when |
|---|---|
| [Getting started](getting-started.md) | You have a model file and want a server |
| [Models and quantizations](models.md) | You want to know if YOUR file will run |
| [Connect an app](connect-apps.md) | You have an app and want it to use Tempo9 |
| [Claude Code](claude-code.md) | You want Claude Code on a local model, and to know what WebSearch does there |
| [Ollama models](ollama-models.md) | You already pulled models with Ollama |
| [CLI reference](cli.md) | You want every flag |
| [API reference](api.md) | You are writing a client |
| [Limits](limits.md) | Before you design around something we do not do |

Engineering notes — investigations, benchmark method, internal design — live
in [`../docs/`](../docs/) and are not user documentation. The split is
deliberate: a page that explains how to connect an app and a page that
records why a crash happened have different readers, and mixing them makes
both harder to find.
