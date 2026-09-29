# Security

Copyright (c) 2026 Jiejing Zhang.

## Reporting a vulnerability

Please do not open a public issue. Report it privately, through this
repository's Security tab ("Report a vulnerability"), or by email to
jiejing@thinkspread.com with "Tempo9 security" in the subject. Include what
you ran, what you sent, and what happened.

Reports about the engine binary are welcome the same way, even though its
source is not in this repository.

## What the server exposes

`tempo9` has no authentication: nothing checks a key, and there is no flag to
make it. That is defensible only because it listens on `127.0.0.1` and never
on a wildcard address, so other machines cannot reach it. Any process on the
same Mac can, while it runs.

Do not put it behind a tunnel, a reverse proxy or a port forward without
putting authentication in front of it. See
[manual/limits.md](manual/limits.md#no-authentication).

An app that embeds the server through the SDK can require a bearer token:
`OpenAIServer` takes a `token`, and then refuses requests without it.
