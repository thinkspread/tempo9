#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# The Anthropic Messages dialect, which is what Claude Code speaks.  Same
# engine, same model, different wire shape: "system" is a top-level field and
# the reply is a content-block array.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"
curl -s "$B/v1/messages" -H 'content-type: application/json' -d '{
  "model": "local",
  "system": "Answer in exactly one word.",
  "messages": [{"role": "user", "content": "Capital of France?"}],
  "max_tokens": 32
}'
echo
