#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Server-sent events: each line is "data: {...}", and the stream ends with
# "data: [DONE]".  The Ollama dialect does NOT work this way -- see
# ollama-chat.sh.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"
curl -sN "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "Count 1 to 5."}],
  "max_tokens": 64,
  "temperature": 0,
  "stream": true
}'
