#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Ollama's native route.  Three differences from the OpenAI dialect, each of
# which fails SILENTLY against a client that assumes otherwise:
#   1. streaming is newline-delimited JSON -- no "data: ", no [DONE]
#   2. "stream" defaults to TRUE, so this asks for false explicitly
#   3. tool arguments are objects, not JSON strings
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"
echo '--- non-streaming ---'
curl -s "$B/api/chat" -H 'content-type: application/json' -d '{
  "model": "local",
  "stream": false,
  "messages": [{"role": "user", "content": "Name one color."}],
  "options": {"temperature": 0, "num_predict": 32}
}'
echo
echo '--- streaming (one JSON object per line, last has done:true) ---'
curl -sN "$B/api/chat" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "Count 1 to 5."}],
  "options": {"temperature": 0, "num_predict": 48}
}'
