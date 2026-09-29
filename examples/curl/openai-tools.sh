#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Tool calling.  Note the schema: "days" is declared integer, and the
# arguments come back typed rather than as strings -- getting that wrong is
# what an evaluator scores as a type error.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "Weather in Paris for 3 days? Use the tool."}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {"type": "object",
      "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
      "required": ["city"]}}}],
  "max_tokens": 128,
  "temperature": 0
}'
echo
