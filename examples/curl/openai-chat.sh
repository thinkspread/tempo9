#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# The plainest request there is. No API key: the server is loopback-only and
# checks nothing (see ../../manual/limits.md).
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "Name one color."}],
  "max_tokens": 32,
  "temperature": 0
}'
echo
