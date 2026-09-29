#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Qwen3.5 4B — the everyday model. Small enough to keep resident, good enough
# for tool calling.
#
# Start the server in another terminal:
#
#   tempo9 --gguf ~/models/qwen35-4b-q4ks.gguf
#
# This script only sends requests; it never starts an engine.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"

echo '--- what is loaded ---'
curl -s "$B/v1/models"; echo

echo '--- a plain answer ---'
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "In one sentence: what is a mutex?"}],
  "max_tokens": 96, "temperature": 0
}'; echo

echo '--- thinking, which this family supports ---'
# enable_thinking is ours: the reasoning trace is split out rather than left
# inline for the caller to strip.
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "If 3 machines take 3 minutes to make 3 parts, how long for 100 machines to make 100 parts?"}],
  "enable_thinking": true,
  "max_tokens": 512, "temperature": 0
}'; echo
