#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Gemma 4 12B — a second architecture, and an honest gap.
#
# Start the server in another terminal:
#
#   tempo9 --gguf ~/models/gemma4-12b-it-q4ks.gguf
#
# TEXT WORKS.  IMAGES DO NOT, on this path, and the reason is worth stating
# plainly rather than leaving you to discover it:
#
#   gemma4-12b-mmproj-f16.gguf exists beside the model, but a GGUF mmproj is
#   NOT what --tower takes.  The vision front end is Core ML and needs an
#   exported tower directory containing tower_meta.json.  No Gemma tower is
#   exported here, so Gemma runs text-only through tempo9.
#
# A GGUF projector is what a tower is exported FROM, offline — it is not a
# drop-in.  Same reason an Ollama model carrying a projector runs text-only.
#
# This script only sends requests; it never starts an engine.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"

echo '--- capabilities (expect no "vision" here) ---'
curl -s "$B/api/show" -H 'content-type: application/json' -d '{}' \
  | python3 -c 'import json,sys; print(" ", json.load(sys.stdin)["capabilities"])'

echo '--- text ---'
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "Name three prime numbers over 100."}],
  "max_tokens": 64, "temperature": 0
}'; echo

echo '--- a long-context request, which is where paged KV matters ---'
# Gemma 4 12B at q4ks is a comfortable fit; the interesting property is that
# the KV cache is paged rather than preallocated to max_length, so a long
# prompt costs what it uses.
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\": \"local\",
  \"messages\": [{\"role\": \"user\", \"content\": \"$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 200))Summarise the preceding text in one sentence.\"}],
  \"max_tokens\": 64, \"temperature\": 0
}"; echo
