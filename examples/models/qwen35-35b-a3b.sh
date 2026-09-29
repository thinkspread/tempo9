#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Qwen3.5 35B-A3B — and the reason the architecture exists.
#
# Start the server in another terminal.  --max-batch is what lets the engine
# schedule requests together; at 1 the server is serial end to end:
#
#   tempo9 --gguf ~/models/qwen35-35b-a3b-q3km.gguf --max-batch 16
#
# A single stream is the least interesting way to measure this engine.  The
# claim worth checking is that eight at once does not collapse: per-request
# latency rises, but total throughput goes UP, where a server that handles
# one request at a time would divide the same throughput eight ways.
#
# This script only sends requests; it never starts an engine.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"

ask() {  # $1 = topic, $2 = "quiet" to discard output
  body="{\"model\":\"local\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Explain $1 in two sentences.\"}],
    \"max_tokens\":160,\"temperature\":0}"
  if [ "$2" = quiet ]; then
    curl -s "$B/v1/chat/completions" -H 'content-type: application/json' \
      -d "$body" > /dev/null
  else
    curl -s "$B/v1/chat/completions" -H 'content-type: application/json' \
      -d "$body" | python3 -c '
import json, sys
d = json.load(sys.stdin)
u = d.get("usage", {})
text = d["choices"][0]["message"]["content"].replace("\n", " ")
print(f"  {u.get(\"completion_tokens\", 0)} tokens | {text[:64]}...")'
  fi
}

TOPICS="a_mutex a_deadlock a_race_condition a_spinlock \
a_condition_variable a_barrier a_read-write_lock a_futex"

echo '--- one request ---'
start=$(date +%s)
ask "a mutex"
echo "  wall: $(( $(date +%s) - start ))s"

echo '--- eight at once ---'
start=$(date +%s)
for t in $TOPICS; do
  ask "$(echo "$t" | tr '_' ' ')" quiet &
done
wait
echo "  wall: $(( $(date +%s) - start ))s for 8"

echo
echo 'If the second number is far below 8x the first, requests are sharing'
echo 'the engine rather than queueing behind each other.'
echo 'For a real measurement use the bench harness — curl and date are a'
echo 'demonstration, not an instrument.'
