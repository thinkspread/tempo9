#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Qwen3.8 27B — dense, and the interesting contrast with the 35B.
#
# Start the server in another terminal:
#
#   tempo9 --gguf ~/models/qwen38-27b-q4ks.gguf --max-batch 16
#
# With vision, if you have the tower exported:
#
#   tempo9 --gguf ~/models/qwen38-27b-q4ks.gguf \
#          --tower ~/models/vlm_towers/Qwen3.8-27B/coreml --max-batch 16
#
# ONE AT A TIME.  Weights are 14.74 GiB; on a 24 GB machine this and the 35B
# cannot both be resident, and trying drags the whole system into swap.
#
# WHY THIS MODEL HAS ITS OWN EXAMPLE
#
# Put it beside qwen35-35b-a3b.sh.  The two occupy almost the same space --
# 14.74 GiB against 15.99 GiB -- and decode completely differently:
#
#   27B   dense    every token touches all 27B of weights
#   35B   A3B MoE  every token touches ~3B of active weights
#
# So the "smaller" model is usually the SLOWER one per token, which is not
# what the parameter count suggests.  This script measures rather than
# asserts it; run both and compare.  (curl and date are a demonstration, not
# an instrument -- use the bench harness for a number worth quoting.)
#
# This script only sends requests; it never starts an engine.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"

measure() {  # $1 = prompt, $2 = max_tokens
  start=$(date +%s)
  out=$(curl -s "$B/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"local\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],
         \"max_tokens\":$2,\"temperature\":0}")
  el=$(( $(date +%s) - start ))
  echo "$out" | ELAPSED="$el" python3 -c '
import json, os, sys
d = json.load(sys.stdin)
u = d.get("usage", {})
n = u.get("completion_tokens", 0)
el = max(1, int(os.environ["ELAPSED"]))
text = d["choices"][0]["message"]["content"].replace("\n", " ")
print(f"  {n} tokens in ~{el}s  (~{n / el:.1f} tok/s, wall-clock)")
print(f"  {text[:72]}...")'
}

echo '--- decode speed, dense ---'
# Compare this figure with the same call in qwen35-35b-a3b.sh.  Nearly the
# same weights on disk; the MoE should come out ahead per token.
measure "Explain what a memory barrier does, in about 120 words." 192

echo
echo '--- long context: 10k tokens of prompt ---'
# LONG10K is a support-matrix column for a reason.  The KV cache is paged
# rather than preallocated to max_length, so a long prompt costs what it
# actually uses instead of reserving for the worst case.
FILLER=$(printf 'Section %s discusses distributed consensus and its failure modes. %.0s' $(seq 1 400))
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\": \"local\",
  \"messages\": [{\"role\": \"user\", \"content\": \"$FILLER Question: how many sections were listed above? Answer with a number.\"}],
  \"max_tokens\": 32, \"temperature\": 0
}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
u = d.get("usage", {})
print(f"  prompt {u.get(\"prompt_tokens\", 0)} tokens -> {d[\"choices\"][0][\"message\"][\"content\"].strip()!r}")'

echo
echo '--- vision, if the server was started with --tower ---'
# Unlike Gemma (see gemma4-12b.sh), a Core ML tower for this model IS
# exported here, so images work -- provided the server was started with it.
curl -s "$B/api/show" -H 'content-type: application/json' -d '{}' \
  | python3 -c '
import json, sys
caps = json.load(sys.stdin)["capabilities"]
print(f"  capabilities: {caps}")
print("  no \"vision\" -> restart with --tower ~/models/vlm_towers/Qwen3.8-27B/coreml"
      if "vision" not in caps else "  vision is live; see qwen3vl-4b.sh for the request shapes")'

echo
echo 'Note: this model has no separate thinking mode — the support matrix'
echo 'records THINK as not applicable, so enable_thinking does nothing here.'
echo 'For that, see qwen35-4b.sh.'
