#!/bin/sh
# Copyright (c) 2026 Jiejing Zhang.
#
# Qwen3-VL 4B — images.
#
# Start the server in another terminal.  --tower must point at the directory
# holding tower_meta.json, which is the coreml/ SUBDIRECTORY of an exported
# tower.  Pointing at the parent, or at a GGUF mmproj, fails with
# VisionTowerError.missingAsset(".../tower_meta.json"):
#
#   tempo9 --gguf ~/models/qwen3vl-4b-q4km.gguf \
#          --tower ~/models/qwen3vl_tower/coreml
#
# Without --tower the server still runs; it just refuses image parts, and
# says why rather than saying "unsupported".
#
# This script only sends requests; it never starts an engine.
set -e
B="${TEMPO9:-http://127.0.0.1:11435}"
# No image ships with this repository -- point IMG at one of your own.
IMG="${IMG:-}"

[ -n "$IMG" ] && [ -f "$IMG" ] || {
  echo "set IMG=/path/to/an/image.png   (any PNG or JPEG the model should look at)" >&2
  exit 2
}
DATA="data:image/png;base64,$(base64 -i "$IMG" | tr -d '\n')"

echo '--- does this build have vision? ---'
# capabilities is not decoration: a client reads it to decide whether to
# offer image attachment at all.  "vision" appears only with a loaded tower.
curl -s "$B/api/show" -H 'content-type: application/json' -d '{}' \
  | python3 -c 'import json,sys; print(" ", json.load(sys.stdin)["capabilities"])'

echo '--- OpenAI shape: typed content parts with a data: URI ---'
curl -s "$B/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\": \"local\",
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"What colour is the rectangle? One word.\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"$DATA\"}}
  ]}],
  \"max_tokens\": 32, \"temperature\": 0
}"; echo

echo '--- Ollama shape: bare base64 in images[] ---'
# Tempo9 rewrites this into the typed parts above, so the same tower serves
# both dialects and the same refusals explain themselves the same way.
curl -s "$B/api/chat" -H 'content-type: application/json' -d "{
  \"model\": \"local\", \"stream\": false,
  \"messages\": [{\"role\": \"user\",
    \"content\": \"What colour is the rectangle? One word.\",
    \"images\": [\"$(base64 -i "$IMG" | tr -d '\n')\"]}],
  \"options\": {\"temperature\": 0, \"num_predict\": 32}
}"; echo

echo
echo 'Note: http(s) image URLs are refused on purpose. Fetching one would'
echo 'make an offline assistant reach the network to answer a question, and'
echo 'tell the host what this machine was asked to look at.'
