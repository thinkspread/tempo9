#!/usr/bin/env python3
# Copyright (c) 2026 Jiejing Zhang.
"""Talk to a local Tempo9 with nothing installed.

The standard library is enough: the server speaks plain HTTP and JSON, so a
client needs no SDK.  Useful when you want to check the server is behaving
before blaming a framework.

    python3 examples/python/no_deps.py
    TEMPO9=http://127.0.0.1:11440 python3 examples/python/no_deps.py
"""
import json
import os
import urllib.request

BASE = os.environ.get("TEMPO9", "http://127.0.0.1:11435")


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def stream(path, body):
    """Yield content deltas from an SSE response."""
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps({**body, "stream": True}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            # Blank lines separate events; [DONE] ends the stream.
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            choices = json.loads(payload).get("choices") or [{}]
            piece = choices[0].get("delta", {}).get("content")
            if piece:
                yield piece


if __name__ == "__main__":
    models = json.load(urllib.request.urlopen(BASE + "/v1/models"))
    name = models["data"][0]["id"]
    print(f"model: {name}\n")

    reply = post("/v1/chat/completions", {
        "model": name,
        "messages": [{"role": "user", "content": "Name one color."}],
        "max_tokens": 32,
        "temperature": 0,
    })
    print("blocking:", reply["choices"][0]["message"]["content"])

    print("streaming: ", end="", flush=True)
    for piece in stream("/v1/chat/completions", {
        "model": name,
        "messages": [{"role": "user", "content": "Count 1 to 5."}],
        "max_tokens": 64,
        "temperature": 0,
    }):
        print(piece, end="", flush=True)
    print()
