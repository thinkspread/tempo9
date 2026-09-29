#!/usr/bin/env python3
# Copyright (c) 2026 Jiejing Zhang.
"""The official OpenAI SDK, pointed at a local server.

This is the whole integration story for most Python code: change base_url,
pass any api_key (nothing checks it), and everything else is unchanged.

    pip install openai
    python3 examples/python/openai_sdk.py
"""
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("TEMPO9", "http://127.0.0.1:11435") + "/v1",
    # Required by the SDK, ignored by the server -- it is loopback-only and
    # authenticates nothing.  See manual/limits.md.
    api_key="not-used",
)

MODEL = client.models.list().data[0].id
print(f"model: {MODEL}\n")

reply = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Name one color."}],
    max_tokens=32,
    temperature=0,
)
print("blocking:", reply.choices[0].message.content)

print("streaming: ", end="", flush=True)
for chunk in client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count 1 to 5."}],
    max_tokens=64,
    temperature=0,
    stream=True,
):
    piece = chunk.choices[0].delta.content if chunk.choices else None
    if piece:
        print(piece, end="", flush=True)
print()

# Tool calling.  The schema declares `days` as an integer, and the arguments
# arrive typed -- a server that returns every argument as a string is what an
# evaluator scores as a type error.
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}]
call = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user",
               "content": "Weather in Paris for 3 days? Use the tool."}],
    tools=tools,
    max_tokens=128,
    temperature=0,
).choices[0].message.tool_calls
print("tool_calls:", call[0].function.arguments if call else None)
