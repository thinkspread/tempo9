#!/usr/bin/env python
# Copyright (c) 2026 Jiejing Zhang.
"""Gate the Swift chat-template renderer against transformers, string for string.

The template is the last per-request piece that tied a native host to Python.
It is not reimplemented -- huggingface/swift-jinja renders it -- so what needs
proving is that the *same template text*, pulled out of the .gguf, produces
the same prompt as `tokenizer.apply_chat_template`.

Exact equality, because the failure mode here is quiet: a prompt that is
almost right still looks like a prompt. One missing newline before
`<|im_start|>assistant` shifts every position the model was trained on and
shows up only as slightly worse answers.

The cases are chosen to hit the branches the Qwen3.5 template actually has:
thinking on/off, tools, multimodal content parts, tool-call round trips,
system messages, and generation-prompt on/off.

Usage:
  .venv/bin/python tools/check_gguf_chat_template_parity.py
"""
import argparse
import difflib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--gguf", default=os.path.expanduser(
    "~/models/qwen35-0.8b-q8_0.gguf"))
parser.add_argument("--tokenizer", default=os.path.expanduser(
    "~/models/Qwen3.5-0.8B"))
parser.add_argument("--ggufctl", default=os.path.join(
    REPO, ".build/release/ggufctl"))
parser.add_argument("--show", type=int, default=3)
args = parser.parse_args()

if not os.path.exists(args.ggufctl):
    sys.exit("ggufctl not built: swift build -c release --product ggufctl")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

CASES = [
    ("plain", {"messages": [{"role": "user", "content": "你好"}]}),
    ("english", {"messages": [{"role": "user", "content": "Hello there!"}]}),
    ("no-generation-prompt", {
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": False}),
    ("system", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"}]}),
    ("multi-turn", {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "and +1?"}]}),
    ("thinking-on", {
        "messages": [{"role": "user", "content": "think about it"}],
        "enable_thinking": True}),
    ("thinking-off", {
        "messages": [{"role": "user", "content": "do not think"}],
        "enable_thinking": False}),
    ("multimodal", {"messages": [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "描述这张图"}]}]}),
    ("multimodal-two-images", {"messages": [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": "compare them"}]}]}),
    ("tools", {
        "messages": [{"role": "user", "content": "weather in Hangzhou?"}],
        "tools": [WEATHER_TOOL]}),
    ("tools-with-system", {
        "messages": [
            {"role": "system", "content": "Use tools when useful."},
            {"role": "user", "content": "weather in Hangzhou?"}],
        "tools": [WEATHER_TOOL]}),
    ("tool-round-trip", {
        "messages": [
            {"role": "user", "content": "weather in Hangzhou?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function",
                "function": {"name": "get_weather",
                             "arguments": {"city": "Hangzhou"}}}]},
            {"role": "tool", "content": "22C, sunny"}],
        "tools": [WEATHER_TOOL]}),
    ("assistant-prefill", {"messages": [
        {"role": "user", "content": "count"},
        {"role": "assistant", "content": "1, 2,"}],
        "add_generation_prompt": False}),
    ("empty-content", {"messages": [{"role": "user", "content": ""}]}),
    ("newlines", {"messages": [
        {"role": "user", "content": "line1\nline2\n\nline4"}]}),
]

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)


def normalize(case):
    """Both sides must receive exactly the same context.

    apply_chat_template defaults add_generation_prompt to False and a serving
    stack always wants True, so leaving it implicit means the two renderers
    are asked different questions -- which shows up as a wall of identical
    "missing assistant header" diffs that say nothing about the renderer.
    """
    payload = dict(case)
    payload.setdefault("add_generation_prompt", True)
    return payload


def hf_render(case):
    payload = dict(case)
    messages = payload.pop("messages")
    return tok.apply_chat_template(messages, tokenize=False, **payload)


def swift_render(cases):
    payload = "\n".join(json.dumps(c, ensure_ascii=False) for _, c in cases) + "\n"
    result = subprocess.run(
        [args.ggufctl, "render", "--gguf", args.gguf, "--batch"],
        input=payload, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit("ggufctl render failed: %s" % result.stderr.strip())
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    if len(lines) != len(cases):
        sys.exit("ggufctl returned %d renders for %d cases"
                 % (len(lines), len(cases)))
    return [json.loads(l)[0] for l in lines]


expected, skipped = [], []
for name, raw in CASES:
    case = normalize(raw)
    try:
        expected.append((name, case, hf_render(case)))
    except Exception as e:
        # A case transformers itself rejects proves nothing about the port.
        skipped.append((name, str(e).splitlines()[0][:90]))

usable = [(name, case) for name, case, _ in expected]
got = swift_render(usable)

failures = []
for (name, _case, want), have in zip(expected, got):
    if want != have:
        failures.append((name, want, have))

print("%d cases rendered, %d skipped, %d mismatch(es)"
      % (len(expected), len(skipped), len(failures)))
for name, why in skipped:
    print("  skipped %-22s %s" % (name, why))

for name, want, have in failures[:args.show]:
    print("\n  === %s ===" % name)
    diff = difflib.unified_diff(want.splitlines(keepends=True),
                                have.splitlines(keepends=True),
                                fromfile="transformers", tofile="swift",
                                n=2)
    sys.stdout.writelines(diff)
    if want.rstrip() == have.rstrip():
        print("  (differs only in trailing whitespace: %r vs %r)"
              % (want[len(want.rstrip()):], have[len(have.rstrip()):]))
if len(failures) > args.show:
    print("\n  ... %d more" % (len(failures) - args.show))

if failures:
    sys.exit("\nchat template parity FAILED")
print("\nchat template parity OK -- byte-identical to apply_chat_template")
