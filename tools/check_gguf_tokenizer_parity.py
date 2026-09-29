#!/usr/bin/env python
# Copyright (c) 2026 Jiejing Zhang.
"""Gate the Swift GGUF tokenizer against HuggingFace, token id for token id.

The tokenizer is the piece a host cannot cache its way around: a model graph
is derived once per file, but tokenization runs on every request, so as long
as it needs Python the process does too. Everything it needs -- vocabulary,
merges, token types, the chat template -- is already inside the .gguf.

Failure here is silent by construction. A wrong pre-tokenizer regex or merge
order still produces *valid* token ids, just different ones, and the model
answers a bit worse for reasons nothing reports. So the bar is exact
equality, on inputs picked to hit the places byte-level BPE actually breaks:
whitespace runs, contractions, CJK, emoji (multi-byte, surrogate pairs),
combining marks, digits, and the special tokens that must never be merged.

Usage:
  .venv/bin/python tools/check_gguf_tokenizer_parity.py \
      [--gguf ~/models/qwen35-0.8b-q8_0.gguf] \
      [--tokenizer ~/models/Qwen3.5-0.8B]
"""
import argparse
import json
import os
import subprocess
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--gguf", default=os.path.expanduser(
    "~/models/qwen35-0.8b-q8_0.gguf"))
parser.add_argument("--tokenizer", default=os.path.expanduser(
    "~/models/Qwen3.5-0.8B"))
parser.add_argument("--ggufctl", default=os.path.join(
    REPO, ".build/release/ggufctl"))
parser.add_argument("--show", type=int, default=6,
                    help="how many mismatches to print in full")
parser.add_argument("--corpus", nargs="*", default=None,
                    help="also check every line of these files. A curated "
                         "case list can pass while real text fails, so this "
                         "should be pointed at something messy -- the repo's "
                         "own mixed Chinese/English/code docs do nicely")
parser.add_argument("--corpus-lines", type=int, default=4000)
parser.add_argument("--ref", choices=["hf", "llama"], default="hf",
                    help="reference tokenizer: 'hf' (AutoTokenizer, needs "
                         "--tokenizer) or 'llama' (llama-tokenize against "
                         "the SAME .gguf -- no HF checkout needed, which is "
                         "the whole point for a model that only exists as a "
                         "gguf on this machine)")
parser.add_argument("--llama-tokenize", default="llama-tokenize")
args = parser.parse_args()

if not os.path.exists(args.ggufctl):
    sys.exit("ggufctl not built. Run:\n"
             "  swift build -c release --product ggufctl")

CASES = [
    # plain
    "hello world",
    "Hello, World!",
    "The quick brown fox jumps over the lazy dog.",
    # contractions -- the regex has a dedicated alternative for these
    "it's", "It's", "IT'S", "don't", "we've", "they'll", "I'd", "he'm",
    # whitespace: leading, trailing, runs, tabs, newlines
    " leading", "trailing ", "double  space", "   three", "\\ttab",
    "line1\\nline2", "\\n\\nblank lines", "trailing newline\\n",
    " ", "  ", "\\n", " \\n ",
    # digits -- \p{N} splits every digit on its own
    "42", "3.14159", "1234567890", "v2.0.1", "10,000",
    # CJK
    "杭州", "杭州是一座美丽的城市。",
    "中文和English混排, 还有数字123。",
    "日本語のテキスト", "한국어 텍스트",
    # emoji and combining marks
    "hello 👋", "👨‍👩‍👧‍👦 family", "café", "cafe\u0301", "naïve",
    "🇨🇳🇺🇸", "e\u0301\u0301 double mark",
    # punctuation and symbols
    "!@#$%^&*()", "a---b", "<<>>", "...", "?!?!",
    "path/to/file.txt", "https://example.com/a?b=c&d=e",
    # code-ish
    "def f(x):\\n    return x + 1",
    "{\"key\": [1, 2, 3]}",
    # special tokens: must be matched literally, never merged
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<|im_start|>user\\nhi<|im_end|>",
    "text before <|im_start|> text after",
    "<|vision_start|><|image_pad|><|vision_end|>",
    # adversarial: near-misses of special tokens
    "<|im_start", "im_start|>", "<|not_a_token|>", "<||>",
    # long-ish mixed
    "杭州 (Hangzhou) is a city of 12,000,000 people — it's famous for 西湖!",
    "",
]


def hf_ids(tokenizer, cases):
    return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in cases]


def swift_ids(cases):
    payload = "\n".join(c.replace("\n", "\\n").replace("\t", "\\t")
                        for c in cases) + "\n"
    result = subprocess.run(
        [args.ggufctl, "encode", "--gguf", args.gguf, "--batch"],
        input=payload, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit("ggufctl failed: %s" % result.stderr.strip())
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    if len(lines) != len(cases):
        sys.exit("ggufctl returned %d lines for %d cases"
                 % (len(lines), len(cases)))
    return [json.loads(l) for l in lines]


def llama_ids(cases):
    """llama-tokenize, one process per case. Slow and dumb on purpose: a
    batch wire format here would be a second thing to get wrong."""
    out = []
    for c in cases:
        if c == "":
            # llama-tokenize refuses an empty -p; the empty string
            # tokenizes to nothing in every dialect here.
            out.append([])
            continue
        result = subprocess.run(
            [args.llama_tokenize, "-m", args.gguf, "-p", c,
             "--ids", "--no-bos"],
            capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit("llama-tokenize failed on %r: %s"
                     % (c, result.stderr.strip()[-200:]))
        last = [l for l in result.stdout.splitlines() if l.startswith("[")]
        out.append(json.loads(last[-1]) if last else [])
    return out


tok = None
if args.ref == "hf":
    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        trust_remote_code=True)

# The literal \n / \t in CASES are for the line-based wire format; unescape
# for HF so both sides see the same string.
resolved = [c.replace("\\n", "\n").replace("\\t", "\t") for c in CASES]

if args.corpus:
    for path in args.corpus:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                # The batch wire format is line based, so a line carrying a
                # literal backslash-n would be unescaped into a real newline
                # on one side only.
                if line and "\\n" not in line and "\\t" not in line:
                    resolved.append(line)
    resolved = resolved[:args.corpus_lines]
    print("corpus: %d lines from %s" % (len(resolved) - len(CASES),
                                        ", ".join(args.corpus)))

expected = hf_ids(tok, resolved) if args.ref == "hf" else llama_ids(resolved)
got = swift_ids(resolved)

mismatches = []
for case, want, have in zip(resolved, expected, got):
    if want != have:
        mismatches.append((case, want, have))

total_tokens = sum(len(x) for x in expected)
print("%d cases, %d tokens, %d mismatch(es)"
      % (len(resolved), total_tokens, len(mismatches)))

for case, want, have in mismatches[:args.show]:
    print("\n  input:    %r" % case)
    print("  ref:      %s" % want)
    print("  swift:    %s" % have)
    if tok is not None:
        print("  ref text: %s" % [tok.decode([i]) for i in want][:24])
        print("  swf text: %s" % [tok.decode([i]) for i in have][:24])
if len(mismatches) > args.show:
    print("\n  ... %d more" % (len(mismatches) - args.show))

# Round trip: decoding our own ids must give the input back -- up to the
# normalizer, which is lossy on purpose. NFC composes "e" + U+0301 into "é"
# and there is no way back; HF's own round trip loses it identically, so the
# target is NFC(input), not input.
round_trip_failures = []
# llama-tokenize has no decoder to compare against; the id-level gate above
# is the whole check in that mode.
for case, ids in ([] if tok is None else zip(resolved, got)):
    if not ids:
        continue
    result = subprocess.run(
        [args.ggufctl, "decode", "--gguf", args.gguf,
         "--ids", ",".join(str(i) for i in ids)],
        capture_output=True, text=True)
    # The target is what HF ITSELF round-trips to, not NFC(input).
    #
    # It used to be NFC(input), which is right for Qwen and only for Qwen:
    # its normalizer is NFC, so decomposed input never comes back. Gemma 4
    # normalizes nothing but the space, so an NFD input round-trips as NFD --
    # and the old expectation failed a tokenizer whose ids matched HF exactly,
    # which is the worst kind of gate: it accuses the implementation of the
    # fixture's assumption. HF is the reference by definition; compare to it.
    hf_round_trip = tok.decode(ids, skip_special_tokens=False)
    if result.stdout != hf_round_trip:
        round_trip_failures.append((case, result.stdout, hf_round_trip))

if round_trip_failures:
    print("\n%d round-trip failure(s):" % len(round_trip_failures))
    for case, decoded, want in round_trip_failures[:args.show]:
        print("  %r -> %r (expected %r)" % (case, decoded, want))

if mismatches or round_trip_failures:
    sys.exit("\ntokenizer parity FAILED")
print("\ngguf tokenizer parity OK -- identical to the reference on every case")
