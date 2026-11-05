<!-- Copyright (c) 2026 Jiejing Zhang. -->

# llama.cpp Jinja (vendored)

The chat-template engine of `libtempo9_llm`. It is llama.cpp's `common/jinja`
plus the two `common/` files it depends on, copied unmodified from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at commit
`162e2bb2e42a0bbf6f9c2d3d8a9eb3cc40b4a4ea` (2026-11-04). MIT license:
`LICENSES/MIT-llama.cpp.txt` at the repository root.

| Path | Upstream |
|---|---|
| `common/jinja/*` | `common/jinja/*` |
| `common/json.{h,cpp}` | `common/json.{h,cpp}` (`common_json`, a pimpl over nlohmann/json) |
| `common/unicode.{h,cpp}` | `common/unicode.{h,cpp}` |
| `common/ggml.h` | **ours**: defines `GGML_ASSERT`, the only thing `json.cpp` takes from ggml |

It replaced google/minja, which stopped receiving changes in September 2025 and
failed on templates we ship: `is true` / `is false` tests (Qwen3.8) and
`tojson(ensure_ascii=...)` (Flint).

To update, copy the same files from a newer commit, change the commit above and
in `cmake/Tempo9LLMRuntimeManifest.json.in`, and rerun
`serving/native/tests/template_parity.py` against the GGUFs we ship (usage in
its header). Do not edit the vendored files; adapt in
`src/template_renderer.cpp` instead.
