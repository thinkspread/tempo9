<!-- Copyright (c) 2026 Jiejing Zhang. -->
# libtempo9_llm

The source of the C/C++ SDK's `libtempo9_llm`: GGUF metadata, tokenizers, chat
templates and the request API in `include/tempo9/llm.h`, Apache-2.0 like the rest
of this repository. The inference engine is not here. It comes as a binary in the
SDK download, under [ENGINE-LICENSE](../ENGINE-LICENSE).

To build it, unpack the SDK and point CMake at it:

```bash
cmake -S CSDK -B build -DCMAKE_PREFIX_PATH=/path/to/tempo9-c-sdk-<version>-macos-arm64
cmake --build build
```

To use the SDK rather than change it, see [docs/c-sdk.md](../docs/c-sdk.md).

This directory is copied from the engine's repository at each change, and
`third_party/llama-jinja` is vendored unmodified from llama.cpp. A pull request
here is ported back by a maintainer.
