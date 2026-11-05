<!-- Copyright (c) 2026 Jiejing Zhang. -->
# The C and C++ SDK

Run Tempo9 inside a C or C++ program on an Apple Silicon Mac: load a `.gguf`,
render a chat, stream the answer. There is no server and no Python.

The SDK is a download on each [release](https://github.com/thinkspread/tempo9/releases),
`tempo9-c-sdk-<version>-macos-arm64.tar.gz`. It needs macOS 26 and has been run
on an M5 Pro. Linux is not available yet.

| In the archive | What it is |
|---|---|
| `include/tempo9/llm.h` | the C API |
| `include/tempo9/llm.hpp` | the C++17 API, a header-only wrapper over the C API |
| `include/allspark/tempo9_engine.h` | the engine's C API, which `llm.h` builds on |
| `lib/libtempo9_llm.dylib` | tokenizer, chat templates, requests |
| `lib/liballspark_framework.dylib` | the engine, with its Metal kernels embedded |
| `lib/cmake/` | CMake packages `Tempo9LLM` and `Tempo9EngineABI` |
| `examples/` | three small programs and their `CMakeLists.txt` |

## Quick start

```bash
tar xzf tempo9-c-sdk-*-macos-arm64.tar.gz
cd tempo9-c-sdk-*-macos-arm64
cmake -S examples -B build -DCMAKE_PREFIX_PATH="$PWD" -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/llm_chat_c /path/to/model.gguf "What is the capital of France?"
```

## In your project

```cmake
find_package(Tempo9LLM 1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Tempo9::llm)      # C
target_link_libraries(my_app PRIVATE Tempo9::llm_cpp)  # C++17
```

Pass the SDK directory in `CMAKE_PREFIX_PATH`. An app that ships must carry both
dylibs with it and find them through its rpath.

## Calling it from C

`examples/llm_chat.c` is the complete program. The shape is:

1. `te9_engine_create`, `te9_engine_build_model`, `te9_engine_start_model`. Point
   `weights_path` at the `.gguf` and leave `graph_path` null.
2. `te9_llm_pipeline_create`.
3. Fill a `te9_llm_chat_request` and call `te9_llm_request_start` with
   `input_kind = TE9_LLM_INPUT_CHAT`.
4. Loop on `te9_llm_request_read`. Zero bytes means the read timed out, so stop
   only when `te9_llm_request_status` is no longer `TE9_GEN_RUNNING`.
5. Release the request, destroy the pipeline, then stop, release and destroy on
   the engine.

Every struct starts with `struct_size`: zero-initialize it and set that field.

Print `te9_gemm_backend()` at startup. `cpu` means the engine fell back from
Metal, and `te9_gemm_backend_reason()` says why. The answer is still correct,
but much slower.

The engine logs to stderr. `GLOG_minloglevel=2` keeps only errors; its
`[tempo9-metal]` startup lines still print.

## Which models work

`libtempo9_llm` tokenizes in C++, so a model has to use a tokenizer it implements.

| GGUF tokenizer | C/C++ SDK | Run on |
|---|---|---|
| `gpt2`, `pre` is `qwen2` or `qwen35` | yes | Qwen3.5 0.8B, Qwen3.6-35B-A3B, Qwen3.8-27B |
| `llama` (SentencePiece) | yes | |
| `gemma4` | no | use the Swift SDK |
| `gpt2`, `pre` is `flint` | no | use the Swift SDK |

A model that is not supported fails at `te9_llm_pipeline_create`, and the error
says which tokenizers are.

## Versions

Three numbers apply, and they answer different questions:

| Number | Meaning |
|---|---|
| SDK version, the one in the file name | The C/C++ SDK and the Swift SDK ship under one number. |
| `te9_llm_api_version()` | Counts changes to the structs in `llm.h`. Check this one to know a field exists. It is not related to the SDK version. |
| `te9_engine_api_version()` | The engine, versioned on its own. |

## License

`libtempo9_llm`, the headers and the examples are Apache-2.0, and their source is
in [CSDK/](../CSDK). `liballspark_framework.dylib` is the engine, a binary under
[ENGINE-LICENSE](../ENGINE-LICENSE): free for individuals and for organisations
with fewer than 100 employees and under US$1M in annual revenue; above that, a
commercial licence is required. You may ship it as part of your application, not
on its own.
