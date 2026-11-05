/* Copyright (c) 2026 Jiejing Zhang. */

// Stand-in for llama.cpp's ggml.h. The vendored json.cpp includes it only to
// define JSON_ASSERT as GGML_ASSERT; nlohmann/json's own default is assert(),
// which is what this restores.

#pragma once

#include <cassert>

#define GGML_ASSERT(x) assert(x)
