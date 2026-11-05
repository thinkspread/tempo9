/* Copyright (c) 2026 Jiejing Zhang. */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace allspark::llm::internal {

struct Vocabulary {
  std::string model;
  std::string pre;
  std::vector<std::string> tokens;
  std::vector<float> scores;
  std::vector<int32_t> types;
  std::vector<std::string> merges;
  int32_t bos_id = -1;
  int32_t eos_id = -1;
  int32_t unk_id = -1;
  int8_t add_bos = -1;
  int8_t add_eos = -1;
  int8_t add_space_prefix = -1;
  std::string chat_template;
};

std::array<uint8_t, 32> Sha256(const uint8_t* data, size_t size);
bool ParseMetadata(const uint8_t* data, size_t size, Vocabulary* vocabulary,
                   std::string* error);

}  // namespace allspark::llm::internal
