/* Copyright (c) 2026 Jiejing Zhang. */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "metadata.h"

namespace allspark::llm::internal {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  static std::unique_ptr<Tokenizer> Create(Vocabulary vocabulary,
                                            std::string* error);

  virtual bool Encode(const std::string& text, bool add_special,
                      std::vector<int64_t>* output,
                      std::string* error) const = 0;
  virtual bool AppendDecoded(int64_t token_id, std::string* output,
                             std::string* error) const = 0;
  virtual int32_t bos_id() const = 0;
  virtual int32_t eos_id() const = 0;
  virtual size_t max_piece_bytes() const = 0;
  virtual bool trim_initial_space() const = 0;
  virtual const std::string& TokenText(int32_t token_id) const = 0;
};

bool ValidateUtf8(const char* data, size_t size, std::string* error);

}  // namespace allspark::llm::internal
