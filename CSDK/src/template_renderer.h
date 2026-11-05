/* Copyright (c) 2026 Jiejing Zhang. */

#pragma once

#include <tempo9/llm.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace allspark::llm::internal {

struct RenderedCachePoint {
  std::string prefix;
  std::string previous_prefix;
  int32_t ttl_seconds = 300;
};

class TemplateRenderer {
 public:
  ~TemplateRenderer();

  static std::unique_ptr<TemplateRenderer> Create(
      const std::string& source, const std::string& bos_token,
      const std::string& eos_token, std::string* error);

  bool Render(const te9_llm_chat_request& request, std::string* output,
              std::vector<RenderedCachePoint>* cache_points,
              std::string* error) const;

 private:
  class Impl;
  explicit TemplateRenderer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace allspark::llm::internal
