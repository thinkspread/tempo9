/* Copyright (c) 2026 Jiejing Zhang. */

#include <tempo9/llm.h>
#include <utf8proc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "metadata.h"
#include "template_renderer.h"
#include "tokenizer.h"

namespace {

using allspark::llm::internal::ParseMetadata;
using allspark::llm::internal::RenderedCachePoint;
using allspark::llm::internal::Sha256;
using allspark::llm::internal::TemplateRenderer;
using allspark::llm::internal::Tokenizer;
using allspark::llm::internal::ValidateUtf8;
using allspark::llm::internal::Vocabulary;

constexpr size_t kLastErrorCapacity = 2048;
thread_local std::array<char, kLastErrorCapacity> g_last_error{};

void SetError(std::string_view error) noexcept {
  const size_t size = std::min(error.size(), g_last_error.size() - 1u);
  if (size != 0) std::memcpy(g_last_error.data(), error.data(), size);
  g_last_error[size] = '\0';
}

te9_status ProviderError(te9_status status, const char* operation) {
  std::string error(operation);
  const char* provider = te9_last_error();
  if (provider != nullptr && provider[0] != '\0') {
    error += ": ";
    error += provider;
  }
  SetError(std::move(error));
  return status;
}

template <typename T>
bool StructOk(const T* value, size_t minimum) {
  return value != nullptr && value->struct_size >= minimum &&
         value->struct_size <= sizeof(T) * 4u;
}

bool ViewOk(te9_llm_string_view view) {
  return view.data != nullptr || view.size == 0;
}

std::string CopyView(te9_llm_string_view view) {
  return view.size == 0 ? std::string() : std::string(view.data, view.size);
}

class OperationGate {
 public:
  class Lease {
   public:
    Lease() = default;
    explicit Lease(OperationGate* gate) : gate_(gate) {}
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept
        : gate_(std::exchange(other.gate_, nullptr)) {}
    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        Reset();
        gate_ = std::exchange(other.gate_, nullptr);
      }
      return *this;
    }
    ~Lease() { Reset(); }
    explicit operator bool() const { return gate_ != nullptr; }

   private:
    void Reset() {
      if (gate_ != nullptr) gate_->Release();
      gate_ = nullptr;
    }
    OperationGate* gate_ = nullptr;
  };

  Lease Borrow() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closing_) return Lease{};
    ++active_;
    return Lease(this);
  }

  void Close() {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    condition_.wait(lock, [&] { return active_ == 0; });
  }

 private:
  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (--active_ == 0) condition_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  size_t active_ = 0;
  bool closing_ = false;
};

template <typename T>
class Registry {
 public:
  uint64_t Add(std::shared_ptr<T> value) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t index = 0;
    while (index < slots_.size() && slots_[index].value != nullptr) ++index;
    if (index == UINT32_MAX) return 0;
    if (index == slots_.size()) slots_.push_back(Slot{});
    Slot& slot = slots_[index];
    if (slot.generation == 0) slot.generation = 1;
    slot.value = std::move(value);
    return (static_cast<uint64_t>(slot.generation) << 32u) |
           static_cast<uint64_t>(index + 1u);
  }

  std::shared_ptr<T> Get(uint64_t handle) const {
    if (handle == 0) return nullptr;
    const uint32_t low = static_cast<uint32_t>(handle);
    const uint32_t generation = static_cast<uint32_t>(handle >> 32u);
    if (low == 0 || generation == 0) return nullptr;
    const size_t index = static_cast<size_t>(low - 1u);
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= slots_.size() || slots_[index].generation != generation)
      return nullptr;
    return slots_[index].value;
  }

  std::shared_ptr<T> Remove(uint64_t handle) {
    if (handle == 0) return nullptr;
    const uint32_t low = static_cast<uint32_t>(handle);
    const uint32_t generation = static_cast<uint32_t>(handle >> 32u);
    if (low == 0 || generation == 0) return nullptr;
    const size_t index = static_cast<size_t>(low - 1u);
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= slots_.size() || slots_[index].generation != generation ||
        slots_[index].value == nullptr)
      return nullptr;
    std::shared_ptr<T> result = std::move(slots_[index].value);
    if (++slots_[index].generation == 0) ++slots_[index].generation;
    return result;
  }

 private:
  struct Slot {
    uint32_t generation = 1;
    std::shared_ptr<T> value;
  };
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
};

struct PipelineRecord {
  te9_engine_t engine = nullptr;
  std::string model_name;
  te9_engine_api_table provider{};
  te9_model_llm_source source{};
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<TemplateRenderer> renderer;
  OperationGate gate;
};

struct RequestRecord {
  ~RequestRecord() {
    if (provider_request != nullptr) te9_request_release(provider_request);
  }

  OperationGate gate;
  std::shared_ptr<PipelineRecord> pipeline;
  te9_request_t provider_request = nullptr;
  std::mutex provider_mutex;
  std::mutex read_mutex;
  std::mutex stop_mutex;
  std::atomic<bool> stopped{false};
  std::string pending;
  std::vector<int64_t> generated_tail;
  std::vector<std::vector<int64_t>> stops;
  size_t max_stop_length = 0;
  size_t matched_stop = SIZE_MAX;
  bool initial_output = true;
};

Registry<PipelineRecord> g_pipelines;
Registry<RequestRecord> g_requests;

bool SameSource(const te9_model_llm_source& left,
                const te9_model_llm_source& right) {
  return left.engine_generation == right.engine_generation &&
         left.model_generation == right.model_generation &&
         left.llm_metadata_format_version ==
             right.llm_metadata_format_version &&
         std::memcmp(left.llm_metadata_sha256, right.llm_metadata_sha256,
                     sizeof(left.llm_metadata_sha256)) == 0;
}

te9_status ReadProviderTable(te9_engine_api_table* table) {
  *table = {};
  const te9_status status = te9_engine_get_api_table(sizeof(*table), table);
  if (status != TE9_OK)
    return ProviderError(status, "engine ABI discovery failed");
  // ABI 3.0 is 1.5 renumbered to follow the engine release (tempo9_engine.h),
  // so 1.1 and later are all usable; what is actually required is checked
  // just below, by capability bit.
  if (!TE9_API_VERSION_IN_FAMILY(table->api_version) ||
      TE9_API_VERSION_RANK(table->api_version) <
          TE9_API_VERSION_RANK(TE9_LLM_REQUIRED_ENGINE_API_VERSION)) {
    SetError("libtempo9_llm requires Tempo9 Engine ABI 1.1 or later (1.x, 3.x)");
    return TE9_ERR_UNSUPPORTED;
  }
  const te9_engine_capabilities required =
      TE9_ENGINE_CAP_MODEL_LLM_SOURCE | TE9_ENGINE_CAP_CHECKED_REQUEST_START;
  if ((table->capabilities & required) != required ||
      table->get_model_llm_source == nullptr ||
      table->request_start_checked == nullptr) {
    SetError("engine provider lacks model metadata or checked admission");
    return TE9_ERR_UNSUPPORTED;
  }
  return TE9_OK;
}

te9_status CopyBytes(const std::string& value, char* output, size_t capacity,
                     size_t* needed) {
  if (needed == nullptr || (output == nullptr && capacity != 0)) {
    SetError("output buffer or needed pointer is invalid");
    return TE9_ERR_INVALID_ARG;
  }
  *needed = value.size();
  if (capacity < value.size()) return TE9_ERR_BUFFER_TOO_SMALL;
  if (!value.empty()) std::memcpy(output, value.data(), value.size());
  return TE9_OK;
}

te9_status Encode(PipelineRecord* pipeline, const std::string& text,
                  bool add_special, std::vector<int64_t>* tokens) {
  if (pipeline->tokenizer == nullptr) {
    SetError("this model exposes no reconstructable tokenizer metadata");
    return TE9_ERR_UNSUPPORTED;
  }
  std::string error;
  if (!pipeline->tokenizer->Encode(text, add_special, tokens, &error)) {
    SetError(std::move(error));
    return TE9_ERR_INVALID_ARG;
  }
  return TE9_OK;
}

te9_status Render(PipelineRecord* pipeline, const te9_llm_chat_request* request,
                  std::string* rendered,
                  std::vector<RenderedCachePoint>* cache_points = nullptr) {
  if (pipeline->renderer == nullptr) {
    SetError("this pipeline has no supported chat template");
    return TE9_ERR_UNSUPPORTED;
  }
  if (request == nullptr) {
    SetError("chat request is null");
    return TE9_ERR_INVALID_ARG;
  }
  std::string error;
  std::vector<RenderedCachePoint> ignored_cache_points;
  if (!pipeline->renderer->Render(
          *request, rendered,
          cache_points == nullptr ? &ignored_cache_points : cache_points,
          &error)) {
    const bool unsupported = error.find("unsupported") != std::string::npos ||
                             error.find("TTL capability") != std::string::npos;
    SetError(std::move(error));
    return unsupported ? TE9_ERR_UNSUPPORTED : TE9_ERR_INVALID_ARG;
  }
  return TE9_OK;
}

bool ParseJsonObject(const std::string& value, const char* name,
                     std::string* error) {
  try {
    if (!nlohmann::json::parse(value).is_object()) {
      *error = std::string(name) + " must be a JSON object";
      return false;
    }
  } catch (const std::exception& exception) {
    *error = std::string(name) + " is invalid JSON: " + exception.what();
    return false;
  }
  return true;
}

size_t CompleteUtf8Prefix(const std::string& bytes, bool* invalid) {
  *invalid = false;
  size_t offset = 0;
  while (offset < bytes.size()) {
    const uint8_t lead = static_cast<uint8_t>(bytes[offset]);
    size_t expected = 0;
    if (lead < 0x80)
      expected = 1;
    else if (lead >= 0xc2 && lead <= 0xdf)
      expected = 2;
    else if (lead >= 0xe0 && lead <= 0xef)
      expected = 3;
    else if (lead >= 0xf0 && lead <= 0xf4)
      expected = 4;
    else {
      *invalid = true;
      return offset;
    }
    if (bytes.size() - offset < expected) {
      for (size_t i = offset + 1; i < bytes.size(); ++i) {
        if ((static_cast<uint8_t>(bytes[i]) & 0xc0u) != 0x80u) {
          *invalid = true;
          break;
        }
      }
      return offset;
    }
    int32_t codepoint = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(bytes.data() + offset),
        static_cast<utf8proc_ssize_t>(expected), &codepoint);
    if (length != static_cast<utf8proc_ssize_t>(expected)) {
      *invalid = true;
      return offset;
    }
    offset += expected;
  }
  return offset;
}

size_t PrefixWithinCapacity(const std::string& bytes, size_t complete,
                            size_t capacity) {
  size_t offset = 0;
  while (offset < complete) {
    const uint8_t lead = static_cast<uint8_t>(bytes[offset]);
    const size_t length = lead < 0x80   ? 1u
                          : lead < 0xe0 ? 2u
                          : lead < 0xf0 ? 3u
                                        : 4u;
    if (offset + length > capacity) break;
    offset += length;
  }
  return offset;
}

bool CheckedMultiply(size_t left, size_t right, size_t* output) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    return false;
  *output = left * right;
  return true;
}

uint64_t DerivedMediaContentHash(const te9_llm_media_embedding& media,
                                 int32_t kind) noexcept {
  uint64_t hash = UINT64_C(14695981039346656037);
  const auto mix = [&](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i != size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
    }
  };
  /* Preserve ABI 1.7 image hashes so exported prefix caches remain reusable
   * after an ABI 1.8 upgrade. Audio uses a new domain and includes its kind. */
  const uint64_t format_version =
      kind == TE9_LLM_MEDIA_IMAGE ? UINT64_C(1) : UINT64_C(2);
  const uint64_t byte_count = media.byte_count;
  const uint64_t deepstack_byte_count = media.deepstack_byte_count;
  const int32_t no_mrope = media.no_mrope != 0 ? 1 : 0;
  mix(&format_version, sizeof(format_version));
  mix(&byte_count, sizeof(byte_count));
  mix(media.data, media.byte_count);
  mix(&media.token_count, sizeof(media.token_count));
  mix(&media.hidden, sizeof(media.hidden));
  mix(&media.dtype, sizeof(media.dtype));
  mix(&media.media_token_id, sizeof(media.media_token_id));
  mix(&media.grid_t, sizeof(media.grid_t));
  mix(&media.grid_h, sizeof(media.grid_h));
  mix(&media.grid_w, sizeof(media.grid_w));
  mix(&media.spatial_merge_size, sizeof(media.spatial_merge_size));
  mix(&no_mrope, sizeof(no_mrope));
  if (kind != TE9_LLM_MEDIA_IMAGE) mix(&kind, sizeof(kind));
  mix(&media.num_deepstack_layers, sizeof(media.num_deepstack_layers));
  mix(&deepstack_byte_count, sizeof(deepstack_byte_count));
  mix(media.deepstack_data, media.deepstack_byte_count);
  return hash == 0 ? UINT64_C(1) : hash;
}

struct PreparedMedia {
  std::vector<te9_image_embeds> provider;
  std::vector<int32_t> positions;
  /* Maps each exclusive offset in the caller's unexpanded prompt to the
   * corresponding exclusive offset in the expanded provider prompt. */
  std::vector<size_t> offset_map;
};

te9_status PrepareMedia(const te9_llm_media_embedding* media,
                        const int32_t* media_kinds, size_t media_count,
                        std::vector<int64_t>* prompt, PreparedMedia* prepared) {
  if (media == nullptr || media_count == 0 || media_count > 64) {
    SetError("precomputed media must contain between one and 64 embeddings");
    return TE9_ERR_INVALID_ARG;
  }
  prepared->provider.clear();
  prepared->provider.resize(media_count);
  constexpr size_t kMediaV17Size = sizeof(te9_llm_media_embedding);
  bool uses_mrope = false;
  int64_t common_hidden = 0;
  int32_t common_deepstack_layers = 0;
  for (size_t i = 0; i != media_count; ++i) {
    const te9_llm_media_embedding& source = media[i];
    if (!StructOk(&source, kMediaV17Size) || source.data == nullptr ||
        source.token_count <= 0 || source.hidden <= 0 ||
        source.media_token_id <= 0) {
      SetError("media embedding " + std::to_string(i) +
               " has an invalid struct, pointer, shape, or token id");
      return TE9_ERR_INVALID_ARG;
    }
    if (static_cast<uint64_t>(source.token_count) >
            std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(source.hidden) >
            std::numeric_limits<size_t>::max()) {
      SetError("media embedding shape exceeds the host size range");
      return TE9_ERR_INVALID_ARG;
    }
    if (source.dtype != TE9_LLM_MEDIA_FLOAT16 &&
        source.dtype != TE9_LLM_MEDIA_FLOAT32) {
      SetError("media embedding " + std::to_string(i) +
               " has an unsupported dtype");
      return TE9_ERR_INVALID_ARG;
    }
    size_t elements = 0;
    size_t expected_bytes = 0;
    if (!CheckedMultiply(static_cast<size_t>(source.token_count),
                         static_cast<size_t>(source.hidden), &elements) ||
        !CheckedMultiply(elements,
                         source.dtype == TE9_LLM_MEDIA_FLOAT16 ? 2u : 4u,
                         &expected_bytes) ||
        source.byte_count != expected_bytes) {
      SetError("media embedding " + std::to_string(i) +
               " byte count does not match token_count*hidden*dtype");
      return TE9_ERR_INVALID_ARG;
    }
    if (source.no_mrope != 0 && source.no_mrope != 1) {
      SetError("media embedding no_mrope must be zero or one");
      return TE9_ERR_INVALID_ARG;
    }
    const int32_t kind =
        media_kinds == nullptr ? TE9_LLM_MEDIA_IMAGE : media_kinds[i];
    if (kind != TE9_LLM_MEDIA_IMAGE && kind != TE9_LLM_MEDIA_AUDIO) {
      SetError("media embedding " + std::to_string(i) +
               " has an unsupported media kind");
      return TE9_ERR_INVALID_ARG;
    }
    if (kind == TE9_LLM_MEDIA_AUDIO && source.no_mrope == 0) {
      SetError("audio embeddings require ordinary positions (no_mrope=1)");
      return TE9_ERR_INVALID_ARG;
    }
    if (source.no_mrope != 0) {
      if (source.grid_t != 0 || source.grid_h != 0 || source.grid_w != 0 ||
          source.spatial_merge_size != 0) {
        SetError("no-M-RoPE media must not declare grid or merge dimensions");
        return TE9_ERR_INVALID_ARG;
      }
    } else {
      uses_mrope = true;
      if (source.grid_t != 1 || source.grid_h <= 0 || source.grid_w <= 0 ||
          source.spatial_merge_size <= 0 ||
          source.grid_h % source.spatial_merge_size != 0 ||
          source.grid_w % source.spatial_merge_size != 0) {
        SetError(
            "M-RoPE image grid must have t=1 and h/w divisible by the "
            "positive spatial merge size");
        return TE9_ERR_INVALID_ARG;
      }
      const int64_t merged_h = source.grid_h / source.spatial_merge_size;
      const int64_t merged_w = source.grid_w / source.spatial_merge_size;
      if (merged_h > std::numeric_limits<int64_t>::max() / merged_w ||
          source.token_count != merged_h * merged_w) {
        SetError("media token_count does not match the merged image grid");
        return TE9_ERR_INVALID_ARG;
      }
    }
    if (source.num_deepstack_layers < 0 || source.num_deepstack_layers > 64) {
      SetError("media deepstack layer count is outside 0..64");
      return TE9_ERR_INVALID_ARG;
    }
    if (kind == TE9_LLM_MEDIA_AUDIO &&
        (source.num_deepstack_layers != 0 || source.deepstack_data != nullptr ||
         source.deepstack_byte_count != 0)) {
      SetError("audio embeddings cannot carry DeepStack features");
      return TE9_ERR_INVALID_ARG;
    }
    if (i == 0) {
      common_hidden = source.hidden;
      common_deepstack_layers = source.num_deepstack_layers;
    } else if (source.hidden != common_hidden ||
               source.num_deepstack_layers != common_deepstack_layers) {
      SetError(
          "all media embeddings in one request must use the same hidden "
          "width and deepstack layer count");
      return TE9_ERR_INVALID_ARG;
    }
    size_t expected_deepstack = 0;
    if (!CheckedMultiply(expected_bytes,
                         static_cast<size_t>(source.num_deepstack_layers),
                         &expected_deepstack) ||
        source.deepstack_byte_count != expected_deepstack ||
        (expected_deepstack != 0 && source.deepstack_data == nullptr)) {
      SetError("media deepstack bytes do not match layers*embedding bytes");
      return TE9_ERR_INVALID_ARG;
    }
  }

  prepared->offset_map.assign(prompt->size() + 1u, 0u);
  size_t next_media = 0;
  size_t expanded_size = 0;
  int64_t previous_media_token = -1;
  for (size_t i = 0; i != prompt->size(); ++i) {
    const int64_t token = (*prompt)[i];
    size_t matching = media_count;
    for (size_t j = 0; j != media_count; ++j) {
      if (media[j].media_token_id == token) {
        matching = j;
        break;
      }
    }
    if (matching == media_count) {
      if (expanded_size == std::numeric_limits<size_t>::max()) {
        SetError("expanded media prompt is too large");
        return TE9_ERR_INVALID_ARG;
      }
      ++expanded_size;
      previous_media_token = -1;
    } else {
      if (next_media == media_count ||
          media[next_media].media_token_id != token) {
        SetError("media embedding order does not match rendered placeholders");
        return TE9_ERR_INVALID_ARG;
      }
      if (previous_media_token == token) {
        SetError(
            "adjacent media placeholders with the same token id are "
            "ambiguous; the chat template must separate them");
        return TE9_ERR_INVALID_ARG;
      }
      const size_t count = static_cast<size_t>(media[next_media].token_count);
      if (expanded_size > std::numeric_limits<size_t>::max() - count) {
        SetError("expanded media prompt is too large");
        return TE9_ERR_INVALID_ARG;
      }
      expanded_size += count;
      previous_media_token = token;
      ++next_media;
    }
    prepared->offset_map[i + 1u] = expanded_size;
  }
  if (next_media != media_count) {
    SetError(
        "rendered media placeholder count is smaller than embedding count");
    return TE9_ERR_INVALID_ARG;
  }
  if (uses_mrope &&
      expanded_size > std::numeric_limits<size_t>::max() / size_t{3}) {
    SetError("expanded media position table is too large");
    return TE9_ERR_INVALID_ARG;
  }

  std::vector<int64_t> expanded;
  expanded.reserve(expanded_size);
  if (uses_mrope) prepared->positions.assign(expanded_size * 3u, 0);
  size_t media_index = 0;
  size_t output_index = 0;
  int64_t current_position = 0;
  for (const int64_t token : *prompt) {
    if (media_index < media_count &&
        token == media[media_index].media_token_id) {
      const te9_llm_media_embedding& source = media[media_index];
      const bool block_uses_mrope = source.no_mrope == 0;
      const int64_t merged_h =
          block_uses_mrope ? source.grid_h / source.spatial_merge_size : 0;
      const int64_t merged_w =
          block_uses_mrope ? source.grid_w / source.spatial_merge_size : 0;
      for (int64_t index = 0; index != source.token_count; ++index) {
        expanded.push_back(token);
        if (uses_mrope) {
          const int64_t row = block_uses_mrope ? index / merged_w : 0;
          const int64_t column = block_uses_mrope ? index % merged_w : 0;
          if (current_position >
              std::numeric_limits<int32_t>::max() - std::max(row, column)) {
            SetError("media M-RoPE position exceeds int32 range");
            return TE9_ERR_INVALID_ARG;
          }
          prepared->positions[output_index] =
              static_cast<int32_t>(current_position);
          prepared->positions[expanded_size + output_index] =
              static_cast<int32_t>(current_position + row);
          prepared->positions[expanded_size * 2u + output_index] =
              static_cast<int32_t>(current_position + column);
        }
        if (!block_uses_mrope) ++current_position;
        ++output_index;
      }
      if (block_uses_mrope) current_position += std::max(merged_h, merged_w);
      ++media_index;
    } else {
      expanded.push_back(token);
      if (uses_mrope) {
        if (current_position > std::numeric_limits<int32_t>::max()) {
          SetError("text M-RoPE position exceeds int32 range");
          return TE9_ERR_INVALID_ARG;
        }
        const int32_t position = static_cast<int32_t>(current_position++);
        prepared->positions[output_index] = position;
        prepared->positions[expanded_size + output_index] = position;
        prepared->positions[expanded_size * 2u + output_index] = position;
      }
      ++output_index;
    }
  }
  *prompt = std::move(expanded);

  for (size_t i = 0; i != media_count; ++i) {
    const te9_llm_media_embedding& source = media[i];
    te9_image_embeds& target = prepared->provider[i];
    target.struct_size = sizeof(target);
    target.data = source.data;
    target.byte_count = source.byte_count;
    target.token_count = source.token_count;
    target.hidden = source.hidden;
    target.is_float16 = source.dtype == TE9_LLM_MEDIA_FLOAT16 ? 1 : 0;
    const int32_t kind =
        media_kinds == nullptr ? TE9_LLM_MEDIA_IMAGE : media_kinds[i];
    target.content_hash = source.content_hash == 0
                              ? DerivedMediaContentHash(source, kind)
                              : source.content_hash;
    target.image_token_id = source.media_token_id;
    if (uses_mrope) {
      target.mrope_positions = prepared->positions.data();
      target.mrope_position_count = prepared->positions.size();
    }
    target.no_mrope = i == 0 ? (uses_mrope ? 0 : 1) : source.no_mrope;
    target.next = i + 1u == media_count ? nullptr : &prepared->provider[i + 1u];
    target.deepstack_data = source.deepstack_data;
    target.deepstack_byte_count = source.deepstack_byte_count;
    target.num_deepstack_layers = source.num_deepstack_layers;
  }
  return TE9_OK;
}

void TrackStops(RequestRecord* request, const int64_t* tokens, size_t count) {
  if (request->max_stop_length == 0) return;
  for (size_t i = 0; i != count; ++i) {
    if (request->generated_tail.size() == request->max_stop_length)
      request->generated_tail.erase(request->generated_tail.begin());
    request->generated_tail.push_back(tokens[i]);
    for (size_t stop = 0; stop != request->stops.size(); ++stop) {
      const std::vector<int64_t>& sequence = request->stops[stop];
      if (sequence.size() <= request->generated_tail.size() &&
          std::equal(sequence.rbegin(), sequence.rend(),
                     request->generated_tail.rbegin())) {
        request->matched_stop = stop;
        break;
      }
    }
  }
}

template <typename F>
te9_status Guard(F&& function) {
  try {
    g_last_error[0] = '\0';
    return function();
  } catch (const std::bad_alloc&) {
    SetError("libtempo9_llm ran out of memory");
    return TE9_ERR_OUT_OF_MEMORY;
  } catch (const std::exception& exception) {
    SetError(exception.what());
    return TE9_ERR_EXCEPTION;
  } catch (...) {
    SetError("unknown C++ exception in libtempo9_llm");
    return TE9_ERR_EXCEPTION;
  }
}

}  // namespace

extern "C" {

uint32_t te9_llm_api_version(void) { return TE9_LLM_API_VERSION; }

te9_status te9_llm_get_api_table(size_t caller_size, te9_llm_api_table* table) {
  return Guard([&] {
    constexpr size_t minimum = offsetof(te9_llm_api_table, capabilities) +
                               sizeof(te9_llm_capabilities);
    if (table == nullptr || caller_size < minimum) {
      SetError(
          "te9_llm_get_api_table: table is null or caller_size is too small");
      return TE9_ERR_INVALID_ARG;
    }
    te9_engine_api_table provider{};
    const te9_status provider_status = ReadProviderTable(&provider);
    te9_llm_api_table result{};
    result.struct_size = sizeof(result);
    result.api_version = TE9_LLM_API_VERSION;
    result.required_engine_api_version = TE9_LLM_REQUIRED_ENGINE_API_VERSION;
    result.required_engine_capabilities =
        TE9_ENGINE_CAP_MODEL_LLM_SOURCE | TE9_ENGINE_CAP_CHECKED_REQUEST_START;
    result.active_engine_api_version = provider.api_version;
    result.metadata_format_version = TE9_LLM_METADATA_FORMAT_V1;
    result.active_engine_capabilities = provider.capabilities;
    if (provider_status == TE9_OK) {
      result.capabilities =
          TE9_LLM_CAP_UTF8_PROMPT | TE9_LLM_CAP_TYPED_CHAT |
          TE9_LLM_CAP_STREAMING_UTF8 | TE9_LLM_CAP_STOP_SEQUENCES |
          TE9_LLM_CAP_RESPONSE_FORMAT | TE9_LLM_CAP_REASONING_CONTROL |
          TE9_LLM_CAP_TOOL_RESULT_ERROR | TE9_LLM_CAP_THINKING_CONTROL |
          TE9_LLM_CAP_PRECOMPUTED_MEDIA | TE9_LLM_CAP_PRECOMPUTED_AUDIO |
          TE9_LLM_CAP_PRESERVE_THINKING;
      if ((provider.capabilities & TE9_ENGINE_CAP_PROMPT_CACHE_TTL) != 0) {
        result.capabilities |=
            TE9_LLM_CAP_PROMPT_CACHE_TTL | TE9_LLM_CAP_CHAT_CACHE_CONTROL;
      }
      if ((provider.capabilities & TE9_ENGINE_CAP_CONCURRENT_REQUEST_STOP) != 0)
        result.capabilities |= TE9_LLM_CAP_CONCURRENT_REQUEST_STOP;
      if ((provider.capabilities & TE9_ENGINE_CAP_PREFIX_CACHE_TRANSFER) != 0 &&
          provider.export_prefix_cache != nullptr &&
          provider.import_prefix_cache != nullptr)
        result.capabilities |= TE9_LLM_CAP_PREFIX_CACHE_TRANSFER;
    }
    std::memcpy(table, &result, std::min(caller_size, sizeof(result)));
    return provider_status;
  });
}

const char* te9_llm_last_error(void) { return g_last_error.data(); }

te9_status te9_llm_pipeline_create(const te9_llm_pipeline_config* config,
                                   te9_llm_pipeline_t* pipeline) {
  return Guard([&] {
    if (pipeline != nullptr) *pipeline = 0;
    if (!StructOk(config, sizeof(te9_llm_pipeline_config)) ||
        pipeline == nullptr || config->engine == nullptr ||
        !ViewOk(config->model_name) || config->model_name.size == 0 ||
        !ViewOk(config->chat_template_override)) {
      SetError("te9_llm_pipeline_config is invalid");
      return TE9_ERR_INVALID_ARG;
    }
    auto value = std::make_shared<PipelineRecord>();
    value->engine = config->engine;
    value->model_name = CopyView(config->model_name);
    std::string model_name_error;
    if (!ValidateUtf8(value->model_name.data(), value->model_name.size(),
                      &model_name_error)) {
      SetError("model_name is not valid UTF-8: " + model_name_error);
      return TE9_ERR_INVALID_ARG;
    }
    if (value->model_name.find('\0') != std::string::npos) {
      SetError("model_name must not contain NUL bytes");
      return TE9_ERR_INVALID_ARG;
    }
    te9_status status = ReadProviderTable(&value->provider);
    if (status != TE9_OK) return status;

    value->source.struct_size = sizeof(value->source);
    size_t needed = 0;
    status = value->provider.get_model_llm_source(
        value->engine, value->model_name.c_str(), &value->source, nullptr, 0,
        &needed);
    if (status != TE9_OK)
      return ProviderError(status, "model metadata size query failed");

    Vocabulary vocabulary;
    if (value->source.llm_metadata_format_version == 0) {
      if (needed != 0) {
        SetError("provider returned metadata bytes with format version zero");
        return TE9_ERR_ENGINE;
      }
      // The engine exports metadata only for the tokenizers below.  Say so:
      // this used to read as a corrupt model.
      SetError(
          "libtempo9_llm cannot read this model's tokenizer: it supports GGUF "
          "tokenizer.ggml.model \"llama\", and \"gpt2\" with "
          "tokenizer.ggml.pre \"qwen2\" or \"qwen35\"");
      return TE9_ERR_UNSUPPORTED;
    } else {
      if (value->source.llm_metadata_format_version !=
              TE9_LLM_METADATA_FORMAT_V1 ||
          needed == 0) {
        SetError("provider returned an unsupported LLM metadata format");
        return TE9_ERR_UNSUPPORTED;
      }
      std::vector<uint8_t> blob(needed);
      te9_model_llm_source fetched{};
      fetched.struct_size = sizeof(fetched);
      size_t fetched_size = 0;
      status = value->provider.get_model_llm_source(
          value->engine, value->model_name.c_str(), &fetched, blob.data(),
          blob.size(), &fetched_size);
      if (status != TE9_OK)
        return ProviderError(status, "model metadata fetch failed");
      if (fetched_size != blob.size() || !SameSource(value->source, fetched)) {
        SetError("model identity changed while metadata was fetched");
        return TE9_ERR_NOT_READY;
      }
      const auto digest = Sha256(blob.data(), blob.size());
      if (std::memcmp(digest.data(), fetched.llm_metadata_sha256,
                      digest.size()) != 0) {
        SetError(
            "model metadata SHA-256 does not match the provider fingerprint");
        return TE9_ERR_ENGINE;
      }
      std::string error;
      if (!ParseMetadata(blob.data(), blob.size(), &vocabulary, &error)) {
        SetError(std::move(error));
        return TE9_ERR_ENGINE;
      }
      value->tokenizer = Tokenizer::Create(vocabulary, &error);
      if (value->tokenizer == nullptr) {
        SetError(std::move(error));
        return TE9_ERR_UNSUPPORTED;
      }
      value->source = fetched;
    }

    std::string chat_template = config->chat_template_override.size != 0
                                    ? CopyView(config->chat_template_override)
                                    : vocabulary.chat_template;
    if (!chat_template.empty()) {
      std::string error;
      if (!ValidateUtf8(chat_template.data(), chat_template.size(), &error)) {
        SetError("chat template is not valid UTF-8: " + error);
        return TE9_ERR_INVALID_ARG;
      }
      const std::string bos =
          value->tokenizer == nullptr
              ? std::string()
              : value->tokenizer->TokenText(value->tokenizer->bos_id());
      const std::string eos =
          value->tokenizer == nullptr
              ? std::string()
              : value->tokenizer->TokenText(value->tokenizer->eos_id());
      value->renderer =
          TemplateRenderer::Create(chat_template, bos, eos, &error);
      if (value->renderer == nullptr) {
        SetError(std::move(error));
        return TE9_ERR_UNSUPPORTED;
      }
    }

    te9_model_llm_source final_source{};
    final_source.struct_size = sizeof(final_source);
    size_t final_needed = 0;
    status = value->provider.get_model_llm_source(
        value->engine, value->model_name.c_str(), &final_source, nullptr, 0,
        &final_needed);
    if (status != TE9_OK)
      return ProviderError(status, "model identity recheck failed");
    if (final_needed != needed || !SameSource(value->source, final_source)) {
      SetError("model identity changed while the pipeline was created");
      return TE9_ERR_NOT_READY;
    }
    const uint64_t handle = g_pipelines.Add(std::move(value));
    if (handle == 0) {
      SetError("pipeline registry is full");
      return TE9_ERR_OUT_OF_MEMORY;
    }
    *pipeline = handle;
    return TE9_OK;
  });
}

te9_status te9_llm_pipeline_destroy(te9_llm_pipeline_t pipeline) {
  return Guard([&] {
    std::shared_ptr<PipelineRecord> value = g_pipelines.Remove(pipeline);
    if (value == nullptr) {
      SetError("pipeline handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    value->gate.Close();
    return TE9_OK;
  });
}

static te9_status PipelinePrefixCacheTransfer(te9_llm_pipeline_t pipeline,
                                              te9_llm_string_view path,
                                              uint64_t* node_count,
                                              bool importing) {
  if (node_count != nullptr) *node_count = 0;
  if (!ViewOk(path) || path.size == 0 || node_count == nullptr) {
    SetError("prefix-cache transfer path or count output is invalid");
    return TE9_ERR_INVALID_ARG;
  }
  std::string path_value = CopyView(path);
  if (path_value.find('\0') != std::string::npos) {
    SetError("prefix-cache transfer path contains an embedded NUL");
    return TE9_ERR_INVALID_ARG;
  }
  std::shared_ptr<PipelineRecord> value = g_pipelines.Get(pipeline);
  if (value == nullptr) {
    SetError("pipeline handle is invalid or stale");
    return TE9_ERR_INVALID_ARG;
  }
  auto lease = value->gate.Borrow();
  if (!lease) {
    SetError("pipeline is closing");
    return TE9_ERR_INVALID_ARG;
  }
  if ((value->provider.capabilities & TE9_ENGINE_CAP_PREFIX_CACHE_TRANSFER) ==
          0 ||
      value->provider.export_prefix_cache == nullptr ||
      value->provider.import_prefix_cache == nullptr) {
    SetError("engine provider does not support prefix-cache transfer");
    return TE9_ERR_UNSUPPORTED;
  }
  const te9_status status = importing
                                ? value->provider.import_prefix_cache(
                                      value->engine, value->model_name.c_str(),
                                      path_value.c_str(), node_count)
                                : value->provider.export_prefix_cache(
                                      value->engine, value->model_name.c_str(),
                                      path_value.c_str(), node_count);
  return status == TE9_OK
             ? TE9_OK
             : ProviderError(status, importing ? "prefix-cache import failed"
                                               : "prefix-cache export failed");
}

te9_status te9_llm_pipeline_export_prefix_cache(te9_llm_pipeline_t pipeline,
                                                te9_llm_string_view path,
                                                uint64_t* node_count) {
  return Guard([&] {
    return PipelinePrefixCacheTransfer(pipeline, path, node_count,
                                       /*importing=*/false);
  });
}

te9_status te9_llm_pipeline_import_prefix_cache(te9_llm_pipeline_t pipeline,
                                                te9_llm_string_view path,
                                                uint64_t* node_count) {
  return Guard([&] {
    return PipelinePrefixCacheTransfer(pipeline, path, node_count,
                                       /*importing=*/true);
  });
}

te9_status te9_llm_pipeline_render_chat(te9_llm_pipeline_t pipeline,
                                        const te9_llm_chat_request* request,
                                        char* utf8, size_t capacity,
                                        size_t* needed) {
  return Guard([&] {
    if (needed != nullptr) *needed = 0;
    std::shared_ptr<PipelineRecord> value = g_pipelines.Get(pipeline);
    if (value == nullptr) {
      SetError("pipeline handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("pipeline is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::string rendered;
    const te9_status status = Render(value.get(), request, &rendered);
    if (status != TE9_OK) return status;
    return CopyBytes(rendered, utf8, capacity, needed);
  });
}

te9_status te9_llm_pipeline_encode(te9_llm_pipeline_t pipeline,
                                   te9_llm_string_view utf8, int64_t* token_ids,
                                   size_t capacity, size_t* needed) {
  return Guard([&] {
    if (needed != nullptr) *needed = 0;
    if (!ViewOk(utf8) || needed == nullptr ||
        (token_ids == nullptr && capacity != 0)) {
      SetError("encode input or output buffer is invalid");
      return TE9_ERR_INVALID_ARG;
    }
    std::shared_ptr<PipelineRecord> value = g_pipelines.Get(pipeline);
    if (value == nullptr) {
      SetError("pipeline handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("pipeline is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::vector<int64_t> tokens;
    const te9_status status =
        Encode(value.get(), CopyView(utf8), true, &tokens);
    if (status != TE9_OK) return status;
    *needed = tokens.size();
    if (capacity < tokens.size()) return TE9_ERR_BUFFER_TOO_SMALL;
    if (!tokens.empty())
      std::memcpy(token_ids, tokens.data(), tokens.size() * sizeof(int64_t));
    return TE9_OK;
  });
}

te9_status te9_llm_request_start(te9_llm_pipeline_t pipeline,
                                 const te9_llm_request_config* config,
                                 te9_llm_request_t* request) {
  return Guard([&] {
    if (request != nullptr) *request = 0;
    constexpr size_t kRequestV10Size =
        offsetof(te9_llm_request_config, embeds) + sizeof(config->embeds);
    constexpr size_t kRequestV16Size =
        offsetof(te9_llm_request_config, cache_point_ttl_seconds) +
        sizeof(config->cache_point_ttl_seconds);
    constexpr size_t kRequestV17Size =
        offsetof(te9_llm_request_config, media_count) +
        sizeof(config->media_count);
    constexpr size_t kRequestV18Size =
        offsetof(te9_llm_request_config, media_kinds) +
        sizeof(config->media_kinds);
    if (!StructOk(config, kRequestV10Size) || request == nullptr ||
        (config->struct_size > kRequestV16Size &&
         config->struct_size < kRequestV17Size) ||
        (config->struct_size > kRequestV17Size &&
         config->struct_size < kRequestV18Size)) {
      SetError("te9_llm_request_config is invalid");
      return TE9_ERR_INVALID_ARG;
    }
    std::shared_ptr<PipelineRecord> pipeline_value = g_pipelines.Get(pipeline);
    if (pipeline_value == nullptr) {
      SetError("pipeline handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto pipeline_lease = pipeline_value->gate.Borrow();
    if (!pipeline_lease) {
      SetError("pipeline is closing");
      return TE9_ERR_INVALID_ARG;
    }
    const int32_t* explicit_cache_point_ttl_seconds =
        config->struct_size >=
                offsetof(te9_llm_request_config, cache_point_ttl_seconds) +
                    sizeof(config->cache_point_ttl_seconds)
            ? config->cache_point_ttl_seconds
            : nullptr;
    const bool has_media_fields = config->struct_size >= kRequestV17Size;
    const te9_llm_media_embedding* media =
        has_media_fields ? config->media : nullptr;
    const size_t media_count = has_media_fields ? config->media_count : 0;
    const int32_t* media_kinds =
        config->struct_size >= kRequestV18Size ? config->media_kinds : nullptr;
    if ((media == nullptr) != (media_count == 0)) {
      SetError(
          "media pointer and media_count must either both be set or empty");
      return TE9_ERR_INVALID_ARG;
    }
    if (media_count == 0 && media_kinds != nullptr) {
      SetError("media_kinds requires a non-empty media array");
      return TE9_ERR_INVALID_ARG;
    }
    if (media_count != 0 && config->embeds != nullptr) {
      SetError("high-level media and raw embeds cannot be mixed");
      return TE9_ERR_INVALID_ARG;
    }
    if (media_count != 0 && config->input_kind == TE9_LLM_INPUT_UTF8) {
      SetError("precomputed media requires chat or token-id input");
      return TE9_ERR_INVALID_ARG;
    }
    if (explicit_cache_point_ttl_seconds != nullptr &&
        (pipeline_value->provider.capabilities &
         TE9_ENGINE_CAP_PROMPT_CACHE_TTL) == 0) {
      SetError("engine provider does not support prompt-cache TTL");
      return TE9_ERR_UNSUPPORTED;
    }
    if (pipeline_value->tokenizer == nullptr) {
      SetError("text generation requires reconstructable tokenizer metadata");
      return TE9_ERR_UNSUPPORTED;
    }

    std::vector<int64_t> prompt;
    std::vector<te9_prompt_cache_point> derived_cache_points;
    std::vector<int32_t> derived_cache_point_ttls;
    te9_status status = TE9_OK;
    if (config->input_kind == TE9_LLM_INPUT_TOKEN_IDS) {
      if (config->prompt_ids == nullptr || config->prompt_id_count == 0) {
        SetError("token-id input must contain at least one token");
        return TE9_ERR_INVALID_ARG;
      }
      prompt.assign(config->prompt_ids,
                    config->prompt_ids + config->prompt_id_count);
    } else if (config->input_kind == TE9_LLM_INPUT_UTF8) {
      if (!ViewOk(config->prompt_utf8) || config->prompt_utf8.size == 0) {
        SetError("UTF-8 input must be non-empty");
        return TE9_ERR_INVALID_ARG;
      }
      status = Encode(pipeline_value.get(), CopyView(config->prompt_utf8), true,
                      &prompt);
      if (status != TE9_OK) return status;
    } else if (config->input_kind == TE9_LLM_INPUT_CHAT) {
      std::string rendered;
      std::vector<RenderedCachePoint> rendered_cache_points;
      status = Render(pipeline_value.get(), config->chat, &rendered,
                      &rendered_cache_points);
      if (status != TE9_OK) return status;
      size_t media_part_count = 0;
      for (size_t i = 0; i != config->chat->message_count; ++i) {
        const te9_llm_chat_message& message = config->chat->messages[i];
        for (size_t j = 0; j != message.content_count; ++j) {
          const int32_t part_kind = message.content[j].kind;
          if (part_kind == TE9_LLM_CHAT_PART_IMAGE ||
              part_kind == TE9_LLM_CHAT_PART_AUDIO) {
            if (media_part_count == std::numeric_limits<size_t>::max()) {
              SetError("chat media part count exceeds the host size range");
              return TE9_ERR_INVALID_ARG;
            }
            if (media_part_count >= media_count) {
              SetError("chat media part count exceeds media embedding count");
              return TE9_ERR_INVALID_ARG;
            }
            const int32_t source_kind = media_kinds == nullptr
                                            ? TE9_LLM_MEDIA_IMAGE
                                            : media_kinds[media_part_count];
            const int32_t expected_kind = part_kind == TE9_LLM_CHAT_PART_AUDIO
                                              ? TE9_LLM_MEDIA_AUDIO
                                              : TE9_LLM_MEDIA_IMAGE;
            if (source_kind != expected_kind) {
              SetError("chat media part kind does not match embedding " +
                       std::to_string(media_part_count));
              return TE9_ERR_INVALID_ARG;
            }
            ++media_part_count;
          }
        }
      }
      if (media_part_count != media_count) {
        SetError("chat media part count " + std::to_string(media_part_count) +
                 " does not match media embedding count " +
                 std::to_string(media_count));
        return TE9_ERR_INVALID_ARG;
      }
      status = Encode(pipeline_value.get(), rendered, false, &prompt);
      if (status != TE9_OK) return status;
      if (!rendered_cache_points.empty() &&
          (pipeline_value->provider.capabilities &
           TE9_ENGINE_CAP_PROMPT_CACHE_TTL) == 0) {
        SetError("engine provider does not support chat cache-control TTL");
        return TE9_ERR_UNSUPPORTED;
      }
      for (const RenderedCachePoint& rendered_point : rendered_cache_points) {
        std::vector<int64_t> prefix;
        std::vector<int64_t> previous_prefix;
        status =
            Encode(pipeline_value.get(), rendered_point.prefix, false, &prefix);
        if (status != TE9_OK) return status;
        status = Encode(pipeline_value.get(), rendered_point.previous_prefix,
                        false, &previous_prefix);
        if (status != TE9_OK) return status;
        const auto mismatch = std::mismatch(prefix.begin(), prefix.end(),
                                            prompt.begin(), prompt.end());
        const size_t offset =
            static_cast<size_t>(mismatch.second - prompt.begin());
        const auto previous_mismatch =
            std::mismatch(previous_prefix.begin(), previous_prefix.end(),
                          prompt.begin(), prompt.end());
        const size_t previous_offset =
            static_cast<size_t>(previous_mismatch.second - prompt.begin());
        if (offset <= previous_offset) {
          SetError(
              "chat cache breakpoint does not add a stable rendered token "
              "prefix for this template");
          return TE9_ERR_UNSUPPORTED;
        }
        if (!derived_cache_points.empty() &&
            offset <= derived_cache_points.back().token_offset) {
          if (offset == derived_cache_points.back().token_offset &&
              rendered_point.ttl_seconds == derived_cache_point_ttls.back()) {
            continue;
          }
          SetError(
              "chat cache breakpoints do not map to strictly increasing "
              "token offsets for this template");
          return TE9_ERR_UNSUPPORTED;
        }
        te9_prompt_cache_point point{};
        point.struct_size = sizeof(point);
        point.token_offset = offset;
        derived_cache_points.push_back(point);
        derived_cache_point_ttls.push_back(rendered_point.ttl_seconds);
      }
    } else {
      SetError("request input_kind is unknown");
      return TE9_ERR_INVALID_ARG;
    }
    if (prompt.empty()) {
      SetError("request input tokenized to an empty prompt");
      return TE9_ERR_INVALID_ARG;
    }
    if (config->input_kind != TE9_LLM_INPUT_TOKEN_IDS &&
        config->cache_point_count != 0) {
      SetError("explicit cache points are valid only with token-id input");
      return TE9_ERR_INVALID_ARG;
    }

    PreparedMedia prepared_media;
    std::vector<te9_prompt_cache_point> expanded_cache_points;
    const te9_image_embeds* provider_embeds = config->embeds;
    if (media_count != 0) {
      status = PrepareMedia(media, media_kinds, media_count, &prompt,
                            &prepared_media);
      if (status != TE9_OK) return status;
      provider_embeds = prepared_media.provider.data();
      const te9_prompt_cache_point* original_points =
          derived_cache_points.empty() ? config->cache_points
                                       : derived_cache_points.data();
      const size_t original_count = derived_cache_points.empty()
                                        ? config->cache_point_count
                                        : derived_cache_points.size();
      expanded_cache_points.reserve(original_count);
      for (size_t i = 0; i != original_count; ++i) {
        if (original_points == nullptr ||
            !StructOk(&original_points[i], sizeof(te9_prompt_cache_point)) ||
            original_points[i].token_offset == 0 ||
            original_points[i].token_offset >=
                prepared_media.offset_map.size()) {
          SetError("media cache point is outside the unexpanded prompt");
          return TE9_ERR_INVALID_ARG;
        }
        te9_prompt_cache_point point{};
        point.struct_size = sizeof(point);
        point.token_offset =
            prepared_media.offset_map[original_points[i].token_offset];
        expanded_cache_points.push_back(point);
      }
    }

    te9_generate_config generation{};
    if (config->generation != nullptr) {
      if (!StructOk(config->generation,
                    offsetof(te9_generate_config, top_k) + sizeof(int32_t))) {
        SetError("te9_generate_config has an invalid struct_size");
        return TE9_ERR_INVALID_ARG;
      }
      std::memcpy(&generation, config->generation,
                  std::min(config->generation->struct_size,
                           sizeof(te9_generate_config)));
      generation.struct_size = sizeof(generation);
    } else {
      generation.struct_size = sizeof(generation);
      generation.max_tokens = 256;
      generation.temperature = 0.8f;
      generation.top_p = 0.95f;
      generation.top_k = 50;
      generation.repetition_penalty = 1.0f;
      generation.do_sample = 1;
    }
    if (generation.eos_token_id == 0 && pipeline_value->tokenizer != nullptr &&
        pipeline_value->tokenizer->eos_id() > 0)
      generation.eos_token_id = pipeline_value->tokenizer->eos_id();

    auto value = std::make_shared<RequestRecord>();
    value->pipeline = pipeline_value;
    std::vector<te9_token_sequence> provider_stops;
    const bool has_new_stops =
        config->stop_string_count != 0 || config->stop_sequence_count != 0;
    if (has_new_stops && (generation.stop_token_count != 0 ||
                          generation.stop_sequence_count != 0)) {
      SetError("legacy stop settings cannot be mixed with LLM stop sequences");
      return TE9_ERR_INVALID_ARG;
    }
    if (config->stop_string_count != 0 && config->stop_strings == nullptr) {
      SetError("stop_strings is null with a nonzero count");
      return TE9_ERR_INVALID_ARG;
    }
    if (config->stop_sequence_count != 0 && config->stop_sequences == nullptr) {
      SetError("stop_sequences is null with a nonzero count");
      return TE9_ERR_INVALID_ARG;
    }
    value->stops.reserve(config->stop_string_count +
                         config->stop_sequence_count);
    for (size_t i = 0; i != config->stop_string_count; ++i) {
      if (!ViewOk(config->stop_strings[i]) ||
          config->stop_strings[i].size == 0) {
        SetError("stop strings must be non-empty UTF-8 values");
        return TE9_ERR_INVALID_ARG;
      }
      std::vector<int64_t> stop;
      status = Encode(pipeline_value.get(), CopyView(config->stop_strings[i]),
                      false, &stop);
      if (status != TE9_OK) return status;
      if (stop.empty()) {
        SetError("a stop string tokenized to an empty sequence");
        return TE9_ERR_INVALID_ARG;
      }
      value->stops.push_back(std::move(stop));
    }
    for (size_t i = 0; i != config->stop_sequence_count; ++i) {
      const te9_llm_token_sequence& stop = config->stop_sequences[i];
      if (!StructOk(&stop, sizeof(stop)) || stop.ids == nullptr ||
          stop.count == 0) {
        SetError("LLM stop token sequence is invalid");
        return TE9_ERR_INVALID_ARG;
      }
      value->stops.emplace_back(stop.ids, stop.ids + stop.count);
    }
    provider_stops.reserve(value->stops.size());
    for (const std::vector<int64_t>& stop : value->stops) {
      te9_token_sequence provider_stop{};
      provider_stop.struct_size = sizeof(provider_stop);
      provider_stop.token_ids = stop.data();
      provider_stop.token_count = stop.size();
      provider_stops.push_back(provider_stop);
      value->max_stop_length = std::max(value->max_stop_length, stop.size());
    }
    if (has_new_stops) {
      generation.stop_token_ids = nullptr;
      generation.stop_token_count = 0;
      generation.stop_sequences = provider_stops.data();
      generation.stop_sequence_count = provider_stops.size();
    }

    std::string response_kind;
    std::string response_schema;
    if (config->response_format != nullptr) {
      if (!StructOk(config->response_format, sizeof(te9_llm_response_format)) ||
          (generation.response_format != nullptr &&
           generation.response_format[0] != '\0') ||
          (generation.response_schema != nullptr &&
           generation.response_schema[0] != '\0')) {
        SetError("new and legacy response formats cannot be mixed");
        return TE9_ERR_INVALID_ARG;
      }
      const te9_llm_response_format& format = *config->response_format;
      if (format.strict != 0 && format.strict != 1) {
        SetError("response-format strict must be zero or one");
        return TE9_ERR_INVALID_ARG;
      }
      if (format.kind == TE9_LLM_RESPONSE_TEXT) {
        if (format.schema_json.size != 0 || format.schema_name.size != 0 ||
            format.strict != 0) {
          SetError("text response format cannot carry a schema");
          return TE9_ERR_INVALID_ARG;
        }
      } else if (format.kind == TE9_LLM_RESPONSE_JSON_OBJECT) {
        if (format.schema_json.size != 0 || format.schema_name.size != 0 ||
            format.strict != 0) {
          SetError("JSON-object response format cannot carry schema fields");
          return TE9_ERR_INVALID_ARG;
        }
        response_kind = "json_object";
      } else if (format.kind == TE9_LLM_RESPONSE_JSON_SCHEMA) {
        if (!ViewOk(format.schema_name) || format.schema_name.size == 0 ||
            !ViewOk(format.schema_json) || format.schema_json.size == 0) {
          SetError("JSON schema response format requires name and schema");
          return TE9_ERR_INVALID_ARG;
        }
        if (format.strict == 0) {
          SetError("non-strict JSON schema output is unsupported");
          return TE9_ERR_UNSUPPORTED;
        }
        std::string error;
        if (!ValidateUtf8(format.schema_name.data, format.schema_name.size,
                          &error)) {
          SetError("response schema name is not valid UTF-8: " + error);
          return TE9_ERR_INVALID_ARG;
        }
        response_schema = CopyView(format.schema_json);
        if (!ParseJsonObject(response_schema, "response schema", &error)) {
          SetError(std::move(error));
          return TE9_ERR_INVALID_ARG;
        }
        response_kind = "json_schema";
      } else {
        SetError("response-format kind is unknown");
        return TE9_ERR_INVALID_ARG;
      }
      generation.response_format =
          response_kind.empty() ? nullptr : response_kind.c_str();
      generation.response_schema =
          response_schema.empty() ? nullptr : response_schema.c_str();
    }

    te9_checked_request_config checked{};
    checked.struct_size = sizeof(checked);
    checked.model_name = pipeline_value->model_name.c_str();
    checked.expected_engine_generation =
        pipeline_value->source.engine_generation;
    checked.expected_model_generation = pipeline_value->source.model_generation;
    checked.input_ids = prompt.data();
    checked.input_id_count = prompt.size();
    checked.generation = &generation;
    checked.cache_points =
        !expanded_cache_points.empty()
            ? expanded_cache_points.data()
            : (derived_cache_points.empty() ? config->cache_points
                                            : derived_cache_points.data());
    checked.cache_point_count =
        !expanded_cache_points.empty()
            ? expanded_cache_points.size()
            : (derived_cache_points.empty() ? config->cache_point_count
                                            : derived_cache_points.size());
    checked.embeds = provider_embeds;
    checked.cache_point_ttl_seconds = derived_cache_point_ttls.empty()
                                          ? explicit_cache_point_ttl_seconds
                                          : derived_cache_point_ttls.data();
    status = pipeline_value->provider.request_start_checked(
        pipeline_value->engine, &checked, &value->provider_request);
    if (status != TE9_OK)
      return ProviderError(status, "checked request admission failed");
    constexpr size_t kMaximumPendingReserve = 16u * 1024u * 1024u;
    const size_t piece_bytes =
        pipeline_value->tokenizer == nullptr
            ? 4u
            : pipeline_value->tokenizer->max_piece_bytes();
    const size_t batch_bytes = piece_bytes > kMaximumPendingReserve / 64u
                                   ? kMaximumPendingReserve
                                   : piece_bytes * 64u;
    const size_t reserve =
        std::max<size_t>(256u, std::min(batch_bytes, kMaximumPendingReserve));
    value->pending.reserve(reserve + 4u);
    value->generated_tail.reserve(value->max_stop_length);
    const uint64_t handle = g_requests.Add(std::move(value));
    if (handle == 0) {
      SetError("request registry is full");
      return TE9_ERR_OUT_OF_MEMORY;
    }
    *request = handle;
    return TE9_OK;
  });
}

te9_status te9_llm_request_read(te9_llm_request_t request, int32_t timeout_ms,
                                char* utf8, size_t capacity, size_t* written) {
  return Guard([&] {
    if (written != nullptr) *written = 0;
    if (written == nullptr || (utf8 == nullptr && capacity != 0) ||
        timeout_ms < 0) {
      SetError("read output buffer, count, or timeout is invalid");
      return TE9_ERR_INVALID_ARG;
    }
    std::shared_ptr<RequestRecord> value = g_requests.Get(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("request is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::unique_lock<std::mutex> read_lock(value->read_mutex, std::try_to_lock);
    if (!read_lock.owns_lock()) {
      SetError("only one thread may read an LLM request");
      return TE9_ERR_INVALID_ARG;
    }
    if (value->stopped.load(std::memory_order_acquire)) return TE9_OK;
    if (capacity == 0) return TE9_OK;

    bool invalid = false;
    size_t complete = CompleteUtf8Prefix(value->pending, &invalid);
    if (invalid) {
      SetError("tokenizer produced invalid UTF-8");
      return TE9_ERR_ENGINE;
    }
    if (complete == 0) {
      std::array<int64_t, 64> tokens{};
      size_t count = 0;
      te9_status status;
      {
        // Serialize ordinary provider operations while allowing stop() to run
        // concurrently and wake a blocked wait.
        std::lock_guard<std::mutex> provider_lock(value->provider_mutex);
        status = te9_request_wait(value->provider_request, timeout_ms,
                                  tokens.data(), tokens.size(), &count);
      }
      if (status != TE9_OK) return ProviderError(status, "request wait failed");
      if (value->stopped.load(std::memory_order_acquire)) return TE9_OK;
      TrackStops(value.get(), tokens.data(), count);
      if (count != 0 && value->pipeline->tokenizer == nullptr) {
        SetError("token-id-only pipeline cannot decode generated text");
        return TE9_ERR_UNSUPPORTED;
      }
      for (size_t i = 0; i != count; ++i) {
        std::string error;
        if (!value->pipeline->tokenizer->AppendDecoded(
                tokens[i], &value->pending, &error)) {
          SetError(std::move(error));
          return TE9_ERR_ENGINE;
        }
        if (value->initial_output) {
          value->initial_output = false;
          if (value->pipeline->tokenizer->trim_initial_space() &&
              !value->pending.empty() && value->pending.front() == ' ')
            value->pending.erase(value->pending.begin());
        }
      }
      complete = CompleteUtf8Prefix(value->pending, &invalid);
      if (invalid) {
        SetError("tokenizer produced invalid UTF-8");
        return TE9_ERR_ENGINE;
      }
      if (complete == 0 && count == 0 && !value->pending.empty()) {
        te9_gen_status generation_status = TE9_GEN_RUNNING;
        {
          std::lock_guard<std::mutex> provider_lock(value->provider_mutex);
          status =
              te9_request_status(value->provider_request, &generation_status);
        }
        if (status != TE9_OK)
          return ProviderError(status, "request status failed");
        if (generation_status != TE9_GEN_RUNNING) {
          SetError("generation ended with an incomplete UTF-8 sequence");
          return TE9_ERR_ENGINE;
        }
      }
    }
    const size_t count =
        PrefixWithinCapacity(value->pending, complete, capacity);
    if (complete != 0 && count == 0) {
      SetError("read buffer cannot hold the next complete UTF-8 code point");
      return TE9_ERR_BUFFER_TOO_SMALL;
    }
    if (count != 0) {
      std::memcpy(utf8, value->pending.data(), count);
      value->pending.erase(0, count);
    }
    *written = count;
    return TE9_OK;
  });
}

te9_status te9_llm_request_status(te9_llm_request_t request,
                                  te9_gen_status* status) {
  return Guard([&] {
    if (status == nullptr) {
      SetError("request status output pointer is null");
      return TE9_ERR_INVALID_ARG;
    }
    std::shared_ptr<RequestRecord> value = g_requests.Get(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("request is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> read_lock(value->read_mutex);
    if (value->stopped.load(std::memory_order_acquire)) {
      *status = TE9_GEN_INTERRUPTED;
      return TE9_OK;
    }
    if (!value->pending.empty()) {
      *status = TE9_GEN_RUNNING;
      return TE9_OK;
    }
    std::lock_guard<std::mutex> lock(value->provider_mutex);
    const te9_status result =
        te9_request_status(value->provider_request, status);
    return result == TE9_OK ? TE9_OK
                            : ProviderError(result, "request status failed");
  });
}

te9_status te9_llm_request_finish_reason(te9_llm_request_t request,
                                         te9_finish_reason* reason) {
  return Guard([&] {
    if (reason == nullptr) {
      SetError("finish-reason output pointer is null");
      return TE9_ERR_INVALID_ARG;
    }
    std::shared_ptr<RequestRecord> value = g_requests.Get(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("request is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> read_lock(value->read_mutex);
    if (value->stopped.load(std::memory_order_acquire)) {
      *reason = TE9_FINISH_INTERRUPTED;
      return TE9_OK;
    }
    if (!value->pending.empty()) {
      *reason = TE9_FINISH_NONE;
      return TE9_OK;
    }
    std::lock_guard<std::mutex> lock(value->provider_mutex);
    const te9_status result =
        te9_request_finish_reason(value->provider_request, reason);
    return result == TE9_OK
               ? TE9_OK
               : ProviderError(result, "request finish-reason query failed");
  });
}

te9_status te9_llm_request_get_stats(te9_llm_request_t request,
                                     te9_request_stats* stats) {
  return Guard([&] {
    if (stats == nullptr) {
      SetError("request stats output pointer is null");
      return TE9_ERR_INVALID_ARG;
    }
    std::shared_ptr<RequestRecord> value = g_requests.Get(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("request is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(value->provider_mutex);
    const te9_status result =
        te9_request_get_stats(value->provider_request, stats);
    return result == TE9_OK
               ? TE9_OK
               : ProviderError(result, "request stats query failed");
  });
}

te9_status te9_llm_request_matched_stop(te9_llm_request_t request,
                                        size_t* stop_index) {
  return Guard([&] {
    if (stop_index == nullptr) {
      SetError("matched-stop output pointer is null");
      return TE9_ERR_INVALID_ARG;
    }
    *stop_index = SIZE_MAX;
    std::shared_ptr<RequestRecord> value = g_requests.Get(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("request is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> read_lock(value->read_mutex);
    if (!value->pending.empty()) return TE9_OK;
    te9_finish_reason reason = TE9_FINISH_NONE;
    {
      std::lock_guard<std::mutex> provider_lock(value->provider_mutex);
      const te9_status status =
          te9_request_finish_reason(value->provider_request, &reason);
      if (status != TE9_OK)
        return ProviderError(status, "request finish-reason query failed");
    }
    if (reason == TE9_FINISH_STOP) *stop_index = value->matched_stop;
    return TE9_OK;
  });
}

te9_status te9_llm_request_stop(te9_llm_request_t request) {
  return Guard([&] {
    std::shared_ptr<RequestRecord> value = g_requests.Get(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    auto lease = value->gate.Borrow();
    if (!lease) {
      SetError("request is closing");
      return TE9_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> stop_lock(value->stop_mutex);
    if (value->stopped.load(std::memory_order_acquire)) return TE9_OK;

    // A capable provider lets stop wake a blocked wait. Older ABI 1.1
    // providers retain the original serialized request-handle contract.
    te9_status result = TE9_OK;
    if ((value->pipeline->provider.capabilities &
         TE9_ENGINE_CAP_CONCURRENT_REQUEST_STOP) != 0) {
      // The operation gate keeps provider_request alive until both calls
      // return.
      result = te9_request_stop(value->provider_request);
    } else {
      std::lock_guard<std::mutex> provider_lock(value->provider_mutex);
      result = te9_request_stop(value->provider_request);
    }
    if (result != TE9_OK) return ProviderError(result, "request stop failed");

    value->stopped.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> read_lock(value->read_mutex);
    value->pending.clear();
    return TE9_OK;
  });
}

te9_status te9_llm_request_release(te9_llm_request_t request) {
  return Guard([&] {
    std::shared_ptr<RequestRecord> value = g_requests.Remove(request);
    if (value == nullptr) {
      SetError("request handle is invalid or stale");
      return TE9_ERR_INVALID_ARG;
    }
    value->gate.Close();
    std::lock_guard<std::mutex> lock(value->provider_mutex);
    te9_request_release(value->provider_request);
    value->provider_request = nullptr;
    return TE9_OK;
  });
}

}  // extern "C"
