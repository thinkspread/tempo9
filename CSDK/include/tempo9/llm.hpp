/* Copyright (c) 2026 Jiejing Zhang. */

#pragma once

#include <tempo9/llm.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace tempo9 {

class Status {
 public:
  Status() = default;
  Status(te9_status code, std::string message)
      : code_(code), message_(std::move(message)) {}

  bool ok() const noexcept { return code_ == TE9_OK; }
  explicit operator bool() const noexcept { return ok(); }
  te9_status code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }

  static Status Engine(te9_status code) {
    return code == TE9_OK
               ? Status{}
               : Status(code,
                        te9_last_error() == nullptr ? "" : te9_last_error());
  }
  static Status Llm(te9_status code) {
    return code == TE9_OK ? Status{}
                          : Status(code, te9_llm_last_error() == nullptr
                                             ? ""
                                             : te9_llm_last_error());
  }

 private:
  te9_status code_ = TE9_OK;
  std::string message_;
};

template <typename T>
class StatusOr {
 public:
  StatusOr(Status status) : status_(std::move(status)) {
    if (status_.ok())
      status_ = Status(TE9_ERR_UNKNOWN, "StatusOr received OK without a value");
  }
  StatusOr(T value) : value_(std::move(value)) {}

  bool ok() const noexcept { return value_.has_value(); }
  explicit operator bool() const noexcept { return ok(); }
  const Status& status() const noexcept { return status_; }
  T& value() & {
    if (!value_) throw std::logic_error(status_.message());
    return *value_;
  }
  const T& value() const& {
    if (!value_) throw std::logic_error(status_.message());
    return *value_;
  }
  T&& value() && {
    if (!value_) throw std::logic_error(status_.message());
    return std::move(*value_);
  }
  T value_or_throw() && {
    if (!value_) throw std::runtime_error(status_.message());
    return std::move(*value_);
  }

 private:
  Status status_;
  std::optional<T> value_;
};

struct ModelConfig {
  std::string model_name;
  std::string graph_path;
  std::string weights_path;
  std::string compute_unit = "CPU:0";
  int64_t max_length = 0;
  int32_t max_batch = 0;
  bool enable_prefix_cache = false;
  int32_t speculation_k = 0;
  int32_t prefix_cache_ttl_seconds = 300;
};

namespace detail {

struct EngineState {
  ~EngineState() noexcept { CloseNoexcept(); }

  void CloseNoexcept() noexcept {
    if (handle == nullptr) return;
    if (model_started && !model_name.empty())
      (void)te9_engine_stop_model(handle, model_name.c_str());
    if (!model_name.empty())
      (void)te9_engine_release_model(handle, model_name.c_str());
    (void)te9_engine_destroy(handle);
    handle = nullptr;
    model_name.clear();
    model_started = false;
  }

  mutable std::mutex operation;
  te9_engine_t handle = nullptr;
  std::string model_name;
  bool model_started = false;
};

struct ModelLoadCompletion {
  std::mutex mutex;
  std::condition_variable changed;
  bool done = false;
  Status result{TE9_ERR_UNKNOWN, "model load has not completed"};
};

inline Status ValidateModelConfig(const ModelConfig& value) {
  if (value.prefix_cache_ttl_seconds < 0) {
    return Status(TE9_ERR_INVALID_ARG,
                  "prefix_cache_ttl_seconds must be non-negative");
  }
  if (value.enable_prefix_cache && value.prefix_cache_ttl_seconds != 0 &&
      value.prefix_cache_ttl_seconds != 300) {
    te9_engine_api_table table{};
    const te9_status status = te9_engine_get_api_table(sizeof(table), &table);
    if (status != TE9_OK) return Status::Engine(status);
    if ((table.capabilities & TE9_ENGINE_CAP_PROMPT_CACHE_TTL) == 0) {
      return Status(TE9_ERR_UNSUPPORTED,
                    "provider does not support a non-default prefix-cache "
                    "TTL");
    }
  }
  return {};
}

inline Status BuildModelLocked(EngineState* state, const ModelConfig& value,
                               te9_engine_cancel_token_t token = 0) {
  const Status validation = ValidateModelConfig(value);
  if (!validation) return validation;
  if (state == nullptr || state->handle == nullptr)
    return Status(TE9_ERR_INVALID_ARG, "engine is closed");

  te9_model_config config{};
  config.struct_size = sizeof(config);
  config.model_name = value.model_name.c_str();
  config.graph_path =
      value.graph_path.empty() ? nullptr : value.graph_path.c_str();
  config.weights_path = value.weights_path.c_str();
  config.compute_unit =
      value.compute_unit.empty() ? nullptr : value.compute_unit.c_str();
  config.max_length = value.max_length;
  config.max_batch = value.max_batch;
  config.enable_prefix_cache = value.enable_prefix_cache ? 1 : 0;
  config.speculation_k = value.speculation_k;
  config.prefix_cache_ttl_seconds = value.prefix_cache_ttl_seconds;
  config.load_cancel_token = token;
  const te9_status status = te9_engine_build_model(state->handle, &config);
  if (status == TE9_OK) {
    state->model_name = value.model_name;
    state->model_started = false;
  }
  return Status::Engine(status);
}

inline Status StartModelLocked(EngineState* state,
                               std::string_view model_name) {
  if (state == nullptr || state->handle == nullptr)
    return Status(TE9_ERR_INVALID_ARG, "engine is closed");
  const std::string name(model_name);
  const te9_status status = te9_engine_start_model(state->handle, name.c_str());
  if (status == TE9_OK && name == state->model_name)
    state->model_started = true;
  return Status::Engine(status);
}

inline void ReleaseFailedLoadLocked(EngineState* state,
                                    std::string_view model_name) noexcept {
  if (state == nullptr || state->handle == nullptr) return;
  const std::string name(model_name);
  if (te9_engine_release_model(state->handle, name.c_str()) == TE9_OK &&
      name == state->model_name) {
    state->model_name.clear();
    state->model_started = false;
  }
}

}  // namespace detail

class ModelLoad {
 public:
  ModelLoad() = default;
  ModelLoad(const ModelLoad&) = delete;
  ModelLoad& operator=(const ModelLoad&) = delete;
  ModelLoad(ModelLoad&& other) noexcept
      : state_(std::move(other.state_)),
        completion_(std::move(other.completion_)),
        worker_(std::move(other.worker_)),
        token_(std::exchange(other.token_, 0)),
        cancel_(std::exchange(other.cancel_, nullptr)),
        destroy_(std::exchange(other.destroy_, nullptr)) {}
  ModelLoad& operator=(ModelLoad&& other) noexcept {
    if (this != &other) {
      CloseNoexcept();
      state_ = std::move(other.state_);
      completion_ = std::move(other.completion_);
      worker_ = std::move(other.worker_);
      token_ = std::exchange(other.token_, 0);
      cancel_ = std::exchange(other.cancel_, nullptr);
      destroy_ = std::exchange(other.destroy_, nullptr);
    }
    return *this;
  }
  ~ModelLoad() noexcept { CloseNoexcept(); }

  Status Cancel() {
    if (ready()) return {};
    if (token_ == 0 || cancel_ == nullptr) return {};
    return Status::Engine(cancel_(token_));
  }

  bool ready() const noexcept {
    if (completion_ == nullptr) return true;
    std::lock_guard<std::mutex> lock(completion_->mutex);
    return completion_->done;
  }

  template <typename Rep, typename Period>
  bool WaitFor(const std::chrono::duration<Rep, Period>& timeout) const {
    if (completion_ == nullptr) return true;
    std::unique_lock<std::mutex> lock(completion_->mutex);
    return completion_->changed.wait_for(lock, timeout,
                                         [this] { return completion_->done; });
  }

  Status Wait() {
    if (worker_.joinable()) worker_.join();
    if (completion_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "model load is empty");
    std::lock_guard<std::mutex> lock(completion_->mutex);
    return completion_->result;
  }

  explicit operator bool() const noexcept { return completion_ != nullptr; }

 private:
  friend class Engine;
  ModelLoad(std::shared_ptr<detail::EngineState> state,
            std::shared_ptr<detail::ModelLoadCompletion> completion,
            std::thread worker, te9_engine_cancel_token_t token,
            te9_engine_cancel_token_cancel_fn cancel,
            te9_engine_cancel_token_destroy_fn destroy)
      : state_(std::move(state)),
        completion_(std::move(completion)),
        worker_(std::move(worker)),
        token_(token),
        cancel_(cancel),
        destroy_(destroy) {}

  void CloseNoexcept() noexcept {
    if (worker_.joinable()) {
      if (token_ != 0 && cancel_ != nullptr) (void)cancel_(token_);
      worker_.join();
    }
    if (token_ != 0 && destroy_ != nullptr) (void)destroy_(token_);
    token_ = 0;
    cancel_ = nullptr;
    destroy_ = nullptr;
    state_.reset();
    completion_.reset();
  }

  std::shared_ptr<detail::EngineState> state_;
  std::shared_ptr<detail::ModelLoadCompletion> completion_;
  std::thread worker_;
  te9_engine_cancel_token_t token_ = 0;
  te9_engine_cancel_token_cancel_fn cancel_ = nullptr;
  te9_engine_cancel_token_destroy_fn destroy_ = nullptr;
};

class Engine {
 public:
  Engine() = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) noexcept = default;
  Engine& operator=(Engine&& other) noexcept {
    if (this != &other) {
      state_ = std::move(other.state_);
    }
    return *this;
  }
  ~Engine() = default;

  static StatusOr<Engine> Create() {
    Engine engine;
    auto state = std::make_shared<detail::EngineState>();
    const te9_status status = te9_engine_create(&state->handle);
    if (status != TE9_OK) return Status::Engine(status);
    engine.state_ = std::move(state);
    return engine;
  }

  Status BuildModel(const ModelConfig& value) {
    if (state_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    std::lock_guard<std::mutex> lock(state_->operation);
    return detail::BuildModelLocked(state_.get(), value);
  }

  StatusOr<ModelLoad> LoadModelAsync(ModelConfig value) {
    if (state_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    const Status validation = detail::ValidateModelConfig(value);
    if (!validation) return validation;

    te9_engine_api_table table{};
    const te9_status table_status =
        te9_engine_get_api_table(sizeof(table), &table);
    if (table_status != TE9_OK) return Status::Engine(table_status);
    const size_t required =
        offsetof(te9_engine_api_table, cancel_token_destroy) +
        sizeof(table.cancel_token_destroy);
    if (table.struct_size < required ||
        (table.capabilities & TE9_ENGINE_CAP_CANCELABLE_MODEL_LOAD) == 0 ||
        table.cancel_token_create == nullptr ||
        table.cancel_token_cancel == nullptr ||
        table.cancel_token_destroy == nullptr) {
      return Status(TE9_ERR_UNSUPPORTED,
                    "provider does not support cancellable model loading");
    }

    te9_engine_cancel_token_t token = 0;
    const te9_status create_status = table.cancel_token_create(&token);
    if (create_status != TE9_OK) return Status::Engine(create_status);

    auto state = state_;
    auto completion = std::make_shared<detail::ModelLoadCompletion>();
    std::thread worker;
    try {
      worker = std::thread([state, completion, value = std::move(value),
                            token] {
        Status result;
        try {
          std::lock_guard<std::mutex> lock(state->operation);
          result = detail::BuildModelLocked(state.get(), value, token);
          if (result) {
            result = detail::StartModelLocked(state.get(), value.model_name);
            if (!result)
              detail::ReleaseFailedLoadLocked(state.get(), value.model_name);
          }
        } catch (const std::exception& error) {
          result = Status(TE9_ERR_EXCEPTION, error.what());
        } catch (...) {
          result = Status(TE9_ERR_EXCEPTION,
                          "unknown exception during asynchronous model load");
        }
        {
          std::lock_guard<std::mutex> lock(completion->mutex);
          completion->result = std::move(result);
          completion->done = true;
        }
        completion->changed.notify_all();
      });
    } catch (const std::exception& error) {
      (void)table.cancel_token_destroy(token);
      return Status(TE9_ERR_EXCEPTION, error.what());
    } catch (...) {
      (void)table.cancel_token_destroy(token);
      return Status(TE9_ERR_EXCEPTION,
                    "cannot start asynchronous model-load thread");
    }
    return ModelLoad(std::move(state), std::move(completion), std::move(worker),
                     token, table.cancel_token_cancel,
                     table.cancel_token_destroy);
  }

  Status StartModel(std::string_view model_name) {
    if (state_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    std::lock_guard<std::mutex> lock(state_->operation);
    return detail::StartModelLocked(state_.get(), model_name);
  }
  Status StopModel(std::string_view model_name) {
    if (state_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    std::lock_guard<std::mutex> lock(state_->operation);
    const std::string name(model_name);
    const te9_status status =
        te9_engine_stop_model(state_->handle, name.c_str());
    if (status == TE9_OK && name == state_->model_name)
      state_->model_started = false;
    return Status::Engine(status);
  }
  Status ReleaseModel(std::string_view model_name) {
    if (state_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    std::lock_guard<std::mutex> lock(state_->operation);
    const std::string name(model_name);
    const te9_status status =
        te9_engine_release_model(state_->handle, name.c_str());
    if (status == TE9_OK && name == state_->model_name) {
      state_->model_name.clear();
      state_->model_started = false;
    }
    return Status::Engine(status);
  }
  Status Close() {
    if (state_ == nullptr) return {};
    const std::shared_ptr<detail::EngineState> state = state_;
    te9_status status = TE9_OK;
    {
      std::lock_guard<std::mutex> lock(state->operation);
      if (state->handle != nullptr && !state->model_name.empty()) {
        if (state->model_started) {
          const te9_status stopped =
              te9_engine_stop_model(state->handle, state->model_name.c_str());
          if (stopped != TE9_OK) return Status::Engine(stopped);
          state->model_started = false;
        }
        const te9_status released =
            te9_engine_release_model(state->handle, state->model_name.c_str());
        if (released != TE9_OK) return Status::Engine(released);
        state->model_name.clear();
      }
      if (state->handle != nullptr) {
        status = te9_engine_destroy(state->handle);
        if (status == TE9_OK) state->handle = nullptr;
      }
    }
    if (status == TE9_OK) state_.reset();
    return Status::Engine(status);
  }
  te9_engine_t native_handle() const noexcept {
    if (state_ == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(state_->operation);
    return state_->handle;
  }
  explicit operator bool() const noexcept {
    return state_ != nullptr && native_handle() != nullptr;
  }

 private:
  friend class Pipeline;
  std::shared_ptr<detail::EngineState> state_;
};

enum class Role : int32_t {
  System = TE9_LLM_CHAT_SYSTEM,
  Developer = TE9_LLM_CHAT_DEVELOPER,
  User = TE9_LLM_CHAT_USER,
  Assistant = TE9_LLM_CHAT_ASSISTANT,
  Tool = TE9_LLM_CHAT_TOOL
};

enum class CacheControl : int32_t {
  None = TE9_LLM_CACHE_NONE,
  Ephemeral = TE9_LLM_CACHE_EPHEMERAL,
  Ephemeral5m = TE9_LLM_CACHE_EPHEMERAL_5M,
  Ephemeral1h = TE9_LLM_CACHE_EPHEMERAL_1H
};

struct ContentPart {
  enum class Kind : int32_t {
    Text = TE9_LLM_CHAT_PART_TEXT,
    ToolCall = TE9_LLM_CHAT_PART_TOOL_CALL,
    ToolResult = TE9_LLM_CHAT_PART_TOOL_RESULT,
    ToolErrorResult = TE9_LLM_CHAT_PART_TOOL_ERROR_RESULT,
    Image = TE9_LLM_CHAT_PART_IMAGE,
    Audio = TE9_LLM_CHAT_PART_AUDIO,
    Reasoning = TE9_LLM_CHAT_PART_REASONING
  };

  Kind kind = Kind::Text;
  std::string text;
  std::string tool_call_id;
  std::string tool_name;
  std::string arguments_json;
  CacheControl cache_control = CacheControl::None;

  static ContentPart Text(std::string value,
                          CacheControl cache = CacheControl::None) {
    ContentPart part;
    part.text = std::move(value);
    part.cache_control = cache;
    return part;
  }
  static ContentPart ToolCall(std::string id, std::string name,
                              std::string arguments,
                              CacheControl cache = CacheControl::None) {
    ContentPart part;
    part.kind = Kind::ToolCall;
    part.tool_call_id = std::move(id);
    part.tool_name = std::move(name);
    part.arguments_json = std::move(arguments);
    part.cache_control = cache;
    return part;
  }
  static ContentPart ToolResult(std::string id, std::string value,
                                std::string name = {}, bool is_error = false,
                                CacheControl cache = CacheControl::None) {
    ContentPart part;
    part.kind = is_error ? Kind::ToolErrorResult : Kind::ToolResult;
    part.tool_call_id = std::move(id);
    part.tool_name = std::move(name);
    part.text = std::move(value);
    part.cache_control = cache;
    return part;
  }
  static ContentPart Image(CacheControl cache = CacheControl::None) {
    ContentPart part;
    part.kind = Kind::Image;
    part.cache_control = cache;
    return part;
  }
  static ContentPart Audio(CacheControl cache = CacheControl::None) {
    ContentPart part;
    part.kind = Kind::Audio;
    part.cache_control = cache;
    return part;
  }
  static ContentPart Reasoning(std::string value,
                               CacheControl cache = CacheControl::None) {
    ContentPart part;
    part.kind = Kind::Reasoning;
    part.text = std::move(value);
    part.cache_control = cache;
    return part;
  }
};

struct Message {
  Role role = Role::User;
  std::string name;
  std::vector<ContentPart> content;

  static Message Text(Role role, std::string text, std::string name = {}) {
    Message message;
    message.role = role;
    message.name = std::move(name);
    message.content.push_back(ContentPart::Text(std::move(text)));
    return message;
  }
};

struct Tool {
  std::string name;
  std::string description;
  std::string input_schema_json;
  CacheControl cache_control = CacheControl::None;
};

struct ToolChoice {
  enum class Kind : int32_t {
    Auto = TE9_LLM_TOOL_AUTO,
    None = TE9_LLM_TOOL_NONE,
    Required = TE9_LLM_TOOL_REQUIRED,
    Named = TE9_LLM_TOOL_NAMED
  };
  Kind kind = Kind::Auto;
  std::string name;

  static ToolChoice Named(std::string value) {
    ToolChoice choice;
    choice.kind = Kind::Named;
    choice.name = std::move(value);
    return choice;
  }
};

enum class ReasoningEffort : int32_t {
  Default = TE9_LLM_REASONING_DEFAULT,
  None = TE9_LLM_REASONING_NONE,
  Minimal = TE9_LLM_REASONING_MINIMAL,
  Low = TE9_LLM_REASONING_LOW,
  Medium = TE9_LLM_REASONING_MEDIUM,
  High = TE9_LLM_REASONING_HIGH,
  XHigh = TE9_LLM_REASONING_XHIGH,
  Max = TE9_LLM_REASONING_MAX
};

enum class ThinkingMode : int32_t {
  Default = TE9_LLM_THINKING_DEFAULT,
  Disabled = TE9_LLM_THINKING_DISABLED,
  Enabled = TE9_LLM_THINKING_ENABLED,
  Adaptive = TE9_LLM_THINKING_ADAPTIVE
};

struct ThinkingConfig {
  ThinkingMode mode = ThinkingMode::Default;
  int64_t budget_tokens = 0;

  static ThinkingConfig Disabled() { return {ThinkingMode::Disabled, 0}; }
  static ThinkingConfig Enabled(int64_t budget) {
    return {ThinkingMode::Enabled, budget};
  }
  static ThinkingConfig Adaptive() { return {ThinkingMode::Adaptive, 0}; }
};

struct ChatRequest {
  std::vector<Message> messages;
  std::vector<Tool> tools;
  ToolChoice tool_choice;
  bool add_generation_prompt = true;
  ReasoningEffort reasoning_effort = ReasoningEffort::Default;
  ThinkingConfig thinking;
  std::optional<bool> preserve_thinking;
};

struct ResponseFormat {
  enum class Kind : int32_t {
    Text = TE9_LLM_RESPONSE_TEXT,
    JsonObject = TE9_LLM_RESPONSE_JSON_OBJECT,
    JsonSchema = TE9_LLM_RESPONSE_JSON_SCHEMA
  };
  Kind kind = Kind::Text;
  std::string schema_name;
  std::string schema_json;
  bool strict = false;

  static ResponseFormat JsonObject() {
    ResponseFormat format;
    format.kind = Kind::JsonObject;
    return format;
  }
  static ResponseFormat JsonSchema(std::string name, std::string schema,
                                   bool strict = true) {
    ResponseFormat format;
    format.kind = Kind::JsonSchema;
    format.schema_name = std::move(name);
    format.schema_json = std::move(schema);
    format.strict = strict;
    return format;
  }
};

struct GenerationOptions {
  int64_t max_tokens = 256;
  float temperature = 0.8f;
  float top_p = 0.95f;
  int32_t top_k = 50;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
  bool do_sample = true;
  int32_t speculation_k = 0;
};

struct PromptCachePoint {
  size_t token_offset = 0;
  int32_t ttl_seconds = 300;
};

enum class MediaDtype : int32_t {
  Float16 = TE9_LLM_MEDIA_FLOAT16,
  Float32 = TE9_LLM_MEDIA_FLOAT32,
};

enum class MediaKind : int32_t {
  Image = TE9_LLM_MEDIA_IMAGE,
  Audio = TE9_LLM_MEDIA_AUDIO,
};

/* Owned host-produced media embedding. The buffers need no special allocator;
 * the LLM and Engine layers consume/copy them before Generate returns. */
struct MediaEmbedding {
  std::vector<uint8_t> data;
  int64_t token_count = 0;
  int64_t hidden = 0;
  MediaDtype dtype = MediaDtype::Float16;
  uint64_t content_hash = 0;
  int64_t media_token_id = 0;
  int32_t grid_t = 0;
  int32_t grid_h = 0;
  int32_t grid_w = 0;
  int32_t spatial_merge_size = 0;
  bool no_mrope = false;
  std::vector<uint8_t> deepstack_data;
  int32_t num_deepstack_layers = 0;
  MediaKind kind = MediaKind::Image;
};

struct RequestOptions {
  GenerationOptions generation;
  std::vector<std::string> stop_strings;
  std::vector<std::vector<int64_t>> stop_sequences;
  std::vector<size_t> cache_points;
  std::vector<PromptCachePoint> prompt_cache_points;
  std::optional<ResponseFormat> response_format;
  std::vector<MediaEmbedding> media;
  /* Low-level compatibility path: input token IDs must already be expanded
   * and positions must already cover the final prompt. */
  const te9_image_embeds* embeds = nullptr;
};

class Request {
 public:
  Request() = default;
  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
  Request(Request&& other) noexcept
      : handle_(other.handle_.exchange(0, std::memory_order_acq_rel)) {}
  Request& operator=(Request&& other) noexcept {
    if (this != &other) {
      Request old(handle_.exchange(0, std::memory_order_acq_rel));
      handle_.store(other.handle_.exchange(0, std::memory_order_acq_rel),
                    std::memory_order_release);
    }
    return *this;
  }
  ~Request() noexcept { CloseNoexcept(); }

  StatusOr<std::string> Read(int32_t timeout_ms, size_t capacity = 4096) {
    std::string output(capacity, '\0');
    size_t written = 0;
    const te9_llm_request_t handle = handle_.load(std::memory_order_acquire);
    const te9_status status = te9_llm_request_read(
        handle, timeout_ms, output.empty() ? nullptr : output.data(),
        output.size(), &written);
    if (status != TE9_OK) return Status::Llm(status);
    output.resize(written);
    return output;
  }
  StatusOr<te9_gen_status> GenerationStatus() const {
    te9_gen_status value = TE9_GEN_RUNNING;
    const te9_status status =
        te9_llm_request_status(handle_.load(std::memory_order_acquire), &value);
    if (status != TE9_OK) return Status::Llm(status);
    return value;
  }
  StatusOr<te9_finish_reason> FinishReason() const {
    te9_finish_reason value = TE9_FINISH_NONE;
    const te9_status status = te9_llm_request_finish_reason(
        handle_.load(std::memory_order_acquire), &value);
    if (status != TE9_OK) return Status::Llm(status);
    return value;
  }
  StatusOr<te9_request_stats> Stats() const {
    te9_request_stats value{};
    value.struct_size = sizeof(value);
    const te9_status status = te9_llm_request_get_stats(
        handle_.load(std::memory_order_acquire), &value);
    if (status != TE9_OK) return Status::Llm(status);
    return value;
  }
  StatusOr<size_t> MatchedStop() const {
    size_t value = SIZE_MAX;
    const te9_status status = te9_llm_request_matched_stop(
        handle_.load(std::memory_order_acquire), &value);
    if (status != TE9_OK) return Status::Llm(status);
    return value;
  }
  Status Stop() {
    return Status::Llm(
        te9_llm_request_stop(handle_.load(std::memory_order_acquire)));
  }
  Status Close() {
    const te9_llm_request_t handle =
        handle_.exchange(0, std::memory_order_acq_rel);
    if (handle == 0) return {};
    const te9_status status = te9_llm_request_release(handle);
    return Status::Llm(status);
  }
  te9_llm_request_t native_handle() const noexcept {
    return handle_.load(std::memory_order_acquire);
  }
  explicit operator bool() const noexcept { return native_handle() != 0; }

 private:
  friend class Pipeline;
  explicit Request(te9_llm_request_t handle) : handle_(handle) {}
  void CloseNoexcept() noexcept {
    const te9_llm_request_t handle =
        handle_.exchange(0, std::memory_order_acq_rel);
    if (handle != 0) (void)te9_llm_request_release(handle);
  }
  std::atomic<te9_llm_request_t> handle_{0};
};

class Pipeline {
 public:
  Pipeline() = default;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&& other) noexcept
      : handle_(other.handle_.exchange(0, std::memory_order_acq_rel)) {}
  Pipeline& operator=(Pipeline&& other) noexcept {
    if (this != &other) {
      Pipeline old(handle_.exchange(0, std::memory_order_acq_rel));
      handle_.store(other.handle_.exchange(0, std::memory_order_acq_rel),
                    std::memory_order_release);
    }
    return *this;
  }
  ~Pipeline() noexcept { CloseNoexcept(); }

  static StatusOr<Pipeline> Create(const Engine& engine,
                                   std::string_view model_name,
                                   std::string_view template_override = {}) {
    if (engine.state_ == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    std::lock_guard<std::mutex> lock(engine.state_->operation);
    if (engine.state_->handle == nullptr)
      return Status(TE9_ERR_INVALID_ARG, "engine is closed");
    te9_llm_pipeline_config config{};
    config.struct_size = sizeof(config);
    config.engine = engine.state_->handle;
    config.model_name = View(model_name);
    config.chat_template_override = View(template_override);
    te9_llm_pipeline_t handle = 0;
    const te9_status status = te9_llm_pipeline_create(&config, &handle);
    if (status != TE9_OK) return Status::Llm(status);
    return Pipeline(handle);
  }

  StatusOr<std::vector<int64_t>> Encode(std::string_view text) const {
    const te9_llm_pipeline_t handle = handle_.load(std::memory_order_acquire);
    size_t needed = 0;
    te9_status status =
        te9_llm_pipeline_encode(handle, View(text), nullptr, 0, &needed);
    if (status != TE9_ERR_BUFFER_TOO_SMALL && status != TE9_OK)
      return Status::Llm(status);
    std::vector<int64_t> tokens(needed);
    status = te9_llm_pipeline_encode(handle, View(text), tokens.data(),
                                     tokens.size(), &needed);
    if (status != TE9_OK) return Status::Llm(status);
    tokens.resize(needed);
    return tokens;
  }

  StatusOr<std::string> Render(const ChatRequest& chat) const {
    const Status support = CheckChatSupport(chat);
    if (!support) return support;
    ChatC storage(chat);
    const te9_llm_pipeline_t handle = handle_.load(std::memory_order_acquire);
    size_t needed = 0;
    te9_status status = te9_llm_pipeline_render_chat(handle, &storage.request,
                                                     nullptr, 0, &needed);
    if (status != TE9_ERR_BUFFER_TOO_SMALL && status != TE9_OK)
      return Status::Llm(status);
    std::string output(needed, '\0');
    status = te9_llm_pipeline_render_chat(
        handle, &storage.request, output.empty() ? nullptr : output.data(),
        output.size(), &needed);
    if (status != TE9_OK) return Status::Llm(status);
    output.resize(needed);
    return output;
  }

  StatusOr<uint64_t> ExportPrefixCache(std::string_view path) const {
    uint64_t node_count = 0;
    const te9_status status = te9_llm_pipeline_export_prefix_cache(
        handle_.load(std::memory_order_acquire), View(path), &node_count);
    if (status != TE9_OK) return Status::Llm(status);
    return node_count;
  }

  StatusOr<uint64_t> ImportPrefixCache(std::string_view path) const {
    uint64_t node_count = 0;
    const te9_status status = te9_llm_pipeline_import_prefix_cache(
        handle_.load(std::memory_order_acquire), View(path), &node_count);
    if (status != TE9_OK) return Status::Llm(status);
    return node_count;
  }

  StatusOr<Request> Generate(std::string_view prompt,
                             const RequestOptions& options = {}) const {
    te9_llm_request_config config{};
    config.struct_size = sizeof(config);
    config.input_kind = TE9_LLM_INPUT_UTF8;
    config.prompt_utf8 = View(prompt);
    return Start(config, options);
  }
  StatusOr<Request> Generate(const std::vector<int64_t>& token_ids,
                             const RequestOptions& options = {}) const {
    te9_llm_request_config config{};
    config.struct_size = sizeof(config);
    config.input_kind = TE9_LLM_INPUT_TOKEN_IDS;
    config.prompt_ids = token_ids.data();
    config.prompt_id_count = token_ids.size();
    return Start(config, options);
  }
  StatusOr<Request> Generate(const ChatRequest& chat,
                             const RequestOptions& options = {}) const {
    const Status support = CheckChatSupport(chat);
    if (!support) return support;
    ChatC storage(chat);
    te9_llm_request_config config{};
    config.struct_size = sizeof(config);
    config.input_kind = TE9_LLM_INPUT_CHAT;
    config.chat = &storage.request;
    return Start(config, options);
  }

  Status Close() {
    const te9_llm_pipeline_t handle =
        handle_.exchange(0, std::memory_order_acq_rel);
    if (handle == 0) return {};
    const te9_status status = te9_llm_pipeline_destroy(handle);
    return Status::Llm(status);
  }
  te9_llm_pipeline_t native_handle() const noexcept {
    return handle_.load(std::memory_order_acquire);
  }
  explicit operator bool() const noexcept { return native_handle() != 0; }

 private:
  static Status CheckChatSupport(const ChatRequest& chat) {
    bool needs_tool_result_error = false;
    bool needs_cache_control = false;
    bool needs_image = false;
    bool needs_audio = false;
    bool needs_historical_reasoning = false;
    for (const Message& message : chat.messages) {
      for (const ContentPart& part : message.content) {
        needs_tool_result_error |=
            part.kind == ContentPart::Kind::ToolErrorResult;
        needs_cache_control |= part.cache_control != CacheControl::None;
        needs_image |= part.kind == ContentPart::Kind::Image;
        needs_audio |= part.kind == ContentPart::Kind::Audio;
        needs_historical_reasoning |= part.kind == ContentPart::Kind::Reasoning;
      }
    }
    for (const Tool& tool : chat.tools)
      needs_cache_control |= tool.cache_control != CacheControl::None;
    if (chat.reasoning_effort == ReasoningEffort::Default &&
        chat.thinking.mode == ThinkingMode::Default &&
        !chat.preserve_thinking.has_value() &&
        !needs_tool_result_error && !needs_cache_control && !needs_image &&
        !needs_audio && !needs_historical_reasoning)
      return {};
    te9_llm_api_table table{};
    const te9_status status = te9_llm_get_api_table(sizeof(table), &table);
    if (status != TE9_OK) return Status::Llm(status);
    if (chat.reasoning_effort != ReasoningEffort::Default &&
        (table.capabilities & TE9_LLM_CAP_REASONING_CONTROL) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support reasoning controls");
    }
    if (chat.thinking.mode != ThinkingMode::Default &&
        (table.capabilities & TE9_LLM_CAP_THINKING_CONTROL) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support thinking controls");
    }
    if (chat.preserve_thinking.has_value() &&
        (table.capabilities & TE9_LLM_CAP_PRESERVE_THINKING) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support preserve_thinking");
    }
    if (needs_historical_reasoning &&
        (table.capabilities & TE9_LLM_CAP_PRESERVE_THINKING) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support historical reasoning");
    }
    if (needs_tool_result_error &&
        (table.capabilities & TE9_LLM_CAP_TOOL_RESULT_ERROR) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support tool-result errors");
    }
    if (needs_cache_control &&
        (table.capabilities & TE9_LLM_CAP_CHAT_CACHE_CONTROL) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support chat cache controls");
    }
    if (needs_image &&
        (table.capabilities & TE9_LLM_CAP_PRECOMPUTED_MEDIA) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support image chat parts");
    }
    if (needs_audio &&
        (table.capabilities & TE9_LLM_CAP_PRECOMPUTED_AUDIO) == 0) {
      return Status(
          TE9_ERR_UNSUPPORTED,
          "loaded Tempo9LLM runtime does not support audio chat parts");
    }
    return {};
  }

  struct ChatC {
    explicit ChatC(const ChatRequest& input) {
      parts.resize(input.messages.size());
      messages.resize(input.messages.size());
      for (size_t i = 0; i != input.messages.size(); ++i) {
        const Message& source = input.messages[i];
        parts[i].resize(source.content.size());
        for (size_t j = 0; j != source.content.size(); ++j) {
          const ContentPart& content = source.content[j];
          te9_llm_chat_content_part& target = parts[i][j];
          target.struct_size = sizeof(target);
          target.kind = static_cast<int32_t>(content.kind);
          target.text = View(content.text);
          target.tool_call_id = View(content.tool_call_id);
          target.tool_name = View(content.tool_name);
          target.arguments_json = View(content.arguments_json);
          target.cache_control = static_cast<int32_t>(content.cache_control);
        }
        te9_llm_chat_message& target = messages[i];
        target.struct_size = sizeof(target);
        target.role = static_cast<int32_t>(source.role);
        target.name = View(source.name);
        target.content = parts[i].data();
        target.content_count = parts[i].size();
      }
      tools.resize(input.tools.size());
      for (size_t i = 0; i != input.tools.size(); ++i) {
        const Tool& source = input.tools[i];
        te9_llm_chat_tool& target = tools[i];
        target.struct_size = sizeof(target);
        target.name = View(source.name);
        target.description = View(source.description);
        target.input_schema_json = View(source.input_schema_json);
        target.cache_control = static_cast<int32_t>(source.cache_control);
      }
      request.struct_size = sizeof(request);
      request.messages = messages.data();
      request.message_count = messages.size();
      request.tools = tools.data();
      request.tool_count = tools.size();
      request.tool_choice = static_cast<int32_t>(input.tool_choice.kind);
      request.named_tool = View(input.tool_choice.name);
      request.add_generation_prompt = input.add_generation_prompt ? 1 : 0;
      if (input.reasoning_effort != ReasoningEffort::Default) {
        reasoning.struct_size = sizeof(reasoning);
        reasoning.effort = static_cast<int32_t>(input.reasoning_effort);
        request.reasoning = &reasoning;
      }
      if (input.thinking.mode != ThinkingMode::Default) {
        thinking.struct_size = sizeof(thinking);
        thinking.mode = static_cast<int32_t>(input.thinking.mode);
        thinking.budget_tokens = input.thinking.budget_tokens;
        request.thinking = &thinking;
      }
      if (input.preserve_thinking.has_value()) {
        request.preserve_thinking =
            *input.preserve_thinking ? TE9_LLM_PRESERVE_THINKING_ENABLED
                                     : TE9_LLM_PRESERVE_THINKING_DISABLED;
      }
    }

    std::vector<std::vector<te9_llm_chat_content_part>> parts;
    std::vector<te9_llm_chat_message> messages;
    std::vector<te9_llm_chat_tool> tools;
    te9_llm_reasoning_config reasoning{};
    te9_llm_thinking_config thinking{};
    te9_llm_chat_request request{};
  };

  static te9_llm_string_view View(std::string_view value) {
    return {value.data(), value.size()};
  }

  StatusOr<Request> Start(te9_llm_request_config config,
                          const RequestOptions& options) const {
    if (!options.cache_points.empty() && !options.prompt_cache_points.empty())
      return Status(TE9_ERR_INVALID_ARG,
                    "cache_points and prompt_cache_points cannot be mixed");
    if (!options.media.empty() && options.embeds != nullptr)
      return Status(TE9_ERR_INVALID_ARG,
                    "media and raw embeds cannot be mixed");
    bool needs_audio = false;
    if (!options.media.empty()) {
      te9_llm_api_table table{};
      const te9_status table_status =
          te9_llm_get_api_table(sizeof(table), &table);
      if (table_status != TE9_OK) return Status::Llm(table_status);
      if ((table.capabilities & TE9_LLM_CAP_PRECOMPUTED_MEDIA) == 0)
        return Status(
            TE9_ERR_UNSUPPORTED,
            "loaded Tempo9LLM runtime does not support precomputed media");
      for (const MediaEmbedding& value : options.media) {
        if (value.kind != MediaKind::Image && value.kind != MediaKind::Audio)
          return Status(TE9_ERR_INVALID_ARG, "unknown media kind");
        needs_audio |= value.kind == MediaKind::Audio;
      }
      if (needs_audio &&
          (table.capabilities & TE9_LLM_CAP_PRECOMPUTED_AUDIO) == 0)
        return Status(
            TE9_ERR_UNSUPPORTED,
            "loaded Tempo9LLM runtime does not support precomputed audio");
    }
    te9_generate_config generation{};
    generation.struct_size = sizeof(generation);
    generation.max_tokens = options.generation.max_tokens;
    generation.temperature = options.generation.temperature;
    generation.top_p = options.generation.top_p;
    generation.top_k = options.generation.top_k;
    generation.repetition_penalty = options.generation.repetition_penalty;
    generation.seed = options.generation.seed;
    generation.do_sample = options.generation.do_sample ? 1 : 0;
    generation.speculation_k = options.generation.speculation_k;
    config.generation = &generation;

    std::vector<te9_llm_string_view> stop_strings;
    stop_strings.reserve(options.stop_strings.size());
    for (const std::string& value : options.stop_strings)
      stop_strings.push_back(View(value));
    config.stop_strings = stop_strings.data();
    config.stop_string_count = stop_strings.size();
    std::vector<te9_llm_token_sequence> stop_sequences;
    stop_sequences.reserve(options.stop_sequences.size());
    for (const std::vector<int64_t>& value : options.stop_sequences) {
      te9_llm_token_sequence sequence{};
      sequence.struct_size = sizeof(sequence);
      sequence.ids = value.data();
      sequence.count = value.size();
      stop_sequences.push_back(sequence);
    }
    config.stop_sequences = stop_sequences.data();
    config.stop_sequence_count = stop_sequences.size();
    std::vector<te9_prompt_cache_point> cache_points;
    std::vector<int32_t> cache_point_ttls;
    cache_points.reserve(options.cache_points.empty()
                             ? options.prompt_cache_points.size()
                             : options.cache_points.size());
    for (size_t value : options.cache_points) {
      te9_prompt_cache_point point{};
      point.struct_size = sizeof(point);
      point.token_offset = value;
      cache_points.push_back(point);
    }
    for (const PromptCachePoint& value : options.prompt_cache_points) {
      te9_prompt_cache_point point{};
      point.struct_size = sizeof(point);
      point.token_offset = value.token_offset;
      cache_points.push_back(point);
      cache_point_ttls.push_back(value.ttl_seconds);
    }
    config.cache_points = cache_points.data();
    config.cache_point_count = cache_points.size();
    config.cache_point_ttl_seconds =
        cache_point_ttls.empty() ? nullptr : cache_point_ttls.data();
    te9_llm_response_format response{};
    if (options.response_format) {
      response.struct_size = sizeof(response);
      response.kind = static_cast<int32_t>(options.response_format->kind);
      response.schema_name = View(options.response_format->schema_name);
      response.schema_json = View(options.response_format->schema_json);
      response.strict = options.response_format->strict ? 1 : 0;
      config.response_format = &response;
    }
    config.embeds = options.embeds;
    std::vector<te9_llm_media_embedding> media;
    std::vector<int32_t> media_kinds;
    media.reserve(options.media.size());
    if (needs_audio) media_kinds.reserve(options.media.size());
    for (const MediaEmbedding& source : options.media) {
      te9_llm_media_embedding target{};
      target.struct_size = sizeof(target);
      target.data = source.data.empty() ? nullptr : source.data.data();
      target.byte_count = source.data.size();
      target.token_count = source.token_count;
      target.hidden = source.hidden;
      target.dtype = static_cast<int32_t>(source.dtype);
      target.content_hash = source.content_hash;
      target.media_token_id = source.media_token_id;
      target.grid_t = source.grid_t;
      target.grid_h = source.grid_h;
      target.grid_w = source.grid_w;
      target.spatial_merge_size = source.spatial_merge_size;
      target.no_mrope = source.no_mrope ? 1 : 0;
      target.deepstack_data = source.deepstack_data.empty()
                                  ? nullptr
                                  : source.deepstack_data.data();
      target.deepstack_byte_count = source.deepstack_data.size();
      target.num_deepstack_layers = source.num_deepstack_layers;
      media.push_back(target);
      if (needs_audio) media_kinds.push_back(static_cast<int32_t>(source.kind));
    }
    config.media = media.empty() ? nullptr : media.data();
    config.media_count = media.size();
    config.media_kinds = media_kinds.empty() ? nullptr : media_kinds.data();
    te9_llm_request_t request = 0;
    const te9_status status = te9_llm_request_start(
        handle_.load(std::memory_order_acquire), &config, &request);
    if (status != TE9_OK) return Status::Llm(status);
    return Request(request);
  }

  explicit Pipeline(te9_llm_pipeline_t handle) : handle_(handle) {}
  void CloseNoexcept() noexcept {
    const te9_llm_pipeline_t handle =
        handle_.exchange(0, std::memory_order_acq_rel);
    if (handle != 0) (void)te9_llm_pipeline_destroy(handle);
  }
  std::atomic<te9_llm_pipeline_t> handle_{0};
};

}  // namespace tempo9
