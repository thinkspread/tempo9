/* Copyright (c) 2026 Jiejing Zhang. */

#ifndef TEMPO9_LLM_H_
#define TEMPO9_LLM_H_

#include <stddef.h>
#include <stdint.h>
#include <tempo9_engine.h>

#if defined(_WIN32)
#if defined(TEMPO9_LLM_BUILDING_LIBRARY)
#define TE9_LLM_API __declspec(dllexport)
#else
#define TE9_LLM_API __declspec(dllimport)
#endif
#else
#define TE9_LLM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TE9_LLM_API_VERSION TE9_API_VERSION_ENCODE(1u, 10u)
#define TE9_LLM_REQUIRED_ENGINE_API_VERSION TE9_API_VERSION_ENCODE(1u, 1u)

typedef uint64_t te9_llm_capabilities;
enum {
  TE9_LLM_CAP_NONE = 0,
  TE9_LLM_CAP_UTF8_PROMPT = UINT64_C(1) << 0,
  TE9_LLM_CAP_TYPED_CHAT = UINT64_C(1) << 1,
  TE9_LLM_CAP_STREAMING_UTF8 = UINT64_C(1) << 2,
  TE9_LLM_CAP_STOP_SEQUENCES = UINT64_C(1) << 3,
  TE9_LLM_CAP_RESPONSE_FORMAT = UINT64_C(1) << 4,
  TE9_LLM_CAP_PROMPT_CACHE_TTL = UINT64_C(1) << 5,
  /* request_stop interrupts a read blocked in the Engine provider. */
  TE9_LLM_CAP_CONCURRENT_REQUEST_STOP = UINT64_C(1) << 6,
  TE9_LLM_CAP_PREFIX_CACHE_TRANSFER = UINT64_C(1) << 7,
  TE9_LLM_CAP_REASONING_CONTROL = UINT64_C(1) << 8,
  TE9_LLM_CAP_TOOL_RESULT_ERROR = UINT64_C(1) << 9,
  /* Typed chat cache markers are rendered into provider token offsets. */
  TE9_LLM_CAP_CHAT_CACHE_CONTROL = UINT64_C(1) << 10,
  /* Manual/adaptive thinking configuration is available on typed chat. */
  TE9_LLM_CAP_THINKING_CONTROL = UINT64_C(1) << 11,
  /* Typed image markers plus validated precomputed media embeddings. */
  TE9_LLM_CAP_PRECOMPUTED_MEDIA = UINT64_C(1) << 12,
  /* Typed audio markers plus host-produced decoder-ready embeddings. */
  TE9_LLM_CAP_PRECOMPUTED_AUDIO = UINT64_C(1) << 13,
  /* Chat templates can retain or discard historical reasoning blocks. */
  TE9_LLM_CAP_PRESERVE_THINKING = UINT64_C(1) << 14
};

typedef struct {
  size_t struct_size;
  uint32_t api_version;
  uint32_t required_engine_api_version;
  te9_llm_capabilities capabilities;
  te9_engine_capabilities required_engine_capabilities;
  uint32_t active_engine_api_version;
  uint32_t metadata_format_version;
  te9_engine_capabilities active_engine_capabilities;
} te9_llm_api_table;

TE9_LLM_API uint32_t te9_llm_api_version(void);
TE9_LLM_API te9_status te9_llm_get_api_table(size_t caller_size,
                                             te9_llm_api_table* table);
TE9_LLM_API const char* te9_llm_last_error(void);

typedef uint64_t te9_llm_pipeline_t;
typedef uint64_t te9_llm_request_t;

typedef struct {
  const char* data;
  size_t size;
} te9_llm_string_view;

typedef struct {
  size_t struct_size;
  te9_engine_t engine;
  te9_llm_string_view model_name;
  te9_llm_string_view chat_template_override;
} te9_llm_pipeline_config;

/* A pipeline snapshots the model's tokenizer/template metadata and generation
 * identity. The engine and named model must outlive the pipeline and every
 * request created from it. Recreate the pipeline after rebuilding a model. */
TE9_LLM_API te9_status te9_llm_pipeline_create(
    const te9_llm_pipeline_config* config, te9_llm_pipeline_t* pipeline);
TE9_LLM_API te9_status te9_llm_pipeline_destroy(te9_llm_pipeline_t pipeline);
/* Persist or restore this pipeline's prefix cache. The model must be stopped
 * before import. Snapshot bytes are opaque and valid only for the same model
 * and a compatible cache layout. `node_count` is required. */
TE9_LLM_API te9_status te9_llm_pipeline_export_prefix_cache(
    te9_llm_pipeline_t pipeline, te9_llm_string_view path,
    uint64_t* node_count);
TE9_LLM_API te9_status te9_llm_pipeline_import_prefix_cache(
    te9_llm_pipeline_t pipeline, te9_llm_string_view path,
    uint64_t* node_count);

typedef int32_t te9_llm_cache_control;
enum {
  TE9_LLM_CACHE_NONE = 0,
  /* The historical ephemeral value is the five-minute default. */
  TE9_LLM_CACHE_EPHEMERAL = 1,
  TE9_LLM_CACHE_EPHEMERAL_5M = TE9_LLM_CACHE_EPHEMERAL,
  TE9_LLM_CACHE_EPHEMERAL_1H = 2
};

typedef int32_t te9_llm_chat_role;
enum {
  TE9_LLM_CHAT_SYSTEM = 0,
  TE9_LLM_CHAT_DEVELOPER = 1,
  TE9_LLM_CHAT_USER = 2,
  TE9_LLM_CHAT_ASSISTANT = 3,
  TE9_LLM_CHAT_TOOL = 4
};

typedef int32_t te9_llm_chat_part_kind;
enum {
  TE9_LLM_CHAT_PART_TEXT = 0,
  TE9_LLM_CHAT_PART_TOOL_CALL = 1,
  TE9_LLM_CHAT_PART_TOOL_RESULT = 2,
  /* A tool result whose execution failed. The text carries the error. */
  TE9_LLM_CHAT_PART_TOOL_ERROR_RESULT = 3,
  /* An ordered image marker. Embedding data is supplied on the request. */
  TE9_LLM_CHAT_PART_IMAGE = 4,
  /* An ordered audio marker. Embedding data is supplied on the request. */
  TE9_LLM_CHAT_PART_AUDIO = 5,
  /* Historical assistant reasoning consumed by model chat templates. */
  TE9_LLM_CHAT_PART_REASONING = 6
};

typedef struct {
  size_t struct_size;
  int32_t kind;
  te9_llm_string_view text;
  te9_llm_string_view tool_call_id;
  te9_llm_string_view tool_name;
  te9_llm_string_view arguments_json;
  int32_t cache_control;
} te9_llm_chat_content_part;

typedef struct {
  size_t struct_size;
  int32_t role;
  te9_llm_string_view name;
  const te9_llm_chat_content_part* content;
  size_t content_count;
} te9_llm_chat_message;

typedef struct {
  size_t struct_size;
  te9_llm_string_view name;
  te9_llm_string_view description;
  te9_llm_string_view input_schema_json;
  int32_t cache_control;
} te9_llm_chat_tool;

typedef int32_t te9_llm_tool_choice;
enum {
  TE9_LLM_TOOL_AUTO = 0,
  TE9_LLM_TOOL_NONE = 1,
  TE9_LLM_TOOL_REQUIRED = 2,
  TE9_LLM_TOOL_NAMED = 3
};

typedef int32_t te9_llm_reasoning_effort;
enum {
  TE9_LLM_REASONING_DEFAULT = 0,
  TE9_LLM_REASONING_NONE = 1,
  TE9_LLM_REASONING_MINIMAL = 2,
  TE9_LLM_REASONING_LOW = 3,
  TE9_LLM_REASONING_MEDIUM = 4,
  TE9_LLM_REASONING_HIGH = 5,
  TE9_LLM_REASONING_XHIGH = 6,
  TE9_LLM_REASONING_MAX = 7
};

typedef struct {
  size_t struct_size;
  int32_t effort;
} te9_llm_reasoning_config;

typedef int32_t te9_llm_thinking_mode;
enum {
  TE9_LLM_THINKING_DEFAULT = 0,
  TE9_LLM_THINKING_DISABLED = 1,
  TE9_LLM_THINKING_ENABLED = 2,
  TE9_LLM_THINKING_ADAPTIVE = 3
};

typedef struct {
  size_t struct_size;
  int32_t mode;
  int64_t budget_tokens;
} te9_llm_thinking_config;

typedef int32_t te9_llm_preserve_thinking;
enum {
  TE9_LLM_PRESERVE_THINKING_DEFAULT = 0,
  TE9_LLM_PRESERVE_THINKING_DISABLED = 1,
  TE9_LLM_PRESERVE_THINKING_ENABLED = 2
};

typedef struct {
  size_t struct_size;
  const te9_llm_chat_message* messages;
  size_t message_count;
  const te9_llm_chat_tool* tools;
  size_t tool_count;
  int32_t tool_choice;
  te9_llm_string_view named_tool;
  int32_t add_generation_prompt;
  /* Optional since ABI 1.3. Null preserves the model template's default. */
  const te9_llm_reasoning_config* reasoning;
  /* Optional since ABI 1.6. Manual mode requires at least 1024 tokens. */
  const te9_llm_thinking_config* thinking;
  /* Optional since ABI 1.10. Default leaves the template's policy unchanged. */
  int32_t preserve_thinking;
} te9_llm_chat_request;

TE9_LLM_API te9_status te9_llm_pipeline_render_chat(
    te9_llm_pipeline_t pipeline, const te9_llm_chat_request* request,
    char* utf8, size_t capacity, size_t* needed);
/* Both buffer APIs always report the required element count. A null output
 * with zero capacity is a size query; a short buffer returns
 * TE9_ERR_BUFFER_TOO_SMALL without a partial result. */
TE9_LLM_API te9_status te9_llm_pipeline_encode(te9_llm_pipeline_t pipeline,
                                               te9_llm_string_view utf8,
                                               int64_t* token_ids,
                                               size_t capacity, size_t* needed);

typedef int32_t te9_llm_input_kind;
enum {
  TE9_LLM_INPUT_UTF8 = 0,
  TE9_LLM_INPUT_TOKEN_IDS = 1,
  TE9_LLM_INPUT_CHAT = 2
};

typedef struct {
  size_t struct_size;
  const int64_t* ids;
  size_t count;
} te9_llm_token_sequence;

typedef int32_t te9_llm_response_format_kind;
enum {
  TE9_LLM_RESPONSE_TEXT = 0,
  TE9_LLM_RESPONSE_JSON_OBJECT = 1,
  TE9_LLM_RESPONSE_JSON_SCHEMA = 2
};

typedef struct {
  size_t struct_size;
  int32_t kind;
  te9_llm_string_view schema_name;
  te9_llm_string_view schema_json;
  int32_t strict; /* JSON_SCHEMA requires 1; other kinds require 0 */
} te9_llm_response_format;

typedef int32_t te9_llm_media_dtype;
enum { TE9_LLM_MEDIA_FLOAT16 = 0, TE9_LLM_MEDIA_FLOAT32 = 1 };

typedef int32_t te9_llm_media_kind;
enum { TE9_LLM_MEDIA_IMAGE = 0, TE9_LLM_MEDIA_AUDIO = 1 };

/* A host-produced media embedding. The LLM layer expands exactly one rendered
 * media_token_id placeholder to token_count copies and builds the full
 * request M-RoPE table for image media from grid_t/grid_h/grid_w and
 * spatial_merge_size. Set no_mrope=1 for images whose media tokens use
 * ordinary positions. `te9_llm_request_config.media_kinds` identifies each
 * descriptor; a missing kind array means image. Audio requires no_mrope=1,
 * zero grid fields, and no DeepStack payload.
 *
 * Every pointed-to byte is consumed synchronously by te9_llm_request_start.
 * The provider copies it before that function returns. */
typedef struct {
  size_t struct_size;
  const void* data;
  size_t byte_count;
  int64_t token_count;
  int64_t hidden;
  int32_t dtype;
  uint64_t content_hash;
  int64_t media_token_id;
  int32_t grid_t;
  int32_t grid_h;
  int32_t grid_w;
  int32_t spatial_merge_size;
  int32_t no_mrope;
  const void* deepstack_data;
  size_t deepstack_byte_count;
  int32_t num_deepstack_layers;
} te9_llm_media_embedding;

typedef struct {
  size_t struct_size;
  int32_t input_kind;
  te9_llm_string_view prompt_utf8;
  const int64_t* prompt_ids;
  size_t prompt_id_count;
  const te9_llm_chat_request* chat;
  const te9_generate_config* generation;
  const te9_llm_string_view* stop_strings;
  size_t stop_string_count;
  const te9_llm_token_sequence* stop_sequences;
  size_t stop_sequence_count;
  const te9_prompt_cache_point* cache_points;
  size_t cache_point_count;
  const te9_llm_response_format* response_format;
  const te9_image_embeds* embeds;
  const int32_t* cache_point_ttl_seconds; /* parallel to cache_points */
  /* Optional since ABI 1.7. Mutually exclusive with raw `embeds`. */
  const te9_llm_media_embedding* media;
  size_t media_count;
  /* Optional since ABI 1.8. Parallel to media; NULL means every entry is
   * TE9_LLM_MEDIA_IMAGE. The array is consumed synchronously. */
  const int32_t* media_kinds;
} te9_llm_request_config;

TE9_LLM_API te9_status te9_llm_request_start(
    te9_llm_pipeline_t pipeline, const te9_llm_request_config* config,
    te9_llm_request_t* request);
/* One thread may read a request. Other threads may query, stop, or release it.
 * A successful read writes only complete UTF-8 code points. Zero bytes is a
 * normal timeout/poll result; inspect request_status to detect completion.
 * Capacity zero performs no provider poll. Stop interrupts a provider wait
 * when TE9_LLM_CAP_CONCURRENT_REQUEST_STOP is advertised; otherwise it safely
 * waits for that provider call to return. */
TE9_LLM_API te9_status te9_llm_request_read(te9_llm_request_t request,
                                            int32_t timeout_ms, char* utf8,
                                            size_t capacity, size_t* written);
TE9_LLM_API te9_status te9_llm_request_status(te9_llm_request_t request,
                                              te9_gen_status* status);
TE9_LLM_API te9_status te9_llm_request_finish_reason(te9_llm_request_t request,
                                                     te9_finish_reason* reason);
TE9_LLM_API te9_status te9_llm_request_get_stats(te9_llm_request_t request,
                                                 te9_request_stats* stats);
TE9_LLM_API te9_status te9_llm_request_matched_stop(te9_llm_request_t request,
                                                    size_t* stop_index);
TE9_LLM_API te9_status te9_llm_request_stop(te9_llm_request_t request);
/* Release invalidates the integer handle, waits for active wrapper calls, and
 * releases the provider request exactly once. */
TE9_LLM_API te9_status te9_llm_request_release(te9_llm_request_t request);

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef TE9_LLM_API

#endif /* TEMPO9_LLM_H_ */
