/*!
 * Copyright (c) 2026 Jiejing Zhang.
 * @file    tempo9_engine.h
 *
 * A C ABI over the engine, so a Swift app can embed it without a Python
 * process and without C++ interop.
 *
 * `allspark.h` is C++: std::shared_ptr, std::string, std::map, virtual bases,
 * DLPack. Swift's C++ interop does not swallow that -- `shared_ptr<Request
 * Content>` and the abstract `ResultQueue` in particular. This header is the
 * narrow surface an app actually needs, expressed in plain C.
 *
 * ── Design rules, so the next person extending this does not break an app ──
 *
 * 1. Opaque handles only. No struct in this header exposes engine internals,
 *    so the app does not have to be rebuilt when they change.
 *
 * 2. Every struct that crosses the boundary starts with `size_t struct_size`.
 *    Callers set it to `sizeof(the struct they compiled against)`; the
 *    implementation refuses anything it does not recognise and reads only the
 *    fields that fit. Adding a trailing field is then source- and
 *    binary-compatible, which matters when the engine and the app ship on
 *    different schedules (a shipped app on the App Store cannot be recompiled
 *    to match a new engine).
 *
 * 3. No ownership crosses the boundary implicitly. Anything the caller gets
 *    back that needs freeing has an explicit `as_*_free`. Buffers the caller
 *    passes in are copied by the implementation before it returns -- the app
 *    may free them immediately. This is deliberate: the DLPack capsule
 *    lifetime rule on the Python side ("must outlive StartRequest") has
 *    already cost one debugging session, whose symptom was a silent
 *    `embedding_len=1` rather than a crash. A C ABI aimed at App Store
 *    binaries must not carry that kind of rule.
 *
 * 4. Errors are returned, never thrown. The C++ side throws (AsException,
 *    std::bad_alloc); every entry point here catches everything and maps it
 *    to a code. An exception crossing into Swift is undefined behaviour.
 *
 * Threading: an `te9_engine` is safe to use from multiple threads. Exactly one
 * thread owns destruction; callers must serialize te9_engine_destroy for a
 * handle and must not start new calls after destruction begins. A single
 * `te9_request` handle is not generally thread-safe -- serialize calls for one
 * request. A provider advertising TE9_ENGINE_CAP_CONCURRENT_REQUEST_STOP makes
 * one exception: te9_request_stop may run concurrently with request wait,
 * status, finish-reason, or stats calls so it can interrupt a blocked wait.
 * Request release must still wait until every other call has returned.
 */

#ifndef TEMPO9_ENGINE_H_
#define TEMPO9_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Status ─────────────────────────────────────────────────────────────── */

typedef enum {
  TE9_OK = 0,
  TE9_ERR_UNKNOWN = 1,
  TE9_ERR_INVALID_ARG = 2, /* null handle, bad struct_size, bad shape   */
  TE9_ERR_NOT_FOUND = 3,   /* model / file missing                      */
  TE9_ERR_OUT_OF_MEMORY = 4,
  TE9_ERR_ENGINE = 5,      /* engine returned a non-success AsStatus    */
  TE9_ERR_TIMEOUT = 6,     /* wait deadline elapsed, request still live */
  TE9_ERR_EXCEPTION = 7,   /* C++ threw; message via te9_last_error()    */
  TE9_ERR_UNSUPPORTED = 8, /* provider cannot supply the requested form */
  TE9_ERR_NOT_READY = 9,   /* model identity changed or is not startable */
  TE9_ERR_BUFFER_TOO_SMALL = 10, /* retry with the reported required size */
  TE9_ERR_CANCELLED = 11         /* cooperative operation cancellation    */
} te9_status;

#ifdef __cplusplus
static_assert(sizeof(te9_status) == sizeof(int32_t),
              "Tempo9 Engine ABI requires 32-bit te9_status");
#else
_Static_assert(sizeof(te9_status) == sizeof(int32_t),
               "Tempo9 Engine ABI requires 32-bit te9_status");
#endif

/* ── ABI discovery ────────────────────────────────────────────────────── */

/* From 3.0 the encoded ABI version follows the engine release: an engine
 * 3.0.0-rc3 reports ABI 3.0 (the pre-release suffix has no slot here; read it
 * from te9_version()).  Up to 1.5 the ABI was numbered on its own, and 3.0 is
 * that same ABI renumbered -- binary compatible with 1.1 through 1.5.
 *
 * Compatibility is therefore NOT "same major".  It is what it always was in
 * practice: struct_size on every versioned struct (a caller passes the size it
 * was compiled with; the provider fills that prefix) and the capability bits in
 * te9_engine_api_table.  Check a capability before calling the entry point it
 * guards.  A minor release may only append fields, enum values, capability
 * bits, and optional table entries. */
#define TE9_API_VERSION_ENCODE(major, minor) \
  ((((uint32_t)(major)) << 16) | ((uint32_t)(minor) & 0xffffu))
#define TE9_API_VERSION_MAJOR(version) (((uint32_t)(version)) >> 16)
#define TE9_API_VERSION_MINOR(version) (((uint32_t)(version)) & 0xffffu)
#define TE9_ENGINE_API_VERSION TE9_API_VERSION_ENCODE(3u, 0u)
/* 1.x and 3.x are one ABI family: 3.0 is 1.5 renumbered.  RANK puts both on
 * one scale (1.0 -> 0 ... 1.5 -> 5, 3.0 -> 5, 3.1 -> 6, ...) so "is this
 * provider at least what I was built for" stays a single comparison; a major
 * outside the family (2, 4, ...) is not comparable at all. */
#define TE9_API_VERSION_IN_FAMILY(version)      \
  (TE9_API_VERSION_MAJOR(version) == 1u ||       \
   TE9_API_VERSION_MAJOR(version) == 3u)
#define TE9_API_VERSION_RANK(version)                          \
  (TE9_API_VERSION_MAJOR(version) == 3u                         \
       ? 5u + TE9_API_VERSION_MINOR(version)                    \
       : TE9_API_VERSION_MINOR(version))

typedef uint64_t te9_engine_capabilities;
enum {
  TE9_ENGINE_CAP_NONE = 0,
  TE9_ENGINE_CAP_MODEL_LLM_SOURCE = UINT64_C(1) << 0,
  TE9_ENGINE_CAP_CHECKED_REQUEST_START = UINT64_C(1) << 1,
  TE9_ENGINE_CAP_PROMPT_CACHE_TTL = UINT64_C(1) << 2,
  /* te9_request_stop may run concurrently with another operation on the same
   * request. Without this bit, callers must serialize stop as well. */
  TE9_ENGINE_CAP_CONCURRENT_REQUEST_STOP = UINT64_C(1) << 3,
  /* The provider can persist and restore the named model's prefix cache. */
  TE9_ENGINE_CAP_PREFIX_CACHE_TRANSFER = UINT64_C(1) << 4,
  /* Model build observes a caller-owned cancellation token. */
  TE9_ENGINE_CAP_CANCELABLE_MODEL_LOAD = UINT64_C(1) << 5
};

typedef struct te9_engine_s* te9_engine_t;
typedef struct te9_request_s* te9_request_t;
typedef uint64_t te9_engine_cancel_token_t;
struct te9_model_llm_source_s;
struct te9_checked_request_config_s;

typedef te9_status (*te9_engine_get_model_llm_source_fn)(
    te9_engine_t engine, const char* model_name,
    struct te9_model_llm_source_s* source, uint8_t* metadata_blob,
    size_t capacity, size_t* needed);
typedef te9_status (*te9_request_start_checked_fn)(
    te9_engine_t engine, const struct te9_checked_request_config_s* config,
    te9_request_t* request);
typedef te9_status (*te9_engine_prefix_cache_transfer_fn)(
    te9_engine_t engine, const char* model_name, const char* path,
    uint64_t* node_count);
typedef te9_status (*te9_engine_cancel_token_create_fn)(
    te9_engine_cancel_token_t* token);
typedef te9_status (*te9_engine_cancel_token_cancel_fn)(
    te9_engine_cancel_token_t token);
typedef te9_status (*te9_engine_cancel_token_destroy_fn)(
    te9_engine_cancel_token_t token);

/**
 * Provider-owned ABI discovery prefix.
 *
 * Callers pass the byte size of the table they compiled against. The provider
 * copies only the prefix it knows and reports its complete size in
 * `struct_size`; bytes beyond that provider size are left untouched. Callers
 * must reject a different major and may require capability bits before using
 * functions introduced after v1.0.
 */
typedef struct {
  size_t struct_size;
  uint32_t api_version;
  uint32_t reserved;
  te9_engine_capabilities capabilities;
  /* Optional v1.1 entries. Read only when struct_size reaches the field and
   * the corresponding capability is set. */
  te9_engine_get_model_llm_source_fn get_model_llm_source;
  te9_request_start_checked_fn request_start_checked;
  /* Optional v1.3 entries. A snapshot is opaque and may only be imported for
   * the same model and a compatible cache layout. */
  te9_engine_prefix_cache_transfer_fn export_prefix_cache;
  te9_engine_prefix_cache_transfer_fn import_prefix_cache;
  /* Optional v1.4 entries. A running build retains an internal state lease
   * after the public token handle is destroyed. */
  te9_engine_cancel_token_create_fn cancel_token_create;
  te9_engine_cancel_token_cancel_fn cancel_token_cancel;
  te9_engine_cancel_token_destroy_fn cancel_token_destroy;
} te9_engine_api_table;

uint32_t te9_engine_api_version(void);
te9_status te9_engine_get_api_table(size_t caller_size,
                                    te9_engine_api_table* table);

/**
 * Human-readable detail for the calling thread's most recent failure.
 *
 * Thread-local and valid until the next failing call on the same thread, so
 * copy it if you keep it. Long diagnostics may be truncated. Returns "" when
 * nothing has failed. The status code is the contract; this string is for logs
 * and bug reports, never for control flow.
 */
const char* te9_last_error(void);

/* ── Engine ─────────────────────────────────────────────────────────────── */

te9_status te9_engine_create(te9_engine_t* out_engine);
/**
 * Destroy an engine and consume its handle.
 *
 * NULL is a successful no-op. The caller must serialize destruction of each
 * handle. On failure the handle remains live and the owning thread may retry;
 * only TE9_OK permits the caller to discard the handle.
 */
te9_status te9_engine_destroy(te9_engine_t engine);

/**
 * How a model is loaded. Mirrors the fields of AsModelConfig that an on-device
 * app actually sets; the rest keep engine defaults.
 *
 * The GGUF path is the one that matters on Apple: Qwen3.5's GatedDeltaNet
 * conv1d.weight cannot be serialized to .asparam, so HF-safetensors is not an
 * option for this model family at all. `graph_path` is the prebuilt
 * .asgraph, OR NULL when weights_path is a .gguf: the engine then builds the
 * graph itself (C++ builder, leaf-identical to the Python one on all 16
 * support-matrix models) and caches it under ~/.cache/tempo9 keyed by the
 * gguf's size+mtime.  One file is enough.
 *
 * One engine owns one live model. Build returns TE9_ERR_NOT_READY until the
 * prior model is released; create another engine for concurrent models.
 */
typedef struct {
  size_t struct_size;
  const char* model_name;   /* required; names the model for later calls   */
  const char* graph_path;   /* .asgraph; NULL => engine builds from .gguf  */
  const char* weights_path; /* required; .gguf                             */
  /* "CPU:0" / "CUDA:0,1" -- the device index is required, not optional:
   * the engine throws on a unit with no colon. NULL => "CPU:0". */
  const char* compute_unit;
  int64_t max_length;          /* 0 => engine default                         */
  int32_t max_batch;           /* 0 => engine default                         */
  int32_t enable_prefix_cache; /* 0/1                                       */
  int32_t speculation_k;       /* 0 = off. MTP draft depth.                   */
  /* Global fallback for prefix nodes without an explicit cache-point TTL.
   * 0 keeps the engine default (300 seconds). Added in Engine ABI v1.2. */
  int32_t prefix_cache_ttl_seconds;
  /* Optional cooperative build/start cancellation. Zero disables it. Added in
   * Engine ABI v1.4. The provider leases the state for the active load. */
  te9_engine_cancel_token_t load_cancel_token;
} te9_model_config;

/* Optional v1.4 functions. Discover them through te9_engine_api_table before
 * calling so one binary can continue to run against an older v1 provider. */
te9_status te9_engine_cancel_token_create(te9_engine_cancel_token_t* token);
te9_status te9_engine_cancel_token_cancel(te9_engine_cancel_token_t token);
te9_status te9_engine_cancel_token_destroy(te9_engine_cancel_token_t token);

te9_status te9_engine_build_model(te9_engine_t engine,
                                  const te9_model_config* config);
te9_status te9_engine_start_model(te9_engine_t engine, const char* model_name);
/* Stops the model and releases any outstanding request handles for it.
 * The caller must still pass each opaque request object to te9_request_release;
 * after model stop, all other operations on those request objects are invalid.
 * Repeating stop for an already-stopped model succeeds. */
te9_status te9_engine_stop_model(te9_engine_t engine, const char* model_name);
te9_status te9_engine_release_model(te9_engine_t engine,
                                    const char* model_name);

/**
 * Persist or restore the named model's prefix cache.
 *
 * `node_count` is required and is set to zero before validation. The provider
 * replaces exports atomically: readers observe either the old file or a
 * complete new file, and failures before replacement leave an existing
 * destination unchanged. Snapshot bytes are opaque; importing into another
 * model or an incompatible cache layout fails. Import is a startup operation
 * and requires the model to be stopped. Providers without a portable snapshot
 * implementation return TE9_ERR_UNSUPPORTED.
 */
te9_status te9_engine_export_prefix_cache(te9_engine_t engine,
                                          const char* model_name,
                                          const char* path,
                                          uint64_t* node_count);
te9_status te9_engine_import_prefix_cache(te9_engine_t engine,
                                          const char* model_name,
                                          const char* path,
                                          uint64_t* node_count);

/* ── LLM model source ─────────────────────────────────────────────────── */

/* Canonical metadata blob v1 is little-endian and has no padding:
 *
 *   magic[4] = "T9TM", u32 format_version, u32 record_count,
 *   u64 total_byte_length,
 *   repeated { u16 field_id, u8 type, u8 flags, u64 element_count,
 *              u64 payload_byte_length, payload[payload_byte_length] }
 *
 * Field ids are strictly increasing. String-array payloads repeat
 * { u64 byte_length, raw_bytes } element_count times. Numeric arrays are
 * tightly packed little-endian scalars. Unknown required fields must fail;
 * unknown optional fields may be skipped by their checked payload length. */
#define TE9_LLM_METADATA_FORMAT_V1 1u
#define TE9_LLM_METADATA_REQUIRED 1u

typedef int32_t te9_llm_metadata_type;
enum {
  TE9_LLM_METADATA_I8 = 1,
  TE9_LLM_METADATA_I32 = 2,
  TE9_LLM_METADATA_UTF8 = 3,
  TE9_LLM_METADATA_I32_ARRAY = 4,
  TE9_LLM_METADATA_F32_ARRAY = 5,
  TE9_LLM_METADATA_UTF8_ARRAY = 6
};

typedef int32_t te9_llm_metadata_field;
enum {
  TE9_LLM_FIELD_TOKENIZER_MODEL = 1,
  TE9_LLM_FIELD_TOKENIZER_PRE = 2,
  TE9_LLM_FIELD_TOKENS = 3,
  TE9_LLM_FIELD_SCORES = 4,
  TE9_LLM_FIELD_TOKEN_TYPES = 5,
  TE9_LLM_FIELD_MERGES = 6,
  TE9_LLM_FIELD_BOS_TOKEN_ID = 7,
  TE9_LLM_FIELD_EOS_TOKEN_ID = 8,
  TE9_LLM_FIELD_UNK_TOKEN_ID = 9,
  TE9_LLM_FIELD_ADD_BOS = 10,
  TE9_LLM_FIELD_ADD_EOS = 11,
  TE9_LLM_FIELD_ADD_SPACE_PREFIX = 12,
  TE9_LLM_FIELD_CHAT_TEMPLATE = 13
};

typedef struct te9_model_llm_source_s {
  size_t struct_size;
  uint64_t engine_generation;
  uint64_t model_generation;
  uint32_t llm_metadata_format_version;
  uint32_t reserved;
  uint8_t llm_metadata_sha256[32];
} te9_model_llm_source;

/* Query size with metadata_blob=NULL/capacity=0, then retry with `needed`
 * bytes. The source identity is filled on both calls. A model without an
 * exactly reconstructable embedded tokenizer returns TE9_OK with format 0,
 * an all-zero digest, and needed=0; its generation identity is still valid
 * for token-id requests. A short non-null buffer returns
 * TE9_ERR_BUFFER_TOO_SMALL without changing engine state. */
te9_status te9_engine_get_model_llm_source(te9_engine_t engine,
                                           const char* model_name,
                                           te9_model_llm_source* source,
                                           uint8_t* metadata_blob,
                                           size_t capacity, size_t* needed);

/* ── Engine stats ───────────────────────────────────────────────────────── */

/* A snapshot of what the engine is doing, for a runtime inspector or a bug
 * report. Everything here is cumulative or instantaneous as marked; nothing
 * is reset by reading it.
 *
 * On a CPU/Metal build these come from the CPU paged-KV pool and
 * CpuPrefixCacheManager rather than span attention. That distinction used to
 * be visible as every field reading zero. */
typedef struct {
  size_t struct_size;
  int64_t total_token; /* KV capacity, in tokens (instantaneous)      */
  int64_t free_token;
  int64_t total_span;
  int64_t free_span;
  int64_t span_size; /* tokens per span                             */
  int32_t running_request;
  int32_t pending_request;
  int64_t total_generated_token; /* cumulative                            */
  int64_t total_prefill_token;   /* cumulative                            */
  int64_t prefix_cache_hit_token;
  int64_t prefix_cache_miss_token;
  float prefix_cache_hit_rate;  /* 0..1 over tokens, not requests        */
  float token_usage_percentage; /* 0..1                                  */
} te9_engine_stats;

/* Fill `out` with the current stats. `out->struct_size` must be set by the
 * caller before the call, the same as every other struct here. */
/**
 * Which GEMM backend is actually in use: "metal" or "cpu".
 *
 * Reported rather than assumed. When the Metal kernels cannot be found the
 * engine falls back to the CPU for every weight GEMM, answers correctly,
 * and runs about four times slower — a state that looked identical from
 * outside to a UI printing "Apple GPU" from a string literal.
 *
 * Never NULL. Valid for the life of the process.
 */
const char* te9_gemm_backend(void);

/**
 * Why `te9_gemm_backend()` says "cpu": one line, the same text the engine
 * logged to stderr when it decided. The cases so far: no Metal device; the
 * device is not in the tensor-API allowlist (M5/M6/A19/A20); the Metal 4
 * tensor-API probe kernel failed to compile — typically the HOST binary is
 * linked against a pre-macOS-26 SDK (its LC_BUILD_VERSION sdk field) and
 * the runtime compiler hides the Metal 4 headers from it; the kernel library
 * failed to compile; AS_METAL_KERNEL_DIR points somewhere unreadable; or
 * AS_GEMM_BACKEND is set to something other than "metal".
 *
 * Empty while the backend is "metal". On a build without Metal support it
 * names that. Hosts should print it next to the backend at startup: the CPU
 * fallback is a 10x cliff that has otherwise looked like nothing happening.
 *
 * Added in API 1.5 (plain symbol, like te9_gemm_backend; not in the api
 * table). Never NULL. Valid for the life of the process.
 */
const char* te9_gemm_backend_reason(void);

/** Actual storage observed across successful prefix checkpoint captures.
 * Returns "none", "fp32", "bf16", "int8", or "mixed". */
const char* te9_prefix_snapshot_storage(void);

/** Number of successful prefix checkpoint captures observed in this process. */
uint64_t te9_prefix_snapshot_capture_count(void);

te9_status te9_engine_get_stats(te9_engine_t engine, const char* model_name,
                                te9_engine_stats* out);

/* ── Generation config ──────────────────────────────────────────────────── */

typedef struct {
  size_t struct_size;
  const int64_t* token_ids;
  size_t token_count;
} te9_token_sequence;

typedef struct {
  size_t struct_size;
  int64_t max_tokens; /* NEW tokens, not total; prompt is added on */
  float temperature;
  float top_p;
  int32_t top_k;
  float repetition_penalty;
  uint64_t seed;
  int32_t do_sample;     /* 0/1                                            */
  int32_t speculation_k; /* per-request override; 0 = use the model's      */
  const int64_t* stop_token_ids;
  size_t stop_token_count;
  /* The end-of-sequence token. NOT optional in practice: GenerateConfig
   * defaults it to 102, which is some other model's EOS, so a caller that
   * leaves this at 0 gets generation that runs past the end of the answer
   * and starts repeating. Appended after stop_token_count, so a binary
   * compiled against the shorter struct still works -- that is what
   * struct_size is for. 0 means "leave the engine default". */
  int64_t eos_token_id;
  /* Guided decoding (xgrammar in the engine). response_format is
   * "json_object" (any valid JSON) or "json_schema" (constrained to
   * response_schema, a JSON Schema document); NULL = free text. Appended
   * after eos_token_id; struct_size gates it like eos_token_id. The engine
   * reads the tokenizer vocabulary from the model's GGUF on first use. */
  const char* response_format;
  const char* response_schema;
  /* Complete token sequences. The legacy stop_token_ids field above keeps
   * its established meaning of one singleton stop per id. */
  const te9_token_sequence* stop_sequences;
  size_t stop_sequence_count;
} te9_generate_config;

/* ── Requests ───────────────────────────────────────────────────────────── */

/**
 * Image embeddings produced by the app's own vision tower.
 *
 * The app runs the tower (Core ML on Apple) and hands the engine the result,
 * so the engine never sees pixels. `data` is row-major [token_count, hidden],
 * and is copied before this call returns.
 *
 * There is no loud failure downstream if this is wrong. A shape mismatch is
 * not rejected by the model -- the embeddings are spliced into the prompt and
 * the model describes an image nobody encoded. So the implementation checks
 * what it can: `token_count` must equal the number of placeholder tokens in
 * `input_ids`, and `byte_count` must equal token_count * hidden *
 * sizeof(dtype).
 */
typedef struct te9_image_embeds_s {
  size_t struct_size;
  const void* data;
  size_t byte_count;
  int64_t token_count;
  int64_t hidden;
  int32_t is_float16; /* 1 = fp16 (what the tower caches), 0 = fp32       */
  /* Replaces the image placeholder tokens in the prefix-cache key, so a
   * second question about the same image reuses its prefill. 0 means the
   * engine derives one, so reuse is never silently lost by omitting it. */
  uint64_t content_hash;

  /* --- the two the engine also requires, appended after content_hash --- */

  /* Which token the chat template emitted as the image placeholder. It is
   * the KEY the engine looks the embedding up under, not just metadata:
   * MultiMediaInfo stores embeddings under str(image_token_id). */
  int64_t image_token_id;
  /* Interleaved M-RoPE position table, int32, [3, sequence_length], row
   * major. VisionTowerKit's llmMRoPEPositions() produces it; the engine does
   * NOT derive it, and without it the image tokens carry text positions and
   * the model attends to the picture as if it were a flat run of words.
   *
   * The prompt handed to te9_request_start must already have the single
   * placeholder expanded to (gridH/2)*(gridW/2) copies -- the template emits
   * one, the expansion is the caller's job (TowerAux.expandImagePlaceholder),
   * and this table is built over the expanded ids. */
  const int32_t* mrope_positions;
  size_t mrope_position_count; /* 3 * sequence_length */

  /* --- appended after mrope_position_count --- */

  /* Set to 1 for a model whose media tokens do NOT use M-RoPE -- Gemma 4,
   * whose image run carries ordinary sequential positions and gets its
   * two-dimensional structure from intra-block bidirectional attention
   * instead (the graph's `bidi_token_ids` attribute).  Such a caller passes
   * mrope_positions = NULL and mrope_position_count = 0.
   *
   * It is a separate flag rather than "NULL means no M-RoPE" on purpose: a
   * Qwen caller that simply FORGOT the table would then be silently served
   * text positions, which is the exact failure this struct's checks exist to
   * prevent -- the model answers fluently about an image it misread.  An
   * older caller, whose struct_size stops short of this field, still gets
   * the strict behaviour. */
  int32_t no_mrope;

  /* --- appended after no_mrope --- */

  /* Another media block in the same request, or NULL.
   *
   * One prompt can carry more than one: "look at this and answer what I am
   * asking" is an image block and an audio block, and Gemma 4 takes both.
   * Each block is stored under its OWN media_token_id -- MultiMediaInfo is
   * keyed by token id, so the engine finds each one where its placeholder
   * run is -- and the M-RoPE table, if any, belongs to the head block and
   * covers the whole sequence.
   *
   * A chain rather than an array so this stays an append to an existing
   * struct: an older caller stops short of the field and carries exactly one
   * block, as it always did. */
  const struct te9_image_embeds_s* next;

  /* --- appended after next --- */

  /* DeepStack features (Qwen3-VL family): the vision tower's intermediate
   * feature maps, added into the FIRST num_deepstack_layers decoder layers'
   * outputs at the image-token positions. Row-major
   * [num_deepstack_layers, token_count, hidden], same dtype as `data`
   * (is_float16 applies to both), or NULL for a model without deepstack.
   *
   * Per BLOCK: each image carries its own features, and the engine
   * reassembles them layer-major across the request's images, which is the
   * order DeepStackInjectOp indexes ("deepstack"[layer * n_images + img]).
   * A block that sets num_deepstack_layers > 0 with NULL data (or a
   * byte_count that disagrees with the shape) is refused loudly -- a
   * silently dropped deepstack produces plausible, wrong answers, the
   * same failure shape as a forgotten M-RoPE table. */
  const void* deepstack_data;
  size_t deepstack_byte_count;
  int32_t num_deepstack_layers;
} te9_image_embeds;

/**
 * Start generating. `input_ids` are already-tokenized ids -- tokenization and
 * chat-template rendering belong to the app (GGUFKit / ChatTemplateKit),
 * which is what lets the engine ship without Python.
 *
 * `embeds` may be NULL for a text-only request.
 */
te9_status te9_request_start(te9_engine_t engine, const char* model_name,
                             const int64_t* input_ids, size_t input_count,
                             const te9_generate_config* gen_config,
                             const te9_image_embeds* embeds /* nullable */,
                             te9_request_t* out_request);

typedef struct te9_prompt_cache_point_s {
  size_t struct_size;
  size_t token_offset; /* exclusive prompt-token offset */
} te9_prompt_cache_point;

typedef struct te9_checked_request_config_s {
  size_t struct_size;
  const char* model_name;
  uint64_t expected_engine_generation;
  uint64_t expected_model_generation;
  const int64_t* input_ids;
  size_t input_id_count;
  const te9_generate_config* generation;
  const te9_prompt_cache_point* cache_points;
  size_t cache_point_count;
  const te9_image_embeds* embeds;
  /* Parallel to cache_points. NULL preserves the model-level fallback.
   * Values are 0 (5-minute default), 300, or 3600. Added in ABI v1.2. */
  const int32_t* cache_point_ttl_seconds;
} te9_checked_request_config;

/* Validate model identity and admit the request under one lifecycle lock.
 * A stale pipeline receives TE9_ERR_NOT_READY and never reaches StartRequest.
 */
te9_status te9_request_start_checked(te9_engine_t engine,
                                     const te9_checked_request_config* config,
                                     te9_request_t* request);

/** Request lifecycle. `release` frees the handle; `stop` only ends generation.
 */
te9_status te9_request_stop(te9_request_t request);
void te9_request_release(te9_request_t request);

/* ── Results ────────────────────────────────────────────────────────────── */

typedef enum {
  TE9_GEN_RUNNING = 0,
  TE9_GEN_FINISHED = 1,
  TE9_GEN_INTERRUPTED = 2
} te9_gen_status;

#ifdef __cplusplus
static_assert(sizeof(te9_gen_status) == sizeof(int32_t),
              "Tempo9 Engine ABI requires 32-bit te9_gen_status");
#else
_Static_assert(sizeof(te9_gen_status) == sizeof(int32_t),
               "Tempo9 Engine ABI requires 32-bit te9_gen_status");
#endif

typedef enum {
  TE9_FINISH_NONE = 0, /* still running                                  */
  TE9_FINISH_EOS = 1,
  TE9_FINISH_LENGTH = 2,
  TE9_FINISH_STOP = 3,
  TE9_FINISH_INTERRUPTED = 4
} te9_finish_reason;

#ifdef __cplusplus
static_assert(sizeof(te9_finish_reason) == sizeof(int32_t),
              "Tempo9 Engine ABI requires 32-bit te9_finish_reason");
#else
_Static_assert(sizeof(te9_finish_reason) == sizeof(int32_t),
               "Tempo9 Engine ABI requires 32-bit te9_finish_reason");
#endif

/**
 * Block for up to `timeout_ms` for new tokens, then copy up to `capacity` of
 * them into `out_ids`.
 *
 * Returns TE9_OK with `*out_count == 0` when the deadline passed with nothing
 * new and the request is still running -- that is a normal tick, not an error,
 * and lets a Swift AsyncStream stay cancellable. TE9_ERR_TIMEOUT is reserved
 * for a caller-visible deadline being exceeded, not for an empty poll.
 *
 * Call `te9_request_status` to learn that generation ended; a final wait may
 * return tokens and a finished status together, so drain before stopping.
 */
te9_status te9_request_wait(te9_request_t request, int32_t timeout_ms,
                            int64_t* out_ids, size_t capacity,
                            size_t* out_count);

te9_status te9_request_status(te9_request_t request,
                              te9_gen_status* out_status);

/**
 * Why generation stopped.
 *
 * This comes from the engine, never from comparing the emitted count against
 * max_tokens. That inference misreports a KV-eviction abort as a truncation,
 * which is a bug both Python servers shipped before it was fixed centrally.
 */
te9_status te9_request_finish_reason(te9_request_t request,
                                     te9_finish_reason* out_reason);

/* ── Introspection ──────────────────────────────────────────────────────── */

typedef struct {
  size_t struct_size;
  int64_t prompt_tokens;
  int64_t generated_tokens;
  int64_t prefix_cache_hit_tokens;
  double prefill_ms;
  double decode_ms;
} te9_request_stats;

te9_status te9_request_get_stats(te9_request_t request,
                                 te9_request_stats* out_stats);

/** Engine build string, for a runtime inspector and for bug reports. */
/* ---------------------------------------------------------------------
 * Tokenizer
 *
 * The tokenizer carried inside a .gguf, opened without loading weights.
 * It lives here rather than in each host because it is part of the
 * text-in/text-out contract: a host that re-derives it can be subtly
 * wrong, and a subtly wrong tokenizer produces fluent nonsense instead
 * of an error. It is also the only tokenizer a GGUF gets on a platform
 * with no HuggingFace repo beside the file.
 *
 * Implemented so far: SentencePiece ("llama") and Gemma 4 raw-UTF8 BPE.
 * Byte-level BPE ("gpt2") still lives in the host; te9_tokenizer_open
 * fails naming the model string rather than guessing.
 * --------------------------------------------------------------------- */
typedef struct te9_tokenizer_s* te9_tokenizer_t;

te9_status te9_tokenizer_open(const char* gguf_path, te9_tokenizer_t* out_tok);
void te9_tokenizer_close(te9_tokenizer_t tok);

/* Encode `text` into ids. Writes at most `cap` ids and always reports the
 * count it needed in *n_out, so one retry with the reported size is
 * enough; a short buffer returns TE9_ERR_INVALID_ARG with *n_out set.
 * `add_special` applies the file's own BOS/EOS policy -- pass 0 when the
 * caller has already rendered a chat template that emits them. */
te9_status te9_tokenizer_encode(te9_tokenizer_t tok, const char* text,
                                int add_special, int32_t* ids, size_t cap,
                                size_t* n_out);

/* Decode ids to UTF-8 bytes. Same cap/needed convention. The result is
 * NOT null-terminated when it exactly fills `cap`; *n_out is the length. */
te9_status te9_tokenizer_decode(te9_tokenizer_t tok, const int32_t* ids,
                                size_t n_ids, char* buf, size_t cap,
                                size_t* n_out);

/* One id at a time, for incremental decoding. A byte-fallback token
 * returns a single raw byte that may be half a UTF-8 sequence, so the
 * caller reassembles. */
te9_status te9_tokenizer_decode_one(te9_tokenizer_t tok, int32_t id, char* buf,
                                    size_t cap, size_t* n_out);

int32_t te9_tokenizer_bos(te9_tokenizer_t tok);
int32_t te9_tokenizer_eos(te9_tokenizer_t tok);
size_t te9_tokenizer_vocab_size(te9_tokenizer_t tok);
/* The Jinja chat template, or "" when the file carries none. Valid until
 * te9_tokenizer_close. */
const char* te9_tokenizer_chat_template(te9_tokenizer_t tok);

const char* te9_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TEMPO9_ENGINE_H_ */
