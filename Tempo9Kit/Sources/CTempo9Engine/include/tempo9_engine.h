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
 * Threading: an `te9_engine` is safe to use from multiple threads. A single
 * `te9_request` handle is not -- drive one request from one thread, which is
 * the natural shape for an AsyncStream per request anyway.
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
  TE9_ERR_INVALID_ARG = 2,   /* null handle, bad struct_size, bad shape   */
  TE9_ERR_NOT_FOUND = 3,     /* model / file missing                      */
  TE9_ERR_OUT_OF_MEMORY = 4,
  TE9_ERR_ENGINE = 5,        /* engine returned a non-success AsStatus    */
  TE9_ERR_TIMEOUT = 6,       /* wait deadline elapsed, request still live */
  TE9_ERR_EXCEPTION = 7      /* C++ threw; message via te9_last_error()    */
} te9_status;

/**
 * Human-readable detail for the calling thread's most recent failure.
 *
 * Thread-local and valid until the next failing call on the same thread, so
 * copy it if you keep it. Returns "" when nothing has failed. The status code
 * is the contract; this string is for logs and bug reports, never for control
 * flow.
 */
const char* te9_last_error(void);

/* ── Engine ─────────────────────────────────────────────────────────────── */

typedef struct te9_engine_s* te9_engine_t;

te9_status te9_engine_create(te9_engine_t* out_engine);
void te9_engine_destroy(te9_engine_t engine);

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
 */
typedef struct {
  size_t struct_size;
  const char* model_name;    /* required; names the model for later calls   */
  const char* graph_path;    /* .asgraph; NULL => engine builds from .gguf  */
  const char* weights_path;  /* required; .gguf                             */
  /* "CPU:0" / "CUDA:0,1" -- the device index is required, not optional:
   * the engine throws on a unit with no colon. NULL => "CPU:0". */
  const char* compute_unit;
  int64_t max_length;        /* 0 => engine default                         */
  int32_t max_batch;         /* 0 => engine default                         */
  int32_t enable_prefix_cache; /* 0/1                                       */
  int32_t speculation_k;     /* 0 = off. MTP draft depth.                   */
} te9_model_config;

te9_status te9_engine_build_model(te9_engine_t engine,
                                  const te9_model_config* config);
te9_status te9_engine_start_model(te9_engine_t engine, const char* model_name);
te9_status te9_engine_stop_model(te9_engine_t engine, const char* model_name);
te9_status te9_engine_release_model(te9_engine_t engine, const char* model_name);

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
  int64_t total_token;      /* KV capacity, in tokens (instantaneous)      */
  int64_t free_token;
  int64_t total_span;
  int64_t free_span;
  int64_t span_size;        /* tokens per span                             */
  int32_t running_request;
  int32_t pending_request;
  int64_t total_generated_token;  /* cumulative                            */
  int64_t total_prefill_token;    /* cumulative                            */
  int64_t prefix_cache_hit_token;
  int64_t prefix_cache_miss_token;
  float prefix_cache_hit_rate;    /* 0..1 over tokens, not requests        */
  float token_usage_percentage;   /* 0..1                                  */
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

te9_status te9_engine_get_stats(te9_engine_t engine, const char* model_name,
                                te9_engine_stats* out);

/* ── Generation config ──────────────────────────────────────────────────── */

typedef struct {
  size_t struct_size;
  int64_t max_tokens;   /* NEW tokens, not total; prompt is added on */
  float temperature;
  float top_p;
  int32_t top_k;
  float repetition_penalty;
  uint64_t seed;
  int32_t do_sample;      /* 0/1                                            */
  int32_t speculation_k;  /* per-request override; 0 = use the model's      */
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
} te9_generate_config;

/* ── Requests ───────────────────────────────────────────────────────────── */

typedef struct te9_request_s* te9_request_t;

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
 * `input_ids`, and `byte_count` must equal token_count * hidden * sizeof(dtype).
 */
typedef struct te9_image_embeds_s {
  size_t struct_size;
  const void* data;
  size_t byte_count;
  int64_t token_count;
  int64_t hidden;
  int32_t is_float16;   /* 1 = fp16 (what the tower caches), 0 = fp32       */
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
  size_t mrope_position_count;   /* 3 * sequence_length */

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

/** Request lifecycle. `release` frees the handle; `stop` only ends generation. */
te9_status te9_request_stop(te9_request_t request);
void te9_request_release(te9_request_t request);

/* ── Results ────────────────────────────────────────────────────────────── */

typedef enum {
  TE9_GEN_RUNNING = 0,
  TE9_GEN_FINISHED = 1,
  TE9_GEN_INTERRUPTED = 2
} te9_gen_status;

typedef enum {
  TE9_FINISH_NONE = 0,   /* still running                                  */
  TE9_FINISH_EOS = 1,
  TE9_FINISH_LENGTH = 2,
  TE9_FINISH_STOP = 3,
  TE9_FINISH_INTERRUPTED = 4
} te9_finish_reason;

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
 * Implemented so far: SentencePiece (tokenizer.ggml.model == "llama").
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
}  /* extern "C" */
#endif

#endif  /* TEMPO9_ENGINE_H_ */
