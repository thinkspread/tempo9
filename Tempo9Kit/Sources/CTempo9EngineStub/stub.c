/*
 * Copyright (c) 2026 Jiejing Zhang.
 *
 * A link stub for the engine's C ABI, for TESTS ONLY.
 *
 * Package.swift uses this when no engine build is staged: the test targets
 * link against these definitions instead of the engine archives, so the
 * tests that never call into the engine at run time -- request parsing,
 * refusals, the think/harmony splitters, the Ollama store -- run in any
 * clone, CI's and a contributor's included.
 *
 * Every entry point FAILS. A test that does reach the engine gets
 * TE9_ERR_ENGINE and a message naming this file, not a plausible fake: a
 * stub that answered would turn "this test needs an engine" into a pass.
 * Nothing that ships links this: no product depends on it.
 */
#include "tempo9_engine.h"

static const char kStubError[] =
    "no Tempo9 engine is linked: this is the test stub "
    "(Tempo9Kit/Sources/CTempo9EngineStub/stub.c)";

const char* te9_last_error(void) { return kStubError; }
const char* te9_version(void) { return "stub"; }
const char* te9_gemm_backend(void) { return "none"; }

te9_status te9_engine_create(te9_engine_t* out_engine) {
  if (out_engine) *out_engine = NULL;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_destroy(te9_engine_t engine) {
  (void)engine;
  return TE9_ERR_ENGINE;
}

te9_status te9_engine_build_model(te9_engine_t engine,
                                  const te9_model_config* config) {
  (void)engine; (void)config;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_start_model(te9_engine_t engine, const char* model_name) {
  (void)engine; (void)model_name;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_stop_model(te9_engine_t engine, const char* model_name) {
  (void)engine; (void)model_name;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_release_model(te9_engine_t engine,
                                    const char* model_name) {
  (void)engine; (void)model_name;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_get_stats(te9_engine_t engine, const char* model_name,
                                te9_engine_stats* out) {
  (void)engine; (void)model_name; (void)out;
  return TE9_ERR_ENGINE;
}

te9_status te9_request_start(te9_engine_t engine, const char* model_name,
                             const int64_t* input_ids, size_t input_count,
                             const te9_generate_config* gen_config,
                             const te9_image_embeds* embeds,
                             te9_request_t* out_request) {
  (void)engine; (void)model_name; (void)input_ids; (void)input_count;
  (void)gen_config; (void)embeds;
  if (out_request) *out_request = NULL;
  return TE9_ERR_ENGINE;
}
te9_status te9_request_stop(te9_request_t request) {
  (void)request;
  return TE9_ERR_ENGINE;
}
void te9_request_release(te9_request_t request) { (void)request; }
te9_status te9_request_wait(te9_request_t request, int32_t timeout_ms,
                            int64_t* out_ids, size_t capacity,
                            size_t* out_count) {
  (void)request; (void)timeout_ms; (void)out_ids; (void)capacity;
  if (out_count) *out_count = 0;
  return TE9_ERR_ENGINE;
}
te9_status te9_request_status(te9_request_t request,
                              te9_gen_status* out_status) {
  (void)request; (void)out_status;
  return TE9_ERR_ENGINE;
}
te9_status te9_request_finish_reason(te9_request_t request,
                                     te9_finish_reason* out_reason) {
  (void)request; (void)out_reason;
  return TE9_ERR_ENGINE;
}
te9_status te9_request_get_stats(te9_request_t request,
                                 te9_request_stats* out_stats) {
  (void)request; (void)out_stats;
  return TE9_ERR_ENGINE;
}

te9_status te9_tokenizer_open(const char* gguf_path, te9_tokenizer_t* out_tok) {
  (void)gguf_path;
  if (out_tok) *out_tok = NULL;
  return TE9_ERR_ENGINE;
}
void te9_tokenizer_close(te9_tokenizer_t tok) { (void)tok; }
te9_status te9_tokenizer_encode(te9_tokenizer_t tok, const char* text,
                                int add_special, int32_t* ids, size_t cap,
                                size_t* n_out) {
  (void)tok; (void)text; (void)add_special; (void)ids; (void)cap;
  if (n_out) *n_out = 0;
  return TE9_ERR_ENGINE;
}
te9_status te9_tokenizer_decode(te9_tokenizer_t tok, const int32_t* ids,
                                size_t n_ids, char* buf, size_t cap,
                                size_t* n_out) {
  (void)tok; (void)ids; (void)n_ids; (void)buf; (void)cap;
  if (n_out) *n_out = 0;
  return TE9_ERR_ENGINE;
}
te9_status te9_tokenizer_decode_one(te9_tokenizer_t tok, int32_t id, char* buf,
                                    size_t cap, size_t* n_out) {
  (void)tok; (void)id; (void)buf; (void)cap;
  if (n_out) *n_out = 0;
  return TE9_ERR_ENGINE;
}
int32_t te9_tokenizer_bos(te9_tokenizer_t tok) { (void)tok; return -1; }
int32_t te9_tokenizer_eos(te9_tokenizer_t tok) { (void)tok; return -1; }
size_t te9_tokenizer_vocab_size(te9_tokenizer_t tok) { (void)tok; return 0; }
const char* te9_tokenizer_chat_template(te9_tokenizer_t tok) {
  (void)tok;
  return NULL;
}

/* API 1.4 (capabilities table, cancel tokens, prefix-cache transfer, checked
 * request start). Nothing in the Swift layer calls these yet; they are here so
 * that the first caller links in a clone without an engine, and fails like the
 * rest. */
uint32_t te9_engine_api_version(void) { return 0; }
te9_status te9_engine_get_api_table(size_t caller_size,
                                    te9_engine_api_table* table) {
  (void)caller_size; (void)table;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_cancel_token_create(te9_engine_cancel_token_t* token) {
  if (token) *token = 0;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_cancel_token_cancel(te9_engine_cancel_token_t token) {
  (void)token;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_cancel_token_destroy(te9_engine_cancel_token_t token) {
  (void)token;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_export_prefix_cache(te9_engine_t engine,
                                          const char* model_name,
                                          const char* path,
                                          uint64_t* node_count) {
  (void)engine; (void)model_name; (void)path;
  if (node_count) *node_count = 0;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_import_prefix_cache(te9_engine_t engine,
                                          const char* model_name,
                                          const char* path,
                                          uint64_t* node_count) {
  (void)engine; (void)model_name; (void)path;
  if (node_count) *node_count = 0;
  return TE9_ERR_ENGINE;
}
te9_status te9_engine_get_model_llm_source(te9_engine_t engine,
                                           const char* model_name,
                                           te9_model_llm_source* source,
                                           uint8_t* metadata_blob,
                                           size_t capacity, size_t* needed) {
  (void)engine; (void)model_name; (void)source; (void)metadata_blob;
  (void)capacity;
  if (needed) *needed = 0;
  return TE9_ERR_ENGINE;
}
const char* te9_prefix_snapshot_storage(void) { return "none"; }
uint64_t te9_prefix_snapshot_capture_count(void) { return 0; }
te9_status te9_request_start_checked(te9_engine_t engine,
                                     const te9_checked_request_config* config,
                                     te9_request_t* request) {
  (void)engine; (void)config;
  if (request) *request = NULL;
  return TE9_ERR_ENGINE;
}
