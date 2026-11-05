/* Copyright (c) 2026 Jiejing Zhang. */

/* Load a GGUF, send one chat turn, and stream the answer. */

#include <stdio.h>
#include <string.h>

#include <tempo9/llm.h>

static te9_llm_string_view view(const char* text) {
  te9_llm_string_view result = {text, strlen(text)};
  return result;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s MODEL.gguf QUESTION\n", argv[0]);
    return 2;
  }
  const char* const model_name = "local";
  te9_engine_t engine = NULL;
  te9_llm_pipeline_t pipeline = 0;
  te9_llm_request_t request = 0;
  int result = 1;

  if (te9_engine_create(&engine) != TE9_OK) goto done;

  te9_model_config model = {0};
  model.struct_size = sizeof(model);
  model.model_name = model_name;
  model.weights_path = argv[1];
  model.compute_unit = "CPU:0"; /* the engine moves GEMM to Metal by itself */
  if (te9_engine_build_model(engine, &model) != TE9_OK ||
      te9_engine_start_model(engine, model_name) != TE9_OK)
    goto done;
  /* A CPU fallback answers correctly, only much slower. */
  fprintf(stderr, "backend: %s %s\n", te9_gemm_backend(),
          te9_gemm_backend_reason());

  te9_llm_pipeline_config config = {0};
  config.struct_size = sizeof(config);
  config.engine = engine;
  config.model_name = view(model_name);
  if (te9_llm_pipeline_create(&config, &pipeline) != TE9_OK) goto done;

  te9_llm_chat_content_part part = {0};
  part.struct_size = sizeof(part);
  part.kind = TE9_LLM_CHAT_PART_TEXT;
  part.text = view(argv[2]);

  te9_llm_chat_message message = {0};
  message.struct_size = sizeof(message);
  message.role = TE9_LLM_CHAT_USER;
  message.content = &part;
  message.content_count = 1;

  te9_llm_chat_request chat = {0};
  chat.struct_size = sizeof(chat);
  chat.messages = &message;
  chat.message_count = 1;
  chat.add_generation_prompt = 1;

  te9_generate_config generation = {0};
  generation.struct_size = sizeof(generation);
  generation.max_tokens = 512;
  generation.temperature = 0.7f;
  generation.top_p = 0.9f;
  generation.do_sample = 1;

  te9_llm_request_config start = {0};
  start.struct_size = sizeof(start);
  start.input_kind = TE9_LLM_INPUT_CHAT;
  start.chat = &chat;
  start.generation = &generation;
  if (te9_llm_request_start(pipeline, &start, &request) != TE9_OK) goto done;

  for (;;) {
    char chunk[1024];
    size_t written = 0;
    te9_gen_status status = TE9_GEN_RUNNING;
    if (te9_llm_request_read(request, 100, chunk, sizeof(chunk), &written) !=
        TE9_OK)
      goto done;
    fwrite(chunk, 1, written, stdout);
    fflush(stdout);
    /* Zero bytes means the read timed out, not that the answer ended. */
    if (written == 0) {
      if (te9_llm_request_status(request, &status) != TE9_OK) goto done;
      if (status != TE9_GEN_RUNNING) break;
    }
  }
  putchar('\n');

  te9_finish_reason reason = TE9_FINISH_NONE;
  if (te9_llm_request_finish_reason(request, &reason) != TE9_OK) goto done;
  fprintf(stderr, "finish reason: %d\n", (int)reason);
  result = 0;

done:
  if (result != 0) {
    const char* const llm_error = te9_llm_last_error();
    fprintf(stderr, "%s\n", llm_error[0] != '\0' ? llm_error : te9_last_error());
  }
  if (request != 0) (void)te9_llm_request_release(request);
  if (pipeline != 0) (void)te9_llm_pipeline_destroy(pipeline);
  if (engine != NULL) {
    (void)te9_engine_stop_model(engine, model_name);
    (void)te9_engine_release_model(engine, model_name);
    (void)te9_engine_destroy(engine);
  }
  return result;
}
