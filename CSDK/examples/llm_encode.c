/* Copyright (c) 2026 Jiejing Zhang. */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tempo9/llm.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s MODEL.gguf PROMPT\n", argv[0]);
    return 2;
  }
  const char* const model_name = "local";
  te9_engine_t engine = NULL;
  te9_llm_pipeline_t pipeline = 0;
  int result = 1;
  if (te9_engine_create(&engine) != TE9_OK) goto done;

  te9_model_config model = {0};
  model.struct_size = sizeof(model);
  model.model_name = model_name;
  model.weights_path = argv[1];
  model.compute_unit = "CPU:0";
  if (te9_engine_build_model(engine, &model) != TE9_OK ||
      te9_engine_start_model(engine, model_name) != TE9_OK)
    goto done;

  te9_llm_pipeline_config config = {0};
  config.struct_size = sizeof(config);
  config.engine = engine;
  config.model_name.data = model_name;
  config.model_name.size = strlen(model_name);
  if (te9_llm_pipeline_create(&config, &pipeline) != TE9_OK) goto done;

  te9_llm_string_view prompt = {argv[2], strlen(argv[2])};
  size_t count = 0;
  if (te9_llm_pipeline_encode(pipeline, prompt, NULL, 0, &count) !=
      TE9_ERR_BUFFER_TOO_SMALL)
    goto done;
  int64_t* tokens = (int64_t*)malloc(count * sizeof(*tokens));
  if (tokens == NULL) goto done;
  if (te9_llm_pipeline_encode(pipeline, prompt, tokens, count, &count) !=
      TE9_OK) {
    free(tokens);
    goto done;
  }
  for (size_t i = 0; i != count; ++i)
    printf("%s%" PRId64, i == 0 ? "" : " ", tokens[i]);
  putchar('\n');
  free(tokens);
  result = 0;

done:
  if (result != 0)
    fprintf(stderr, "%s%s%s\n", te9_llm_last_error(),
            te9_llm_last_error()[0] != '\0' ? ": " : "", te9_last_error());
  if (pipeline != 0) (void)te9_llm_pipeline_destroy(pipeline);
  if (engine != NULL) {
    (void)te9_engine_stop_model(engine, model_name);
    (void)te9_engine_release_model(engine, model_name);
    (void)te9_engine_destroy(engine);
  }
  return result;
}
