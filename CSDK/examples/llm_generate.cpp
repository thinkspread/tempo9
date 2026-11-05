/* Copyright (c) 2026 Jiejing Zhang. */

#include <iostream>
#include <string>

#include <tempo9/llm.hpp>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf PROMPT\n";
    return 2;
  }
  auto engine_result = tempo9::Engine::Create();
  if (!engine_result) {
    std::cerr << engine_result.status().message() << '\n';
    return 1;
  }
  tempo9::Engine engine = std::move(engine_result).value();
  tempo9::ModelConfig model;
  model.model_name = "local";
  model.weights_path = argv[1];
  const tempo9::Status built = engine.BuildModel(model);
  const tempo9::Status started = built ? engine.StartModel(model.model_name) : built;
  if (!started) {
    std::cerr << started.message() << '\n';
    return 1;
  }
  auto pipeline_result = tempo9::Pipeline::Create(engine, model.model_name);
  if (!pipeline_result) {
    std::cerr << pipeline_result.status().message() << '\n';
    return 1;
  }
  tempo9::Pipeline pipeline = std::move(pipeline_result).value();
  auto request_result = pipeline.Generate(argv[2]);
  if (!request_result) {
    std::cerr << request_result.status().message() << '\n';
    return 1;
  }
  tempo9::Request request = std::move(request_result).value();
  for (;;) {
    auto chunk = request.Read(100);
    if (!chunk) {
      std::cerr << chunk.status().message() << '\n';
      return 1;
    }
    std::cout << chunk.value() << std::flush;
    auto status = request.GenerationStatus();
    if (!status) {
      std::cerr << status.status().message() << '\n';
      return 1;
    }
    if (status.value() != TE9_GEN_RUNNING && chunk.value().empty()) break;
  }
  std::cout << '\n';
  return 0;
}
