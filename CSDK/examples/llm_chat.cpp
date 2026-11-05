/* Copyright (c) 2026 Jiejing Zhang. */

#include <iostream>
#include <string>

#include <tempo9/llm.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf\n";
    return 2;
  }
  auto engine_result = tempo9::Engine::Create();
  if (!engine_result) return 1;
  tempo9::Engine engine = std::move(engine_result).value();
  tempo9::ModelConfig model;
  model.model_name = "local";
  model.weights_path = argv[1];
  if (!engine.BuildModel(model) || !engine.StartModel(model.model_name)) {
    std::cerr << te9_last_error() << '\n';
    return 1;
  }
  auto pipeline_result = tempo9::Pipeline::Create(engine, model.model_name);
  if (!pipeline_result) {
    std::cerr << pipeline_result.status().message() << '\n';
    return 1;
  }
  tempo9::Pipeline pipeline = std::move(pipeline_result).value();
  tempo9::ChatRequest chat;
  chat.messages.push_back(
      tempo9::Message::Text(tempo9::Role::System, "Answer concisely."));
  chat.messages.push_back(
      tempo9::Message::Text(tempo9::Role::User, "Why is the sky blue?"));
  chat.reasoning_effort = tempo9::ReasoningEffort::Low;
  chat.tools.push_back({"lookup", "Look up a short fact.",
                        R"({"type":"object","properties":{"topic":{"type":"string"}},"required":["topic"]})"});

  auto rendered = pipeline.Render(chat);
  if (!rendered) {
    std::cerr << rendered.status().message() << '\n';
    return 1;
  }
  std::cout << "Rendered prompt:\n" << rendered.value() << "\n\nResponse:\n";
  auto request_result = pipeline.Generate(chat);
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
    if (!status) return 1;
    if (status.value() != TE9_GEN_RUNNING && chunk.value().empty()) break;
  }
  std::cout << '\n';
  return 0;
}
