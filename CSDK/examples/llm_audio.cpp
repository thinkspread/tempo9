/* Copyright (c) 2026 Jiejing Zhang. */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tempo9/llm.hpp>
#include <vector>

namespace {

int64_t Number(const char* value, const char* name) {
  try {
    return std::stoll(value);
  } catch (...) {
    std::cerr << name << " must be an integer\n";
    return 0;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: " << argv[0]
              << " model.gguf audio-embeds.f16 tokens hidden media-token-id\n";
    return 2;
  }
  std::ifstream input(argv[2], std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  if (!input.eof()) {
    std::cerr << "could not read embedding file\n";
    return 2;
  }

  auto engine_result = tempo9::Engine::Create();
  if (!engine_result) {
    std::cerr << engine_result.status().message() << '\n';
    return 1;
  }
  tempo9::Engine engine = std::move(engine_result).value();
  tempo9::ModelConfig model;
  model.model_name = "audio";
  model.weights_path = argv[1];
  tempo9::Status status = engine.BuildModel(model);
  if (!status || !(status = engine.StartModel(model.model_name))) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  auto pipeline_result = tempo9::Pipeline::Create(engine, model.model_name);
  if (!pipeline_result) {
    std::cerr << pipeline_result.status().message() << '\n';
    return 1;
  }
  tempo9::Pipeline pipeline = std::move(pipeline_result).value();

  tempo9::ChatRequest chat;
  tempo9::Message message;
  message.role = tempo9::Role::User;
  message.content.push_back(tempo9::ContentPart::Audio());
  message.content.push_back(
      tempo9::ContentPart::Text("Transcribe or describe this audio."));
  chat.messages.push_back(std::move(message));

  tempo9::MediaEmbedding audio;
  audio.kind = tempo9::MediaKind::Audio;
  audio.data = std::move(bytes);
  audio.dtype = tempo9::MediaDtype::Float16;
  audio.token_count = Number(argv[3], "tokens");
  audio.hidden = Number(argv[4], "hidden");
  audio.media_token_id = Number(argv[5], "media-token-id");
  audio.no_mrope = true;

  tempo9::RequestOptions options;
  options.generation.max_tokens = 256;
  options.media.push_back(std::move(audio));
  auto request_result = pipeline.Generate(chat, options);
  if (!request_result) {
    std::cerr << request_result.status().message() << '\n';
    return 1;
  }
  tempo9::Request request = std::move(request_result).value();
  while (true) {
    auto chunk = request.Read(100, 4096);
    if (!chunk) {
      std::cerr << chunk.status().message() << '\n';
      return 1;
    }
    std::cout << chunk.value() << std::flush;
    auto state = request.GenerationStatus();
    if (!state) {
      std::cerr << state.status().message() << '\n';
      return 1;
    }
    if (state.value() != TE9_GEN_RUNNING && chunk.value().empty()) break;
  }
  std::cout << '\n';
  return request.Close().ok() && pipeline.Close().ok() && engine.Close().ok()
             ? 0
             : 1;
}
