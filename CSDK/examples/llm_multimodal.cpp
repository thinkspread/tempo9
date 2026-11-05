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
  if (argc != 8) {
    std::cerr << "usage: " << argv[0]
              << " model.gguf image-embeds.f16 tokens hidden media-token-id"
                 " grid-h grid-w\n";
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
  model.model_name = "multimodal";
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
  message.content.push_back(tempo9::ContentPart::Image());
  message.content.push_back(tempo9::ContentPart::Text("Describe this image."));
  chat.messages.push_back(std::move(message));

  tempo9::MediaEmbedding image;
  image.data = std::move(bytes);
  image.dtype = tempo9::MediaDtype::Float16;
  image.token_count = Number(argv[3], "tokens");
  image.hidden = Number(argv[4], "hidden");
  image.media_token_id = Number(argv[5], "media-token-id");
  image.grid_t = 1;
  image.grid_h = static_cast<int32_t>(Number(argv[6], "grid-h"));
  image.grid_w = static_cast<int32_t>(Number(argv[7], "grid-w"));
  image.spatial_merge_size = 2;

  tempo9::RequestOptions options;
  options.generation.max_tokens = 256;
  options.media.push_back(std::move(image));
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
