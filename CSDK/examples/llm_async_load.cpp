/* Copyright (c) 2026 Jiejing Zhang. */

#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <tempo9/llm.hpp>

namespace {

volatile std::sig_atomic_t stop_requested = 0;

extern "C" void Stop(int) { stop_requested = 1; }

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf\n";
    return 2;
  }

  std::signal(SIGINT, Stop);
  std::signal(SIGTERM, Stop);

  auto created = tempo9::Engine::Create();
  if (!created) {
    std::cerr << created.status().message() << '\n';
    return 1;
  }
  tempo9::Engine engine = std::move(created).value();
  tempo9::ModelConfig model;
  model.model_name = "local";
  model.weights_path = argv[1];

  auto started = engine.LoadModelAsync(model);
  if (!started) {
    std::cerr << started.status().message() << '\n';
    return 1;
  }
  tempo9::ModelLoad load = std::move(started).value();
  while (!load.WaitFor(std::chrono::milliseconds(100))) {
    if (stop_requested != 0) {
      const tempo9::Status status = load.Cancel();
      if (!status) std::cerr << status.message() << '\n';
    }
  }

  const tempo9::Status status = load.Wait();
  if (!status) {
    std::cerr << status.message() << '\n';
    return status.code() == TE9_ERR_CANCELLED ? 130 : 1;
  }
  std::cout << "model ready\n";
  return 0;
}
