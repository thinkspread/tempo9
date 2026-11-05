/* Copyright (c) 2026 Jiejing Zhang. */

#include "metadata.h"

#include <cstring>
#include <limits>

#include <tempo9_engine.h>

namespace allspark::llm::internal {
namespace {

uint16_t ReadU16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(data[1]) << 8;
}

uint32_t ReadU32(const uint8_t* data) {
  uint32_t value = 0;
  for (unsigned i = 0; i != 4; ++i)
    value |= static_cast<uint32_t>(data[i]) << (8u * i);
  return value;
}

uint64_t ReadU64(const uint8_t* data) {
  uint64_t value = 0;
  for (unsigned i = 0; i != 8; ++i)
    value |= static_cast<uint64_t>(data[i]) << (8u * i);
  return value;
}

uint32_t RotateRight(uint32_t value, uint32_t shift) {
  return (value >> shift) | (value << (32u - shift));
}

bool ParseStringArray(const uint8_t* payload, size_t payload_size,
                      uint64_t count, std::vector<std::string>* output,
                      std::string* error) {
  if (count > payload_size / 8u || count > output->max_size()) {
    *error = "T9TM string-array count exceeds payload";
    return false;
  }
  output->clear();
  output->reserve(static_cast<size_t>(count));
  size_t offset = 0;
  for (uint64_t i = 0; i != count; ++i) {
    if (payload_size - offset < 8u) {
      *error = "T9TM string-array length is truncated";
      return false;
    }
    const uint64_t length = ReadU64(payload + offset);
    offset += 8u;
    if (length > payload_size - offset ||
        length > std::numeric_limits<size_t>::max()) {
      *error = "T9TM string-array element exceeds payload";
      return false;
    }
    output->emplace_back(reinterpret_cast<const char*>(payload + offset),
                         static_cast<size_t>(length));
    offset += static_cast<size_t>(length);
  }
  if (offset != payload_size) {
    *error = "T9TM string-array has trailing bytes";
    return false;
  }
  return true;
}

bool Expect(uint8_t actual_type, uint8_t expected_type, uint64_t count,
            uint64_t expected_count, uint64_t payload_size,
            uint64_t expected_size, const char* name, std::string* error) {
  if (actual_type != expected_type || count != expected_count ||
      payload_size != expected_size) {
    *error = std::string("T9TM field ") + name + " has an invalid type or size";
    return false;
  }
  return true;
}

}  // namespace

std::array<uint8_t, 32> Sha256(const uint8_t* data, size_t size) {
  static constexpr uint32_t kRound[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
      0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u,
      0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  auto consume = [&state](const uint8_t* block) {
    uint32_t words[64] = {};
    for (unsigned i = 0; i != 16; ++i) {
      const uint8_t* p = block + 4u * i;
      words[i] = (static_cast<uint32_t>(p[0]) << 24) |
                 (static_cast<uint32_t>(p[1]) << 16) |
                 (static_cast<uint32_t>(p[2]) << 8) | p[3];
    }
    for (unsigned i = 16; i != 64; ++i) {
      const uint32_t s0 = RotateRight(words[i - 15], 7) ^
                          RotateRight(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const uint32_t s1 = RotateRight(words[i - 2], 17) ^
                          RotateRight(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (unsigned i = 0; i != 64; ++i) {
      const uint32_t upper =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t t1 = h + upper + choose + kRound[i] + words[i];
      const uint32_t lower =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = lower + majority;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  };
  size_t offset = 0;
  while (size - offset >= 64u) {
    consume(data + offset);
    offset += 64u;
  }
  std::array<uint8_t, 128> tail{};
  const size_t remainder = size - offset;
  if (remainder != 0) std::memcpy(tail.data(), data + offset, remainder);
  tail[remainder] = 0x80u;
  const size_t tail_size = remainder < 56u ? 64u : 128u;
  const uint64_t bit_size = static_cast<uint64_t>(size) * 8u;
  for (unsigned i = 0; i != 8; ++i)
    tail[tail_size - 1u - i] = static_cast<uint8_t>(bit_size >> (8u * i));
  consume(tail.data());
  if (tail_size == 128u) consume(tail.data() + 64u);
  std::array<uint8_t, 32> digest{};
  for (unsigned i = 0; i != 8; ++i)
    for (unsigned j = 0; j != 4; ++j)
      digest[4u * i + j] = static_cast<uint8_t>(state[i] >> (24u - 8u * j));
  return digest;
}

bool ParseMetadata(const uint8_t* data, size_t size, Vocabulary* vocabulary,
                   std::string* error) {
  if (data == nullptr || vocabulary == nullptr || error == nullptr) return false;
  *vocabulary = Vocabulary{};
  error->clear();
  if (size < 20u || std::memcmp(data, "T9TM", 4) != 0) {
    *error = "T9TM header is missing or truncated";
    return false;
  }
  if (ReadU32(data + 4) != TE9_LLM_METADATA_FORMAT_V1 ||
      ReadU64(data + 12) != size) {
    *error = "T9TM version or declared byte length is invalid";
    return false;
  }
  const uint32_t record_count = ReadU32(data + 8);
  size_t offset = 20;
  uint16_t previous = 0;
  for (uint32_t record = 0; record != record_count; ++record) {
    if (size - offset < 20u) {
      *error = "T9TM record header is truncated";
      return false;
    }
    const uint16_t field = ReadU16(data + offset);
    const uint8_t type = data[offset + 2];
    const uint8_t flags = data[offset + 3];
    const uint64_t count = ReadU64(data + offset + 4);
    const uint64_t payload64 = ReadU64(data + offset + 12);
    offset += 20u;
    if ((flags & ~TE9_LLM_METADATA_REQUIRED) != 0) {
      *error = "T9TM record uses unknown flag bits";
      return false;
    }
    if (field <= previous) {
      *error = "T9TM field ids are duplicated or not increasing";
      return false;
    }
    previous = field;
    if (payload64 > size - offset) {
      *error = "T9TM record payload is truncated";
      return false;
    }
    const size_t payload_size = static_cast<size_t>(payload64);
    const uint8_t* payload = data + offset;
    switch (field) {
      case TE9_LLM_FIELD_TOKENIZER_MODEL:
      case TE9_LLM_FIELD_TOKENIZER_PRE:
      case TE9_LLM_FIELD_CHAT_TEMPLATE: {
        if (type != TE9_LLM_METADATA_UTF8 || count != 1) {
          *error = "T9TM string field has an invalid type or count";
          return false;
        }
        std::string value(reinterpret_cast<const char*>(payload), payload_size);
        if (field == TE9_LLM_FIELD_TOKENIZER_MODEL) vocabulary->model = value;
        if (field == TE9_LLM_FIELD_TOKENIZER_PRE) vocabulary->pre = value;
        if (field == TE9_LLM_FIELD_CHAT_TEMPLATE)
          vocabulary->chat_template = value;
        break;
      }
      case TE9_LLM_FIELD_TOKENS:
      case TE9_LLM_FIELD_MERGES: {
        if (type != TE9_LLM_METADATA_UTF8_ARRAY) {
          *error = "T9TM string array has an invalid type";
          return false;
        }
        std::vector<std::string>* target =
            field == TE9_LLM_FIELD_TOKENS ? &vocabulary->tokens
                                          : &vocabulary->merges;
        if (!ParseStringArray(payload, payload_size, count, target, error))
          return false;
        break;
      }
      case TE9_LLM_FIELD_SCORES: {
        if (type != TE9_LLM_METADATA_F32_ARRAY || count > UINT64_MAX / 4u ||
            count > std::numeric_limits<size_t>::max() ||
            payload64 != count * 4u) {
          *error = "T9TM score array has an invalid type or size";
          return false;
        }
        vocabulary->scores.resize(static_cast<size_t>(count));
        for (size_t i = 0; i != vocabulary->scores.size(); ++i) {
          const uint32_t bits = ReadU32(payload + 4u * i);
          std::memcpy(&vocabulary->scores[i], &bits, sizeof(bits));
        }
        break;
      }
      case TE9_LLM_FIELD_TOKEN_TYPES: {
        if (type != TE9_LLM_METADATA_I32_ARRAY || count > UINT64_MAX / 4u ||
            count > std::numeric_limits<size_t>::max() ||
            payload64 != count * 4u) {
          *error = "T9TM token-type array has an invalid type or size";
          return false;
        }
        vocabulary->types.resize(static_cast<size_t>(count));
        for (size_t i = 0; i != vocabulary->types.size(); ++i)
          vocabulary->types[i] = static_cast<int32_t>(ReadU32(payload + 4u * i));
        break;
      }
      case TE9_LLM_FIELD_BOS_TOKEN_ID:
      case TE9_LLM_FIELD_EOS_TOKEN_ID:
      case TE9_LLM_FIELD_UNK_TOKEN_ID: {
        if (!Expect(type, TE9_LLM_METADATA_I32, count, 1, payload64, 4,
                    "token id", error))
          return false;
        const int32_t value = static_cast<int32_t>(ReadU32(payload));
        if (field == TE9_LLM_FIELD_BOS_TOKEN_ID) vocabulary->bos_id = value;
        if (field == TE9_LLM_FIELD_EOS_TOKEN_ID) vocabulary->eos_id = value;
        if (field == TE9_LLM_FIELD_UNK_TOKEN_ID) vocabulary->unk_id = value;
        break;
      }
      case TE9_LLM_FIELD_ADD_BOS:
      case TE9_LLM_FIELD_ADD_EOS:
      case TE9_LLM_FIELD_ADD_SPACE_PREFIX: {
        if (!Expect(type, TE9_LLM_METADATA_I8, count, 1, payload64, 1,
                    "tokenizer flag", error))
          return false;
        const int8_t value = static_cast<int8_t>(payload[0]);
        if (value < -1 || value > 1) {
          *error = "T9TM tokenizer flag is outside -1..1";
          return false;
        }
        if (field == TE9_LLM_FIELD_ADD_BOS) vocabulary->add_bos = value;
        if (field == TE9_LLM_FIELD_ADD_EOS) vocabulary->add_eos = value;
        if (field == TE9_LLM_FIELD_ADD_SPACE_PREFIX)
          vocabulary->add_space_prefix = value;
        break;
      }
      default:
        if ((flags & TE9_LLM_METADATA_REQUIRED) != 0) {
          *error = "T9TM contains an unknown required field";
          return false;
        }
        break;
    }
    offset += payload_size;
  }
  if (offset != size) {
    *error = "T9TM has bytes after its final record";
    return false;
  }
  if (vocabulary->model.empty() || vocabulary->tokens.empty()) {
    *error = "T9TM is missing tokenizer model or vocabulary";
    return false;
  }
  if (vocabulary->types.size() != vocabulary->tokens.size()) {
    *error = "T9TM token strings and token types have different lengths";
    return false;
  }
  if (vocabulary->model == "llama") {
    if (vocabulary->scores.size() != vocabulary->tokens.size()) {
      *error = "T9TM SentencePiece scores are missing or incomplete";
      return false;
    }
  } else if (vocabulary->model == "gpt2") {
    if ((vocabulary->pre != "qwen2" && vocabulary->pre != "qwen35") ||
        vocabulary->merges.empty()) {
      *error = "T9TM has an unsupported BPE profile or no merge ranks";
      return false;
    }
  } else {
    *error = "T9TM tokenizer model is unsupported: " + vocabulary->model;
    return false;
  }
  return true;
}

}  // namespace allspark::llm::internal
