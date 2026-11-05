/* Copyright (c) 2026 Jiejing Zhang. */

// The SentencePiece and byte-level BPE merge algorithms are adapted from
// llama.cpp commit ad6c66839af3c5646fba8c6c2e2087a1e4e38948. llama.cpp is
// MIT licensed; the complete notice is in LICENSES/MIT-llama.cpp.txt.

#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

#include <utf8proc.h>

namespace allspark::llm::internal {
namespace {

constexpr int32_t kTypeControl = 3;
constexpr int32_t kTypeUserDefined = 4;
constexpr int32_t kTypeByte = 6;
constexpr const char* kSpmSpace = "\xe2\x96\x81";

size_t Utf8Length(unsigned char value) {
  if (value < 0x80) return 1;
  if ((value & 0xe0) == 0xc0) return 2;
  if ((value & 0xf0) == 0xe0) return 3;
  if ((value & 0xf8) == 0xf0) return 4;
  return 1;
}

std::string CodepointToUtf8(int32_t codepoint) {
  std::array<utf8proc_uint8_t, 4> bytes{};
  const int length = utf8proc_encode_char(codepoint, bytes.data());
  if (length <= 0) return {};
  return std::string(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<size_t>(length));
}

bool IsLetter(int32_t codepoint) {
  const utf8proc_category_t category = utf8proc_category(codepoint);
  return category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO;
}

bool IsMark(int32_t codepoint) {
  const utf8proc_category_t category = utf8proc_category(codepoint);
  return category >= UTF8PROC_CATEGORY_MN && category <= UTF8PROC_CATEGORY_ME;
}

bool IsNumber(int32_t codepoint) {
  const utf8proc_category_t category = utf8proc_category(codepoint);
  return category >= UTF8PROC_CATEGORY_ND && category <= UTF8PROC_CATEGORY_NO;
}

bool IsWhitespace(int32_t codepoint) {
  if ((codepoint >= 0x09 && codepoint <= 0x0d) || codepoint == 0x85)
    return true;
  const utf8proc_category_t category = utf8proc_category(codepoint);
  return category == UTF8PROC_CATEGORY_ZS || category == UTF8PROC_CATEGORY_ZL ||
         category == UTF8PROC_CATEGORY_ZP;
}

class SpecialTable {
 public:
  explicit SpecialTable(const Vocabulary& vocabulary) {
    for (size_t i = 0; i != vocabulary.tokens.size(); ++i) {
      if (i >= vocabulary.types.size()) break;
      if ((vocabulary.types[i] == kTypeControl ||
           vocabulary.types[i] == kTypeUserDefined) &&
          !vocabulary.tokens[i].empty()) {
        values_.emplace_back(vocabulary.tokens[i], static_cast<int32_t>(i));
      }
    }
    std::stable_sort(values_.begin(), values_.end(),
                     [](const auto& left, const auto& right) {
                       return left.first.size() > right.first.size();
                     });
    for (size_t i = 0; i != values_.size(); ++i) {
      const uint8_t first = static_cast<uint8_t>(values_[i].first.front());
      by_first_[first].push_back(i);
    }
  }

  template <typename Run>
  bool Partition(const std::string& text, Run&& run,
                 std::vector<int64_t>* output, std::string* error) const {
    size_t cursor = 0;
    size_t ordinary = 0;
    while (cursor < text.size()) {
      const auto& candidates =
          by_first_[static_cast<uint8_t>(text[cursor])];
      const std::pair<std::string, int32_t>* match = nullptr;
      for (size_t index : candidates) {
        const auto& value = values_[index];
        if (value.first.size() <= text.size() - cursor &&
            std::memcmp(text.data() + cursor, value.first.data(),
                        value.first.size()) == 0) {
          match = &value;
          break;
        }
      }
      if (match == nullptr) {
        ++cursor;
        continue;
      }
      if (cursor != ordinary &&
          !run(text.substr(ordinary, cursor - ordinary), output, error))
        return false;
      output->push_back(match->second);
      cursor += match->first.size();
      ordinary = cursor;
    }
    return cursor == ordinary || run(text.substr(ordinary), output, error);
  }

 private:
  std::vector<std::pair<std::string, int32_t>> values_;
  std::array<std::vector<size_t>, 256> by_first_;
};

class SpmTokenizer final : public Tokenizer {
 public:
  explicit SpmTokenizer(Vocabulary vocabulary)
      : vocabulary_(std::move(vocabulary)), specials_(vocabulary_) {}

  static std::unique_ptr<SpmTokenizer> Create(Vocabulary vocabulary,
                                               std::string* error) {
    if (vocabulary.tokens.empty() ||
        vocabulary.scores.size() != vocabulary.tokens.size() ||
        vocabulary.types.size() != vocabulary.tokens.size()) {
      *error = "SentencePiece metadata arrays are missing or inconsistent";
      return nullptr;
    }
    auto tokenizer = std::unique_ptr<SpmTokenizer>(
        new SpmTokenizer(std::move(vocabulary)));
    tokenizer->Initialize();
    return tokenizer;
  }

  bool Encode(const std::string& text, bool add_special,
              std::vector<int64_t>* output, std::string* error) const override {
    if (!ValidateUtf8(text.data(), text.size(), error)) return false;
    output->clear();
    const bool add_bos = vocabulary_.add_bos < 0 || vocabulary_.add_bos != 0;
    const bool add_eos = vocabulary_.add_eos > 0;
    if (add_special && add_bos && vocabulary_.bos_id >= 0)
      output->push_back(vocabulary_.bos_id);
    const bool ok = specials_.Partition(
        text,
        [this](const std::string& run, std::vector<int64_t>* ids,
               std::string*) {
          EncodeRun(run, ids);
          return true;
        },
        output, error);
    if (ok && add_special && add_eos && vocabulary_.eos_id >= 0)
      output->push_back(vocabulary_.eos_id);
    return ok;
  }

  bool AppendDecoded(int64_t token_id, std::string* output,
                     std::string* error) const override {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.tokens.size()) {
      *error = "generated token id is outside the tokenizer vocabulary";
      return false;
    }
    if (token_id == vocabulary_.bos_id || token_id == vocabulary_.eos_id)
      return true;
    const std::string& token = vocabulary_.tokens[static_cast<size_t>(token_id)];
    const int32_t type = vocabulary_.types[static_cast<size_t>(token_id)];
    if (type == kTypeControl) return true;
    if (type == kTypeByte && token.size() == 6 && token.compare(0, 3, "<0x") == 0 &&
        token[5] == '>') {
      auto hex = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
      };
      const int high = hex(token[3]);
      const int low = hex(token[4]);
      if (high < 0 || low < 0) {
        *error = "malformed SentencePiece byte token";
        return false;
      }
      output->push_back(static_cast<char>((high << 4) | low));
      return true;
    }
    for (size_t i = 0; i < token.size();) {
      if (token.compare(i, 3, kSpmSpace) == 0) {
        output->push_back(' ');
        i += 3;
      } else {
        output->push_back(token[i++]);
      }
    }
    return true;
  }

  int32_t bos_id() const override { return vocabulary_.bos_id; }
  int32_t eos_id() const override { return vocabulary_.eos_id; }
  size_t max_piece_bytes() const override { return max_piece_bytes_; }
  bool trim_initial_space() const override {
    return vocabulary_.add_space_prefix != 0;
  }
  const std::string& TokenText(int32_t token_id) const override {
    static const std::string empty;
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.tokens.size())
      return empty;
    return vocabulary_.tokens[static_cast<size_t>(token_id)];
  }

 private:
  struct Symbol {
    size_t position = 0;
    size_t size = 0;
    int previous = -1;
    int next = -1;
  };
  struct Bigram {
    int left = -1;
    int right = -1;
    float score = 0;
    size_t size = 0;
    bool operator<(const Bigram& other) const {
      return score < other.score ||
             (score == other.score && left > other.left);
    }
  };

  void Initialize() {
    token_to_id_.reserve(vocabulary_.tokens.size() * 2u);
    for (size_t i = 0; i != vocabulary_.tokens.size(); ++i) {
      token_to_id_.emplace(vocabulary_.tokens[i], static_cast<int32_t>(i));
      max_piece_bytes_ = std::max(max_piece_bytes_, vocabulary_.tokens[i].size());
    }
  }

  int32_t Lookup(const std::string& token) const {
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
  }

  void EmitBytes(const char* bytes, size_t size,
                 std::vector<int64_t>* output) const {
    static constexpr char kHex[] = "0123456789ABCDEF";
    for (size_t i = 0; i != size; ++i) {
      const uint8_t value = static_cast<uint8_t>(bytes[i]);
      const char token[] = {'<', '0', 'x', kHex[value >> 4],
                            kHex[value & 15], '>', '\0'};
      const int32_t id = Lookup(token);
      if (id >= 0)
        output->push_back(id);
      else if (vocabulary_.unk_id >= 0)
        output->push_back(vocabulary_.unk_id);
    }
  }

  void EncodeRun(const std::string& raw, std::vector<int64_t>* output) const {
    if (raw.empty()) return;
    std::string text;
    if (vocabulary_.add_space_prefix != 0) text.push_back(' ');
    text += raw;
    std::string escaped;
    escaped.reserve(text.size() + text.size() / 4u);
    for (char value : text) {
      if (value == ' ')
        escaped += kSpmSpace;
      else
        escaped.push_back(value);
    }

    std::vector<Symbol> symbols;
    symbols.reserve(escaped.size());
    for (size_t offset = 0; offset < escaped.size();) {
      Symbol symbol;
      symbol.position = offset;
      symbol.size = std::min(Utf8Length(static_cast<uint8_t>(escaped[offset])),
                             escaped.size() - offset);
      offset += symbol.size;
      symbol.previous = static_cast<int>(symbols.size()) - 1;
      symbol.next = offset == escaped.size() ? -1
                                             : static_cast<int>(symbols.size()) + 1;
      symbols.push_back(symbol);
    }
    std::priority_queue<Bigram> queue;
    std::unordered_map<std::string, std::pair<int, int>> origins;
    auto add = [&](int left, int right) {
      if (left < 0 || right < 0) return;
      const std::string joined = escaped.substr(
          symbols[static_cast<size_t>(left)].position,
          symbols[static_cast<size_t>(left)].size +
              symbols[static_cast<size_t>(right)].size);
      const int32_t id = Lookup(joined);
      if (id < 0) return;
      queue.push({left, right, vocabulary_.scores[static_cast<size_t>(id)],
                  joined.size()});
      origins[joined] = {left, right};
    };
    for (size_t i = 1; i < symbols.size(); ++i)
      add(static_cast<int>(i - 1), static_cast<int>(i));
    while (!queue.empty()) {
      const Bigram bigram = queue.top();
      queue.pop();
      Symbol& left = symbols[static_cast<size_t>(bigram.left)];
      Symbol& right = symbols[static_cast<size_t>(bigram.right)];
      if (left.size == 0 || right.size == 0 ||
          left.size + right.size != bigram.size)
        continue;
      left.size += right.size;
      right.size = 0;
      left.next = right.next;
      if (right.next >= 0)
        symbols[static_cast<size_t>(right.next)].previous = bigram.left;
      add(left.previous, bigram.left);
      add(bigram.left, left.next);
    }
    std::vector<int> stack;
    for (int i = 0; i >= 0; i = symbols[static_cast<size_t>(i)].next)
      stack.push_back(i);
    std::reverse(stack.begin(), stack.end());
    while (!stack.empty()) {
      const int index = stack.back();
      stack.pop_back();
      const Symbol& symbol = symbols[static_cast<size_t>(index)];
      if (symbol.size == 0) continue;
      const std::string piece = escaped.substr(symbol.position, symbol.size);
      const int32_t id = Lookup(piece);
      if (id >= 0) {
        output->push_back(id);
        continue;
      }
      const auto origin = origins.find(piece);
      if (origin == origins.end()) {
        EmitBytes(escaped.data() + symbol.position, symbol.size, output);
      } else {
        stack.push_back(origin->second.second);
        stack.push_back(origin->second.first);
      }
    }
  }

  Vocabulary vocabulary_;
  SpecialTable specials_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  size_t max_piece_bytes_ = 1;
};

struct Codepoint {
  int32_t value = 0;
  size_t byte_offset = 0;
};

bool DecodeCodepoints(const std::string& text, std::vector<Codepoint>* output,
                      std::string* error) {
  output->clear();
  size_t offset = 0;
  while (offset < text.size()) {
    int32_t codepoint = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
        static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
    if (length <= 0) {
      *error = "invalid UTF-8 input";
      return false;
    }
    output->push_back({codepoint, offset});
    offset += static_cast<size_t>(length);
  }
  output->push_back({-1, text.size()});
  return true;
}

std::vector<std::string> SplitQwen(const std::string& text, bool include_marks,
                                   std::string* error) {
  std::vector<Codepoint> codepoints;
  if (!DecodeCodepoints(text, &codepoints, error)) return {};
  const size_t count = codepoints.size() - 1u;
  std::vector<std::string> output;
  output.reserve(count);
  auto value = [&](size_t index) { return index < count ? codepoints[index].value : -1; };
  auto letter_run = [&](int32_t cp) {
    return IsLetter(cp) || (include_marks && IsMark(cp));
  };
  size_t position = 0;
  while (position < count) {
    const size_t begin = position;
    const int32_t current = value(position);
    if (current == '\'' && position + 1u < count) {
      const int32_t next = value(position + 1u);
      const int32_t lower = next >= 'A' && next <= 'Z' ? next + 32 : next;
      if (lower == 's' || lower == 't' || lower == 'm' || lower == 'd')
        position += 2u;
      else if (position + 2u < count) {
        const int32_t last_value = value(position + 2u);
        const int32_t last = last_value >= 'A' && last_value <= 'Z'
                                 ? last_value + 32
                                 : last_value;
        if ((lower == 'r' && last == 'e') ||
            (lower == 'v' && last == 'e') ||
            (lower == 'l' && last == 'l'))
          position += 3u;
      }
      if (position != begin) {
        output.emplace_back(text.data() + codepoints[begin].byte_offset,
                            codepoints[position].byte_offset -
                                codepoints[begin].byte_offset);
        continue;
      }
    }
    if (current != '\r' && current != '\n' && !IsNumber(current) &&
        (letter_run(current) || letter_run(value(position + 1u)))) {
      ++position;
      while (position < count && letter_run(value(position))) ++position;
    } else if (IsNumber(current)) {
      ++position;
    } else {
      size_t scan = position + (current == ' ' ? 1u : 0u);
      auto punctuation = [&](int32_t cp) {
        return cp >= 0 && !IsWhitespace(cp) && !IsLetter(cp) &&
               !IsNumber(cp) && !(include_marks && IsMark(cp));
      };
      if (scan < count && punctuation(value(scan))) {
        position = scan + 1u;
        while (position < count && punctuation(value(position))) ++position;
        while (position < count &&
               (value(position) == '\r' || value(position) == '\n'))
          ++position;
      } else if (IsWhitespace(current)) {
        size_t whitespace_end = position;
        size_t newline_end = 0;
        while (whitespace_end < count && IsWhitespace(value(whitespace_end))) {
          if (value(whitespace_end) == '\r' || value(whitespace_end) == '\n')
            newline_end = whitespace_end + 1u;
          ++whitespace_end;
        }
        if (newline_end != 0)
          position = newline_end;
        else if (whitespace_end - position > 1u && whitespace_end < count)
          position = whitespace_end - 1u;
        else
          position = whitespace_end;
      } else {
        ++position;
      }
    }
    output.emplace_back(text.data() + codepoints[begin].byte_offset,
                        codepoints[position].byte_offset -
                            codepoints[begin].byte_offset);
  }
  return output;
}

struct PairHash {
  size_t operator()(const std::pair<std::string, std::string>& value) const {
    const size_t first = std::hash<std::string>{}(value.first);
    const size_t second = std::hash<std::string>{}(value.second);
    return first ^ (second + 0x9e3779b9u + (first << 6u) + (first >> 2u));
  }
};

class BpeTokenizer final : public Tokenizer {
 public:
  explicit BpeTokenizer(Vocabulary vocabulary)
      : vocabulary_(std::move(vocabulary)), specials_(vocabulary_) {}

  static std::unique_ptr<BpeTokenizer> Create(Vocabulary vocabulary,
                                               std::string* error) {
    if (vocabulary.tokens.empty() ||
        vocabulary.types.size() != vocabulary.tokens.size() ||
        vocabulary.merges.empty()) {
      *error = "BPE metadata arrays are missing or inconsistent";
      return nullptr;
    }
    if (vocabulary.pre != "qwen2" && vocabulary.pre != "qwen35") {
      *error = "unsupported BPE pre-tokenizer profile: " + vocabulary.pre;
      return nullptr;
    }
    auto tokenizer = std::unique_ptr<BpeTokenizer>(
        new BpeTokenizer(std::move(vocabulary)));
    if (!tokenizer->Initialize(error)) return nullptr;
    return tokenizer;
  }

  bool Encode(const std::string& text, bool add_special,
              std::vector<int64_t>* output, std::string* error) const override {
    if (!ValidateUtf8(text.data(), text.size(), error)) return false;
    output->clear();
    if (add_special && vocabulary_.add_bos > 0 && vocabulary_.bos_id >= 0)
      output->push_back(vocabulary_.bos_id);
    const bool ok = specials_.Partition(
        text,
        [this](const std::string& run, std::vector<int64_t>* ids,
               std::string* run_error) {
          return EncodeRun(run, ids, run_error);
        },
        output, error);
    if (ok && add_special && vocabulary_.add_eos > 0 && vocabulary_.eos_id >= 0)
      output->push_back(vocabulary_.eos_id);
    return ok;
  }

  bool AppendDecoded(int64_t token_id, std::string* output,
                     std::string* error) const override {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.tokens.size()) {
      *error = "generated token id is outside the tokenizer vocabulary";
      return false;
    }
    if (token_id == vocabulary_.bos_id || token_id == vocabulary_.eos_id)
      return true;
    const size_t index = static_cast<size_t>(token_id);
    if (vocabulary_.types[index] == kTypeControl) return true;
    const std::string& token = vocabulary_.tokens[index];
    if (vocabulary_.types[index] == kTypeUserDefined) {
      output->append(token);
      return true;
    }
    size_t offset = 0;
    while (offset < token.size()) {
      int32_t codepoint = 0;
      const utf8proc_ssize_t length = utf8proc_iterate(
          reinterpret_cast<const utf8proc_uint8_t*>(token.data() + offset),
          static_cast<utf8proc_ssize_t>(token.size() - offset), &codepoint);
      if (length <= 0) {
        *error = "BPE vocabulary token is not valid UTF-8";
        return false;
      }
      const auto byte = unicode_to_byte_.find(codepoint);
      if (byte == unicode_to_byte_.end()) {
        *error = "BPE vocabulary token contains a non-byte-encoder codepoint";
        return false;
      }
      output->push_back(static_cast<char>(byte->second));
      offset += static_cast<size_t>(length);
    }
    return true;
  }

  int32_t bos_id() const override { return vocabulary_.bos_id; }
  int32_t eos_id() const override { return vocabulary_.eos_id; }
  size_t max_piece_bytes() const override { return max_piece_bytes_; }
  bool trim_initial_space() const override { return false; }
  const std::string& TokenText(int32_t token_id) const override {
    static const std::string empty;
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.tokens.size())
      return empty;
    return vocabulary_.tokens[static_cast<size_t>(token_id)];
  }

 private:
  struct Symbol {
    std::string text;
    int previous = -1;
    int next = -1;
    bool active = true;
  };
  struct Bigram {
    int left = -1;
    int right = -1;
    int rank = -1;
    bool operator<(const Bigram& other) const {
      return rank > other.rank || (rank == other.rank && left > other.left);
    }
  };

  bool Initialize(std::string* error) {
    token_to_id_.reserve(vocabulary_.tokens.size() * 2u);
    for (size_t i = 0; i != vocabulary_.tokens.size(); ++i) {
      token_to_id_.emplace(vocabulary_.tokens[i], static_cast<int32_t>(i));
      max_piece_bytes_ = std::max(max_piece_bytes_, vocabulary_.tokens[i].size());
    }
    byte_to_unicode_.fill(-1);
    int next = 0;
    for (int byte = 0x21; byte <= 0x7e; ++byte) byte_to_unicode_[byte] = byte;
    for (int byte = 0xa1; byte <= 0xac; ++byte) byte_to_unicode_[byte] = byte;
    for (int byte = 0xae; byte <= 0xff; ++byte) byte_to_unicode_[byte] = byte;
    for (int byte = 0; byte != 256; ++byte) {
      if (byte_to_unicode_[byte] < 0) byte_to_unicode_[byte] = 256 + next++;
      unicode_to_byte_.emplace(byte_to_unicode_[byte], static_cast<uint8_t>(byte));
    }
    merge_ranks_.reserve(vocabulary_.merges.size() * 2u);
    for (size_t i = 0; i != vocabulary_.merges.size(); ++i) {
      const std::string& merge = vocabulary_.merges[i];
      const size_t separator = merge.find(' ', 1u);
      if (separator == std::string::npos || separator + 1u >= merge.size()) {
        *error = "malformed BPE merge at index " + std::to_string(i);
        return false;
      }
      const auto inserted = merge_ranks_.emplace(
          std::make_pair(merge.substr(0, separator), merge.substr(separator + 1u)),
          static_cast<int>(i));
      if (!inserted.second) {
        *error = "duplicate BPE merge at index " + std::to_string(i);
        return false;
      }
    }
    return true;
  }

  std::string ByteEncode(const std::string& raw) const {
    std::string output;
    output.reserve(raw.size() * 2u);
    for (uint8_t byte : raw) output += CodepointToUtf8(byte_to_unicode_[byte]);
    return output;
  }

  bool EncodeWord(const std::string& word, std::vector<int64_t>* output,
                  std::string* error) const {
    std::vector<Symbol> symbols;
    for (size_t offset = 0; offset < word.size();) {
      int32_t codepoint = 0;
      const utf8proc_ssize_t length = utf8proc_iterate(
          reinterpret_cast<const utf8proc_uint8_t*>(word.data() + offset),
          static_cast<utf8proc_ssize_t>(word.size() - offset), &codepoint);
      if (length <= 0) {
        *error = "internal BPE byte encoding produced invalid UTF-8";
        return false;
      }
      Symbol symbol;
      symbol.text.assign(word.data() + offset, static_cast<size_t>(length));
      symbol.previous = static_cast<int>(symbols.size()) - 1;
      symbol.next = offset + static_cast<size_t>(length) == word.size()
                        ? -1
                        : static_cast<int>(symbols.size()) + 1;
      symbols.push_back(std::move(symbol));
      offset += static_cast<size_t>(length);
    }
    std::priority_queue<Bigram> queue;
    auto add = [&](int left, int right) {
      if (left < 0 || right < 0) return;
      const auto rank = merge_ranks_.find(
          {symbols[static_cast<size_t>(left)].text,
           symbols[static_cast<size_t>(right)].text});
      if (rank != merge_ranks_.end()) queue.push({left, right, rank->second});
    };
    for (size_t i = 1; i < symbols.size(); ++i)
      add(static_cast<int>(i - 1), static_cast<int>(i));
    while (!queue.empty()) {
      const Bigram bigram = queue.top();
      queue.pop();
      Symbol& left = symbols[static_cast<size_t>(bigram.left)];
      Symbol& right = symbols[static_cast<size_t>(bigram.right)];
      if (!left.active || !right.active || left.next != bigram.right) continue;
      const auto current_rank = merge_ranks_.find({left.text, right.text});
      if (current_rank == merge_ranks_.end() || current_rank->second != bigram.rank)
        continue;
      left.text += right.text;
      right.active = false;
      left.next = right.next;
      if (right.next >= 0)
        symbols[static_cast<size_t>(right.next)].previous = bigram.left;
      add(left.previous, bigram.left);
      add(bigram.left, left.next);
    }
    for (const Symbol& symbol : symbols) {
      if (!symbol.active) continue;
      const auto token = token_to_id_.find(symbol.text);
      if (token != token_to_id_.end()) {
        output->push_back(token->second);
        continue;
      }
      size_t offset = 0;
      while (offset < symbol.text.size()) {
        int32_t codepoint = 0;
        const utf8proc_ssize_t length = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t*>(symbol.text.data() + offset),
            static_cast<utf8proc_ssize_t>(symbol.text.size() - offset), &codepoint);
        if (length <= 0) return false;
        const auto byte = unicode_to_byte_.find(codepoint);
        if (byte == unicode_to_byte_.end()) {
          *error = "BPE fallback contains an unknown byte-encoder codepoint";
          return false;
        }
        const std::string encoded = CodepointToUtf8(codepoint);
        const auto byte_token = token_to_id_.find(encoded);
        if (byte_token == token_to_id_.end()) {
          *error = "BPE vocabulary has no token for byte " +
                   std::to_string(byte->second);
          return false;
        }
        output->push_back(byte_token->second);
        offset += static_cast<size_t>(length);
      }
    }
    return true;
  }

  bool EncodeRun(const std::string& raw, std::vector<int64_t>* output,
                 std::string* error) const {
    if (raw.empty()) return true;
    const std::vector<std::string> words =
        SplitQwen(raw, vocabulary_.pre == "qwen35", error);
    if (!error->empty()) return false;
    for (const std::string& word : words)
      if (!EncodeWord(ByteEncode(word), output, error)) return false;
    return true;
  }

  Vocabulary vocabulary_;
  SpecialTable specials_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>
      merge_ranks_;
  std::array<int32_t, 256> byte_to_unicode_{};
  std::unordered_map<int32_t, uint8_t> unicode_to_byte_;
  size_t max_piece_bytes_ = 1;
};

}  // namespace

bool ValidateUtf8(const char* data, size_t size, std::string* error) {
  if (error == nullptr || (data == nullptr && size != 0)) return false;
  size_t offset = 0;
  while (offset < size) {
    int32_t codepoint = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(data + offset),
        static_cast<utf8proc_ssize_t>(size - offset), &codepoint);
    if (length <= 0) {
      *error = "invalid UTF-8 at byte " + std::to_string(offset);
      return false;
    }
    offset += static_cast<size_t>(length);
  }
  return true;
}

std::unique_ptr<Tokenizer> Tokenizer::Create(Vocabulary vocabulary,
                                              std::string* error) {
  if (error == nullptr) return nullptr;
  error->clear();
  auto valid_utf8 = [&](const std::string& value, const char* field) {
    std::string detail;
    if (ValidateUtf8(value.data(), value.size(), &detail)) return true;
    *error = std::string("tokenizer ") + field + " is invalid: " + detail;
    return false;
  };
  if (!valid_utf8(vocabulary.model, "model") ||
      !valid_utf8(vocabulary.pre, "pre-tokenizer") ||
      !valid_utf8(vocabulary.chat_template, "chat template"))
    return nullptr;
  for (size_t i = 0; i != vocabulary.tokens.size(); ++i) {
    if (!valid_utf8(vocabulary.tokens[i], "vocabulary token")) return nullptr;
    if (i >= vocabulary.types.size() || vocabulary.types[i] < 0 ||
        vocabulary.types[i] > kTypeByte) {
      *error = "tokenizer vocabulary contains an unknown token type";
      return nullptr;
    }
  }
  for (const std::string& merge : vocabulary.merges)
    if (!valid_utf8(merge, "merge")) return nullptr;
  auto valid_token_id = [&](int32_t id, const char* field) {
    if (id == -1 ||
        (id >= 0 && static_cast<size_t>(id) < vocabulary.tokens.size()))
      return true;
    *error = std::string("tokenizer ") + field + " is outside the vocabulary";
    return false;
  };
  if (!valid_token_id(vocabulary.bos_id, "BOS id") ||
      !valid_token_id(vocabulary.eos_id, "EOS id") ||
      !valid_token_id(vocabulary.unk_id, "unknown-token id"))
    return nullptr;
  for (float score : vocabulary.scores) {
    if (!std::isfinite(score)) {
      *error = "tokenizer score is not finite";
      return nullptr;
    }
  }
  if (vocabulary.model == "llama")
    return SpmTokenizer::Create(std::move(vocabulary), error);
  if (vocabulary.model == "gpt2")
    return BpeTokenizer::Create(std::move(vocabulary), error);
  *error = "unsupported tokenizer model: " + vocabulary.model;
  return nullptr;
}

}  // namespace allspark::llm::internal
