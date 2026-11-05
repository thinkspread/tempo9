/* Copyright (c) 2026 Jiejing Zhang. */

#include "template_renderer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <nlohmann/json.hpp>
#include <utility>

#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include "tokenizer.h"

namespace allspark::llm::internal {
namespace {

using Json = nlohmann::ordered_json;

template <typename T>
bool StructOk(const T* value, size_t minimum) {
  return value != nullptr && value->struct_size >= minimum &&
         value->struct_size <= sizeof(T) * 4u;
}

bool ViewOk(te9_llm_string_view view) {
  return view.data != nullptr || view.size == 0;
}

std::string Copy(te9_llm_string_view view) {
  return view.size == 0 ? std::string() : std::string(view.data, view.size);
}

bool TextViewOk(te9_llm_string_view view, const char* field, bool allow_empty,
                std::string* error) {
  if (!ViewOk(view) || (!allow_empty && view.size == 0)) {
    *error = std::string(field) + (allow_empty ? " has an invalid string view"
                                               : " must be non-empty");
    return false;
  }
  std::string detail;
  if (!ValidateUtf8(view.data, view.size, &detail)) {
    *error = std::string(field) + " is not valid UTF-8: " + detail;
    return false;
  }
  return true;
}

const char* RoleName(int32_t role) {
  switch (role) {
    case TE9_LLM_CHAT_SYSTEM:
      return "system";
    case TE9_LLM_CHAT_DEVELOPER:
      return "developer";
    case TE9_LLM_CHAT_USER:
      return "user";
    case TE9_LLM_CHAT_ASSISTANT:
      return "assistant";
    case TE9_LLM_CHAT_TOOL:
      return "tool";
    default:
      return nullptr;
  }
}

bool ParseObject(te9_llm_string_view view, const char* field, Json* output,
                 std::string* error) {
  if (!TextViewOk(view, field, false, error)) return false;
  try {
    *output = Json::parse(view.data, view.data + view.size);
  } catch (const std::exception& exception) {
    *error = std::string(field) + " is invalid JSON: " + exception.what();
    return false;
  }
  if (!output->is_object()) {
    *error = std::string(field) + " must be a JSON object";
    return false;
  }
  return true;
}

bool CacheTtl(int32_t cache_control, int32_t* ttl_seconds, std::string* error) {
  switch (cache_control) {
    case TE9_LLM_CACHE_NONE:
      *ttl_seconds = 0;
      return true;
    case TE9_LLM_CACHE_EPHEMERAL:
      *ttl_seconds = 300;
      return true;
    case TE9_LLM_CACHE_EPHEMERAL_1H:
      *ttl_seconds = 3600;
      return true;
    default:
      *error = "chat cache control has an unknown value";
      return false;
  }
}

struct CacheProbe {
  Json prefix;
  Json previous_prefix;
  int32_t ttl_seconds = 300;
};

// Joins the text parts of typed content for a template that only reads string
// content. Media cannot be flattened into text, so it is refused rather than
// dropped.
bool FlattenContent(const Json& parts, Json* output, std::string* error) {
  std::string text;
  for (const Json& part : parts) {
    if (part.value("type", "") != "text") {
      *error = "chat template does not accept " + part.value("type", "") +
               " content";
      return false;
    }
    if (!text.empty()) text += '\n';
    text += part.value("text", "");
  }
  *output = std::move(text);
  return true;
}

}  // namespace

class TemplateRenderer::Impl {
 public:
  Impl(const std::string& source, const std::string& bos_token,
       const std::string& eos_token)
      : bos(bos_token), eos(eos_token) {
    jinja::lexer lexer;
    jinja::lexer_result tokens = lexer.tokenize(source);
    program = jinja::parse_from_tokens(tokens);
    this->source = std::move(tokens.source);
    caps = jinja::caps_get(program);
  }

  // The message rewrites llama.cpp applies before rendering (common/chat.cpp,
  // namespace workaround), keyed on the capabilities probed from the template.
  bool Adapt(Json* messages, std::string* error) const {
    const bool only_string =
        caps.supports_string_content && !caps.supports_typed_content;
    const bool only_typed =
        caps.supports_typed_content && !caps.supports_string_content;
    const bool keep_developer =
        source.find("<|channel|>") != std::string::npos;
    for (Json& message : *messages) {
      if (!keep_developer && message.at("role") == "developer")
        message["role"] = "system";
      Json& content = message["content"];
      if (only_typed && content.is_string()) {
        content = Json::array({{{"type", "text"}, {"text", content}}});
      } else if (only_string && content.is_array()) {
        if (!FlattenContent(content, &content, error)) return false;
      }
      if (!message.contains("tool_calls")) continue;
      for (Json& call : message["tool_calls"]) {
        Json& arguments = call["function"]["arguments"];
        if (caps.supports_object_arguments && arguments.is_string())
          arguments = Json::parse(arguments.get<std::string>());
      }
    }
    if (!caps.supports_system_role && !messages->empty() &&
        messages->front().at("role") == "system") {
      Json system = std::move(messages->front());
      messages->erase(messages->begin());
      if (!messages->empty()) {
        Json& next = messages->front()["content"];
        if (system["content"].is_string() && next.is_string()) {
          next = system["content"].get<std::string>() + "\n" +
                 next.get<std::string>();
        } else if (system["content"].is_array() && next.is_array()) {
          Json merged = system["content"];
          for (Json& part : next) merged.push_back(std::move(part));
          next = std::move(merged);
        } else {
          *error = "chat template has no system role and the system message "
                   "cannot be merged into the next message";
          return false;
        }
      }
    }
    return true;
  }

  bool Apply(Json messages, const Json& tools, bool add_generation_prompt,
             const Json& extra, std::string* output,
             std::string* error) const {
    if (!Adapt(&messages, error)) return false;
    Json input = extra;
    input["messages"] = std::move(messages);
    // Hugging Face passes tools=None when there are none; [] would make
    // `tools is not none` true.
    input["tools"] = tools.empty() ? Json(nullptr) : tools;
    input["add_generation_prompt"] = add_generation_prompt;
    input["bos_token"] = bos;
    input["eos_token"] = eos;
    jinja::context context(source);
    jinja::global_from_json(context, common_json::parse(input.dump()),
                            /*mark_input=*/false);
    jinja::runtime runtime(context);
    const jinja::value_string parts =
        jinja::runtime::gather_string_parts(runtime.execute(program));
    output->clear();
    for (const auto& part : parts->val_str.parts) *output += part.val;
    return true;
  }

  std::string source;
  std::string bos;
  std::string eos;
  jinja::program program;
  jinja::caps caps;
};

TemplateRenderer::TemplateRenderer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TemplateRenderer::~TemplateRenderer() = default;

std::unique_ptr<TemplateRenderer> TemplateRenderer::Create(
    const std::string& source, const std::string& bos_token,
    const std::string& eos_token, std::string* error) {
  if (error == nullptr) return nullptr;
  error->clear();
  if (source.empty()) {
    *error = "model metadata contains no chat template";
    return nullptr;
  }
  try {
    return std::unique_ptr<TemplateRenderer>(new TemplateRenderer(
        std::make_unique<Impl>(source, bos_token, eos_token)));
  } catch (const std::exception& exception) {
    *error = std::string("chat template is unsupported or invalid: ") +
             exception.what();
  } catch (...) {
    *error = "chat template parsing failed with an unknown exception";
  }
  return nullptr;
}

bool TemplateRenderer::Render(const te9_llm_chat_request& request,
                              std::string* output,
                              std::vector<RenderedCachePoint>* cache_points,
                              std::string* error) const {
  if (output == nullptr || cache_points == nullptr || error == nullptr)
    return false;
  output->clear();
  cache_points->clear();
  error->clear();
  constexpr size_t kChatRequestV12Size =
      offsetof(te9_llm_chat_request, reasoning);
  constexpr size_t kChatRequestV13Size =
      offsetof(te9_llm_chat_request, reasoning) + sizeof(request.reasoning);
  constexpr size_t kChatRequestV16Size =
      offsetof(te9_llm_chat_request, thinking) + sizeof(request.thinking);
  constexpr size_t kChatRequestV19Size =
      offsetof(te9_llm_chat_request, preserve_thinking) +
      sizeof(request.preserve_thinking);
  if (!StructOk(&request, kChatRequestV12Size) ||
      (request.struct_size > kChatRequestV12Size &&
       request.struct_size < kChatRequestV13Size) ||
      (request.struct_size > kChatRequestV13Size &&
       request.struct_size < kChatRequestV16Size) ||
      (request.struct_size > kChatRequestV16Size &&
       request.struct_size < kChatRequestV19Size) ||
      (request.message_count != 0 && request.messages == nullptr) ||
      (request.tool_count != 0 && request.tools == nullptr)) {
    *error = "te9_llm_chat_request has an invalid struct or null array";
    return false;
  }
  if (request.message_count == 0) {
    *error = "chat request must contain at least one message";
    return false;
  }
  if (request.add_generation_prompt != 0 &&
      request.add_generation_prompt != 1) {
    *error = "add_generation_prompt must be zero or one";
    return false;
  }
  Json messages = Json::array();
  std::vector<CacheProbe> message_probes;
  size_t cache_point_count = 0;
  for (size_t i = 0; i != request.message_count; ++i) {
    const te9_llm_chat_message& message = request.messages[i];
    if (!StructOk(&message, sizeof(message)) ||
        (message.content_count != 0 && message.content == nullptr) ||
        !ViewOk(message.name)) {
      *error = "chat message " + std::to_string(i) + " is invalid";
      return false;
    }
    if (!TextViewOk(message.name, "chat message name", true, error))
      return false;
    const char* role = RoleName(message.role);
    if (role == nullptr) {
      *error = "chat message " + std::to_string(i) + " has an unknown role";
      return false;
    }
    Json rendered = {{"role", role}};
    if (message.name.size != 0) rendered["name"] = Copy(message.name);
    std::string text;
    const bool message_has_media =
        message.content_count != 0 &&
        std::any_of(message.content, message.content + message.content_count,
                    [](const te9_llm_chat_content_part& part) {
                      return part.kind == TE9_LLM_CHAT_PART_IMAGE ||
                             part.kind == TE9_LLM_CHAT_PART_AUDIO;
                    });
    Json content_blocks = Json::array();
    Json tool_calls = Json::array();
    std::string reasoning_content;
    bool has_reasoning = false;
    bool has_tool_result = false;
    for (size_t j = 0; j != message.content_count; ++j) {
      const auto current_message = [&] {
        Json current = rendered;
        current["content"] = message_has_media ? content_blocks : Json(text);
        if (has_reasoning)
          current["reasoning_content"] = reasoning_content;
        if (!tool_calls.empty()) current["tool_calls"] = tool_calls;
        return current;
      };
      const te9_llm_chat_content_part& part = message.content[j];
      if (!StructOk(&part, sizeof(part)) || !ViewOk(part.text) ||
          !ViewOk(part.tool_call_id) || !ViewOk(part.tool_name) ||
          !ViewOk(part.arguments_json)) {
        *error = "chat content part is invalid";
        return false;
      }
      if (!TextViewOk(part.text, "chat content text", true, error) ||
          !TextViewOk(part.tool_call_id, "tool call id", true, error) ||
          !TextViewOk(part.tool_name, "tool name", true, error) ||
          !TextViewOk(part.arguments_json, "tool-call arguments", true, error))
        return false;
      int32_t cache_ttl_seconds = 0;
      if (!CacheTtl(part.cache_control, &cache_ttl_seconds, error))
        return false;
      Json previous_prefix;
      if (cache_ttl_seconds != 0) {
        if (++cache_point_count > 4) {
          *error = "chat requests support at most four cache breakpoints";
          return false;
        }
        previous_prefix = messages;
        previous_prefix.push_back(current_message());
      }
      if (part.kind == TE9_LLM_CHAT_PART_TEXT) {
        if (part.tool_call_id.size != 0 || part.tool_name.size != 0 ||
            part.arguments_json.size != 0 ||
            message.role == TE9_LLM_CHAT_TOOL) {
          *error = "text chat parts cannot carry tool fields or a tool role";
          return false;
        }
        const std::string value = Copy(part.text);
        text += value;
        if (message_has_media)
          content_blocks.push_back({{"type", "text"}, {"text", value}});
      } else if (part.kind == TE9_LLM_CHAT_PART_REASONING) {
        if (message.role != TE9_LLM_CHAT_ASSISTANT || has_reasoning ||
            part.text.size == 0 ||
            part.tool_call_id.size != 0 || part.tool_name.size != 0 ||
            part.arguments_json.size != 0) {
          *error =
              "historical reasoning requires one assistant reasoning part "
              "with no tool fields";
          return false;
        }
        has_reasoning = true;
        reasoning_content = Copy(part.text);
      } else if (part.kind == TE9_LLM_CHAT_PART_IMAGE ||
                 part.kind == TE9_LLM_CHAT_PART_AUDIO) {
        if (message.role != TE9_LLM_CHAT_USER || part.text.size != 0 ||
            part.tool_call_id.size != 0 || part.tool_name.size != 0 ||
            part.arguments_json.size != 0) {
          *error =
              "media chat parts require a user role and no text or tool fields";
          return false;
        }
        content_blocks.push_back(
            {{"type",
              part.kind == TE9_LLM_CHAT_PART_IMAGE ? "image" : "audio"}});
      } else if (part.kind == TE9_LLM_CHAT_PART_TOOL_CALL) {
        if (message.role != TE9_LLM_CHAT_ASSISTANT || part.text.size != 0 ||
            part.tool_call_id.size == 0 || part.tool_name.size == 0) {
          *error =
              "tool calls require an assistant role, id, name, and no text";
          return false;
        }
        Json arguments;
        if (!ParseObject(part.arguments_json, "tool-call arguments", &arguments,
                         error))
          return false;
        tool_calls.push_back({{"id", Copy(part.tool_call_id)},
                              {"type", "function"},
                              {"function",
                               {{"name", Copy(part.tool_name)},
                                {"arguments", arguments.dump()}}}});
      } else if (part.kind == TE9_LLM_CHAT_PART_TOOL_RESULT ||
                 part.kind == TE9_LLM_CHAT_PART_TOOL_ERROR_RESULT) {
        if (has_tool_result || message.content_count != 1 ||
            message.role != TE9_LLM_CHAT_TOOL || part.tool_call_id.size == 0 ||
            part.arguments_json.size != 0) {
          *error = "a tool result must be the only part of a tool message";
          return false;
        }
        has_tool_result = true;
        text = Copy(part.text);
        rendered["tool_call_id"] = Copy(part.tool_call_id);
        if (part.tool_name.size != 0) rendered["name"] = Copy(part.tool_name);
        if (part.kind == TE9_LLM_CHAT_PART_TOOL_ERROR_RESULT)
          rendered["is_error"] = true;
      } else {
        *error = "chat content part kind is unsupported";
        return false;
      }
      if (cache_ttl_seconds != 0) {
        Json prefix = messages;
        prefix.push_back(current_message());
        message_probes.push_back(
            {std::move(prefix), std::move(previous_prefix), cache_ttl_seconds});
      }
    }
    if (message.role == TE9_LLM_CHAT_TOOL && !has_tool_result) {
      *error = "a tool message requires one tool result with a call id";
      return false;
    }
    rendered["content"] = message_has_media ? content_blocks : Json(text);
    if (has_reasoning)
      rendered["reasoning_content"] = std::move(reasoning_content);
    if (!tool_calls.empty()) rendered["tool_calls"] = std::move(tool_calls);
    messages.push_back(std::move(rendered));
  }

  Json tools = Json::array();
  std::vector<CacheProbe> tool_probes;
  for (size_t i = 0; i != request.tool_count; ++i) {
    const te9_llm_chat_tool& tool = request.tools[i];
    if (!StructOk(&tool, sizeof(tool)) || !ViewOk(tool.name) ||
        !ViewOk(tool.description) || !ViewOk(tool.input_schema_json) ||
        tool.name.size == 0) {
      *error = "chat tool " + std::to_string(i) + " is invalid";
      return false;
    }
    if (!TextViewOk(tool.name, "tool name", false, error) ||
        !TextViewOk(tool.description, "tool description", true, error) ||
        !TextViewOk(tool.input_schema_json, "tool input schema", false, error))
      return false;
    int32_t cache_ttl_seconds = 0;
    if (!CacheTtl(tool.cache_control, &cache_ttl_seconds, error)) return false;
    if (cache_ttl_seconds != 0 && ++cache_point_count > 4) {
      *error = "chat requests support at most four cache breakpoints";
      return false;
    }
    Json schema;
    if (!ParseObject(tool.input_schema_json, "tool input schema", &schema,
                     error))
      return false;
    Json previous_tools;
    if (cache_ttl_seconds != 0) previous_tools = tools;
    tools.push_back({{"type", "function"},
                     {"function",
                      {{"name", Copy(tool.name)},
                       {"description", Copy(tool.description)},
                       {"parameters", std::move(schema)}}}});
    if (cache_ttl_seconds != 0)
      tool_probes.push_back(
          {tools, std::move(previous_tools), cache_ttl_seconds});
  }

  bool saw_five_minute_ttl = false;
  const auto validate_ttl_order = [&](const CacheProbe& probe) {
    if (probe.ttl_seconds == 300) saw_five_minute_ttl = true;
    if (probe.ttl_seconds == 3600 && saw_five_minute_ttl) {
      *error =
          "one-hour cache breakpoints must precede five-minute breakpoints";
      return false;
    }
    return true;
  };
  for (const CacheProbe& probe : tool_probes)
    if (!validate_ttl_order(probe)) return false;
  for (const CacheProbe& probe : message_probes)
    if (!validate_ttl_order(probe)) return false;

  Json extra = Json::object();
  if (request.tool_choice != TE9_LLM_TOOL_NAMED &&
      request.named_tool.size != 0) {
    *error = "named_tool is valid only with named tool choice";
    return false;
  }
  switch (request.tool_choice) {
    case TE9_LLM_TOOL_AUTO:
      extra["tool_choice"] = "auto";
      break;
    case TE9_LLM_TOOL_NONE:
      extra["tool_choice"] = "none";
      break;
    case TE9_LLM_TOOL_REQUIRED:
      if (request.tool_count == 0) {
        *error = "required tool choice needs at least one declared tool";
        return false;
      }
      extra["tool_choice"] = "required";
      break;
    case TE9_LLM_TOOL_NAMED:
      if (!TextViewOk(request.named_tool, "named tool choice", false, error))
        return false;
      if (request.tool_count == 0 ||
          std::none_of(request.tools, request.tools + request.tool_count,
                       [&](const te9_llm_chat_tool& tool) {
                         return tool.name.size == request.named_tool.size &&
                                std::memcmp(tool.name.data,
                                            request.named_tool.data,
                                            tool.name.size) == 0;
                       })) {
        *error = "named tool choice does not match a declared tool";
        return false;
      }
      extra["tool_choice"] =
          Json{{"type", "function"},
               {"function", {{"name", Copy(request.named_tool)}}}};
      break;
    default:
      *error = "chat request has an unknown tool choice";
      return false;
  }
  const bool has_reasoning = request.struct_size >= kChatRequestV13Size;
  if (has_reasoning && request.reasoning != nullptr) {
    constexpr size_t kReasoningMinimum =
        offsetof(te9_llm_reasoning_config, effort) + sizeof(int32_t);
    if (!StructOk(request.reasoning, kReasoningMinimum)) {
      *error = "reasoning config has an invalid struct";
      return false;
    }
    const char* effort = nullptr;
    switch (request.reasoning->effort) {
      case TE9_LLM_REASONING_DEFAULT:
        break;
      case TE9_LLM_REASONING_NONE:
        effort = "none";
        break;
      case TE9_LLM_REASONING_MINIMAL:
        effort = "minimal";
        break;
      case TE9_LLM_REASONING_LOW:
        effort = "low";
        break;
      case TE9_LLM_REASONING_MEDIUM:
        effort = "medium";
        break;
      case TE9_LLM_REASONING_HIGH:
        effort = "high";
        break;
      case TE9_LLM_REASONING_XHIGH:
        effort = "xhigh";
        break;
      case TE9_LLM_REASONING_MAX:
        effort = "max";
        break;
      default:
        *error = "reasoning config has an unknown effort";
        return false;
    }
    if (effort != nullptr) {
      const bool enabled = request.reasoning->effort != TE9_LLM_REASONING_NONE;
      extra["enable_thinking"] = enabled;
      extra["thinking"] = enabled;
      extra["reasoning_effort"] = effort;
    }
  }
  const bool has_thinking = request.struct_size >= kChatRequestV16Size;
  if (has_thinking && request.thinking != nullptr) {
    constexpr size_t kThinkingModeSize =
        offsetof(te9_llm_thinking_config, mode) + sizeof(int32_t);
    constexpr size_t kThinkingBudgetSize =
        offsetof(te9_llm_thinking_config, budget_tokens) + sizeof(int64_t);
    if (!StructOk(request.thinking, kThinkingModeSize)) {
      *error = "thinking config has an invalid struct";
      return false;
    }
    const bool reasoning_is_none =
        has_reasoning && request.reasoning != nullptr &&
        request.reasoning->effort == TE9_LLM_REASONING_NONE;
    switch (request.thinking->mode) {
      case TE9_LLM_THINKING_DEFAULT:
        if (request.thinking->struct_size >= kThinkingBudgetSize &&
            request.thinking->budget_tokens != 0) {
          *error = "default thinking cannot carry a token budget";
          return false;
        }
        break;
      case TE9_LLM_THINKING_DISABLED:
        if (request.thinking->struct_size >= kThinkingBudgetSize &&
            request.thinking->budget_tokens != 0) {
          *error = "disabled thinking cannot carry a token budget";
          return false;
        }
        extra["enable_thinking"] = false;
        extra["thinking"] = false;
        extra["thinking_mode"] = "disabled";
        extra["budget_tokens"] = 0;
        extra["thinking_budget"] = 0;
        break;
      case TE9_LLM_THINKING_ENABLED:
        if (reasoning_is_none) {
          *error = "enabled thinking conflicts with reasoning effort none";
          return false;
        }
        if (request.thinking->struct_size < kThinkingBudgetSize ||
            request.thinking->budget_tokens < 1024) {
          *error = "enabled thinking requires budget_tokens of at least 1024";
          return false;
        }
        extra["enable_thinking"] = true;
        extra["thinking"] = true;
        extra["thinking_mode"] = "enabled";
        extra["budget_tokens"] = request.thinking->budget_tokens;
        extra["thinking_budget"] = request.thinking->budget_tokens;
        break;
      case TE9_LLM_THINKING_ADAPTIVE:
        if (reasoning_is_none) {
          *error = "adaptive thinking conflicts with reasoning effort none";
          return false;
        }
        if (request.thinking->struct_size >= kThinkingBudgetSize &&
            request.thinking->budget_tokens != 0) {
          *error = "adaptive thinking cannot carry a token budget";
          return false;
        }
        extra["enable_thinking"] = true;
        extra["thinking"] = true;
        extra["thinking_mode"] = "adaptive";
        extra["budget_tokens"] = 0;
        extra["thinking_budget"] = 0;
        break;
      default:
        *error = "thinking config has an unknown mode";
        return false;
    }
  }
  if (request.struct_size >= kChatRequestV19Size) {
    switch (request.preserve_thinking) {
      case TE9_LLM_PRESERVE_THINKING_DEFAULT:
        break;
      case TE9_LLM_PRESERVE_THINKING_DISABLED:
        extra["preserve_thinking"] = false;
        break;
      case TE9_LLM_PRESERVE_THINKING_ENABLED:
        extra["preserve_thinking"] = true;
        break;
      default:
        *error = "chat request has an unknown preserve_thinking value";
        return false;
    }
  }
  try {
    const auto render = [&](const Json& input_messages, const Json& input_tools,
                            bool add_generation_prompt, std::string* target) {
      if (!impl_->Apply(input_messages, input_tools, add_generation_prompt,
                        extra, target, error))
        return false;
      std::string detail;
      if (!ValidateUtf8(target->data(), target->size(), &detail)) {
        *error = "chat template output is not valid UTF-8: " + detail;
        target->clear();
        return false;
      }
      return true;
    };
    if (!render(messages, tools, request.add_generation_prompt != 0, output))
      return false;
    Json tool_context_messages = Json::array();
    for (const Json& message : messages) {
      if (!message.contains("role") || !message.at("role").is_string() ||
          (message.at("role") != "system" && message.at("role") != "developer"))
        break;
      tool_context_messages.push_back(message);
    }
    for (const CacheProbe& probe : tool_probes) {
      RenderedCachePoint point;
      point.ttl_seconds = probe.ttl_seconds;
      if (!render(tool_context_messages, probe.prefix, false, &point.prefix) ||
          !render(tool_context_messages, probe.previous_prefix, false,
                  &point.previous_prefix))
        return false;
      cache_points->push_back(std::move(point));
    }
    for (const CacheProbe& probe : message_probes) {
      RenderedCachePoint point;
      point.ttl_seconds = probe.ttl_seconds;
      if (!render(probe.prefix, tools, false, &point.prefix) ||
          !render(probe.previous_prefix, tools, false, &point.previous_prefix))
        return false;
      cache_points->push_back(std::move(point));
    }
    return true;
  } catch (const std::exception& exception) {
    *error = std::string("chat template render failed: ") + exception.what();
  } catch (...) {
    *error = "chat template render failed with an unknown exception";
  }
  return false;
}

}  // namespace allspark::llm::internal
