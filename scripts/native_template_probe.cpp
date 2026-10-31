// Research CPU helper for the exact pinned header-only minja frontend.
// No model weights, Torch, CUDA, network, tools or generated code are executed.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <minja/chat-template.hpp>
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include "native_template_normalization_snapshot.h"
#include "native_request_conversion_snapshot.h"

using Json = nlohmann::ordered_json;

static std::string read_text(const char* path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot read pinned template metadata");
  std::ostringstream data;
  data << file.rdbuf();
  return data.str();
}

static std::string token(const Json& config, const char* key) {
  if (!config.contains(key) || config[key].is_null()) return "";
  if (config[key].is_string()) return config[key].get<std::string>();
  return config[key].value("content", "");
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("Expected template and tokenizer config paths");
    const auto config = Json::parse(read_text(argv[2]));
    minja::chat_template chat(normalize_minja_tests(read_text(argv[1])),
                              token(config, "bos_token"), token(config, "eos_token"));
    minja::chat_template_options options;
    for (std::string line; std::getline(std::cin, line);) {
      if (line.size() > 2 * 1024 * 1024) throw std::runtime_error("Excessive CPU template input");
      const auto request = Json::parse(line);
      minja::chat_template_inputs input;
      input.messages = request.at("messages");
      input.tools = Json::array();
      for (const auto& tool : request.value("tools", Json::array())) {
        const auto& function = tool.at("function");
        google::protobuf::Struct parameters;
        if (function.contains("parameters")) {
          const auto status = google::protobuf::util::JsonStringToMessage(
              function["parameters"].dump(), &parameters);
          if (!status.ok()) throw std::runtime_error("Invalid protobuf schema");
        }
        input.tools.push_back(Json{
            {"type", tool.value("type", "")},
            {"function", Json{{"name", function.value("name", "")},
                              {"description", function.value("description", "")},
                              {"parameters", proto_struct_to_json(parameters)}}}});
      }
      input.add_generation_prompt = true;
      input.extra_context = request.value("chat_template_kwargs", Json::object());
      const auto text = chat.apply(input, options);
      std::cout << Json{{"rendered_text", text},
                        {"requires_object_arguments", chat.original_caps().requires_object_arguments},
                        {"model_or_GPU_loaded", false}}.dump() << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
