#include <iostream>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "qwen25_detector.h"
class ParserHandle final {
 public:
  explicit ParserHandle(xllm::function_call::BaseFormatDetector* detector) : detector_(detector) {}
  xllm::function_call::BaseFormatDetector* get_detector() { return detector_; }
 private:
  xllm::function_call::BaseFormatDetector* detector_;
};
class StreamOutputParser final {
 public:
  explicit StreamOutputParser(xllm::function_call::BaseFormatDetector* detector) : parser_(detector) {}
  ParserHandle* get_tool_call_parser(size_t) { return &parser_; }
 private:
  ParserHandle parser_;
};
using SendFunc = std::function<bool(const std::string&, int)>;
#include "flush_original.hpp"
int main() {
  nlohmann::json input;
  std::cin >> input;
  nlohmann::json results = nlohmann::json::array();
  const std::string model_text = input.at("model_text").get<std::string>();
  std::vector<xllm::JsonTool> tools;
  tools.emplace_back("function", xllm::JsonFunction("generate_image", "", nlohmann::json::object()));
  for (int32_t chunk_size : std::vector<int32_t>{1, 2, 3, 5, 8, 16, 32, 64, 0}) {
    auto detector = std::make_shared<xllm::function_call::Qwen25Detector>();
    std::string arguments;
    nlohmann::json deltas = nlohmann::json::array();
    const size_t width = chunk_size == 0 ? model_text.size() : static_cast<size_t>(chunk_size);
    for (size_t start = 0; start < model_text.size(); start += width) {
      const auto result = detector->parse_streaming_increment(model_text.substr(start, width), tools);
      for (const auto& call : result.calls) {
        arguments += call.parameters;
        deltas.push_back(call.parameters);
      }
    }
    auto stream_parser = std::make_shared<StreamOutputParser>(detector.get());
    check_for_unstreamed_tool_args(stream_parser, 0, [&](const std::string& text, int) {
      arguments += text;
      deltas.push_back(text);
      return true;
    });
    nlohmann::json parsed;
    bool valid = false;
    try { parsed = nlohmann::json::parse(arguments); valid = true; } catch (const nlohmann::json::parse_error&) {}
    const auto complete = detector->detect_and_parse(model_text, tools);
    results.push_back({{"chunk_size", chunk_size}, {"arguments", arguments}, {"deltas", deltas},
      {"valid_JSON", valid}, {"matches_expected", valid && parsed == input.at("expected_arguments")},
      {"nonstream_calls", complete.calls.size()}});
  }
  std::cout << results.dump() << '\n';
}
