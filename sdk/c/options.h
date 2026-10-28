#pragma once
#include "engine/engine.h"
#include <cmath>
#include <climits>
#include <filesystem>
#include <set>
#include <stdexcept>
namespace kapitsa_sdk {
inline std::string dump(const ni::ojson& j) {
  return j.dump(-1, ' ', false, ni::ojson::error_handler_t::replace);
}
inline void fields(const ni::ojson& j, std::initializer_list<const char*> allowed) {
  if (!j.is_object()) throw std::invalid_argument("expected a JSON object");
  const std::set<std::string> names(allowed.begin(), allowed.end());
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!names.count(it.key())) throw std::invalid_argument("unknown field: " + it.key());
}
inline int integer(const ni::ojson& j, const char* key, int fallback, int lo, int hi) {
  if (!j.contains(key)) return fallback;
  if (!j[key].is_number_integer()) throw std::invalid_argument(std::string(key) + " must be an integer");
  // JSON integers above INT64_MAX are unsigned. Check before converting: UINT64_MAX
  // would otherwise become -1 and be accepted as spec_depth's default sentinel.
  if (j[key].is_number_unsigned()) {
    const auto v = j[key].get<uint64_t>();
    if (v > uint64_t(hi) || (lo > 0 && v < uint64_t(lo)))
      throw std::invalid_argument(std::string(key) + " out of range");
    return int(v);
  }
  const auto v = j[key].get<int64_t>();
  if (v < lo || v > hi) throw std::invalid_argument(std::string(key) + " out of range");
  return int(v);
}
inline double number(const ni::ojson& j, const char* key, double fallback, double lo, double hi) {
  if (!j.contains(key)) return fallback;
  if (!j[key].is_number()) throw std::invalid_argument(std::string(key) + " must be a number");
  double v = j[key].get<double>();
  if (!std::isfinite(v) || v < lo || v > hi) throw std::invalid_argument(std::string(key) + " out of range");
  return v;
}
struct CreateOptions { ni::EngineOptions engine; int gpu_lock_timeout_ms = 0; };
inline CreateOptions options(const ni::ojson& j) {
  fields(j, {"model_dir", "context", "slots", "prefill_chunk", "spec_depth", "mem_budget_gb", "dflash", "disk_cache",
             "gpu_lock_timeout_ms"});
  CreateOptions result;
  result.gpu_lock_timeout_ms = integer(j, "gpu_lock_timeout_ms", 0, 0, INT_MAX);
  auto& o = result.engine;
  o.model_dir = j.at("model_dir").get<std::string>();
  if (o.model_dir.empty() || !std::filesystem::is_regular_file(std::filesystem::path(o.model_dir) / "config.json"))
    throw std::invalid_argument("model_dir must be a local Kapitsa model folder containing config.json");
  o.context = integer(j, "context", o.context, 1, 1048576);
  o.slots = integer(j, "slots", 1, 1, 4);
  o.prefill_chunk = integer(j, "prefill_chunk", 0, 0, 8192);
  o.spec_depth = integer(j, "spec_depth", -1, -1, 32);
  o.mem_budget_gb = number(j, "mem_budget_gb", 0, 0, 4096);
  o.dflash = j.value("dflash", std::string{});
  o.disk_cache = j.value("disk_cache", std::string{});
  return result;
}
inline ni::ChatRequest request(const ni::ojson& j, ni::ChatRequest r) {
  fields(j, {"messages", "tools", "template_kwargs", "max_tokens", "temperature", "top_p", "top_k", "seed", "stop"});
  r.messages = j.at("messages");
  if (!r.messages.is_array() || r.messages.empty()) throw std::invalid_argument("messages must be a nonempty array");
  for (const auto& m : r.messages) {
    if (!m.is_object() || !m.contains("role") || !m["role"].is_string())
      throw std::invalid_argument("each message needs a string role");
    const auto role = m["role"].get<std::string>();
    if (role != "system" && role != "user" && role != "assistant" && role != "tool")
      throw std::invalid_argument("unsupported message role");
    if (m.contains("content") && !(m["content"].is_string() || m["content"].is_array() || m["content"].is_null()))
      throw std::invalid_argument("message content must be string, array or null");
  }
  r.tools = j.value("tools", ni::ojson::array());
  r.template_kwargs = j.value("template_kwargs", ni::ojson::object());
  if (!r.tools.is_array() || !r.template_kwargs.is_object()) throw std::invalid_argument("invalid tools/template_kwargs");
  r.max_tokens = integer(j, "max_tokens", r.max_tokens, 1, 1048576);
  r.temperature = float(number(j, "temperature", r.temperature, 0, 100));
  r.top_p = float(number(j, "top_p", r.top_p, 0, 1));
  r.top_k = integer(j, "top_k", r.top_k, 0, 10000000);
  if (j.contains("seed")) {
    if (!j["seed"].is_number_integer() || j["seed"].get<int64_t>() < 0 || j["seed"].get<uint64_t>() > UINT32_MAX)
      throw std::invalid_argument("seed must be a uint32");
    r.seed = j["seed"].get<uint32_t>();
  }
  r.stop = j.value("stop", std::vector<std::string>{});
  return r;
}
inline ni::ojson calls(const std::vector<ni::ToolCall>& calls) {
  auto a = ni::ojson::array();
  for (const auto& c : calls) a.push_back({{"id", c.id}, {"name", c.name}, {"arguments", c.arguments}});
  return a;
}
}  // namespace kapitsa_sdk
