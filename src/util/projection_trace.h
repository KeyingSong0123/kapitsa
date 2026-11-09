#pragma once

// Experimental, buffered metadata trace. No GPU commands, waits, or kernel changes.
// NI_KFLASH_PROJECTION_TRACE=<jsonl> enables it; leave unset for acceptance timing.
#include <nlohmann/json.hpp>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace ni::projection_trace {
inline const char* path() {
  static const char* p = getenv("NI_KFLASH_PROJECTION_TRACE");
  return p && *p ? p : nullptr;
}
struct Key {
  std::string stage, role, phase, kernel;
  std::vector<std::string> names;
  std::vector<uint32_t> N, storage_N, bits, y_ld, add;
  uint32_t K, R, nf, cs, ks, tg, threads, packed;
  auto fields() const { return std::tie(stage, role, phase, kernel, names, N, storage_N, bits, y_ld, add, K, R, nf, cs, ks, tg, threads, packed); }
  bool operator<(const Key& o) const { return fields() < o.fields(); }
};
struct Cycle {
  int cycle, position, accepted_drafts, emitted, verified_rows;
  double draft_start_ms, draft_end_ms, verify_start_ms, verify_end_ms;
};
struct CanonicalXpCounters {
  uint64_t plain = 0, gu = 0, xp_packs = 0, wsb_producers = 0, gemms = 0, raw_fallback_unpacks = 0;
  uint64_t serialized_groups = 0, serialized_segments = 0, lost_grouped_overlap_sets = 0;
};
struct State {
  bool active = false;
  std::string stage = "prefill", role = "unknown", phase = "unknown";
  std::unordered_map<const void*, std::string> names;
  std::map<Key, uint64_t> calls;
  std::vector<Cycle> cycles;
  CanonicalXpCounters canonical_xp;
  double final_flush_ms = 0, terminal_extra_draft_ms = 0;
};
inline State& state() { static thread_local State s; return s; }
inline bool enabled() { return path() && state().active; }
inline void begin() {
  if (!path()) return;
  auto& s = state();
  s.active = true; s.stage = "prefill"; s.calls.clear(); s.cycles.clear(); s.canonical_xp = {};
  s.cycles.reserve(512); s.final_flush_ms = s.terminal_extra_draft_ms = 0;
}
inline void label(const void* key, std::string value) { state().names[key] = std::move(value); }
struct Scope {
  std::string role, phase;
  bool on;
  Scope(const char* r, const char* p) : on(enabled()) {
    if (on) { auto& s = state(); role = s.role; phase = s.phase; s.role = r; s.phase = p; }
  }
  ~Scope() { if (on) { state().role = role; state().phase = phase; } }
};
template<class Segments>
inline void record(const std::string& kernel, const Segments& segs, uint32_t K, uint32_t R,
                   uint32_t nf, uint32_t cs, uint32_t ks, uint32_t tg, uint32_t threads, bool packed = false, uint32_t selected_n = 0) {
  if (!enabled()) return;
  auto& s = state();
  Key k{s.stage, s.role, s.phase, kernel, {}, {}, {}, {}, {}, {}, K, R, nf, cs, ks, tg, threads, packed ? 1u : 0u};
  for (const auto& seg : segs) {
    const auto it = s.names.find(seg.L);
    k.names.push_back(it == s.names.end() ? "unregistered" : it->second);
    k.N.push_back(selected_n ? selected_n : seg.L->N); k.storage_N.push_back(seg.L->N); k.bits.push_back(uint32_t(seg.L->bits));
    k.y_ld.push_back(seg.y_ld ? seg.y_ld : seg.L->N);
    if constexpr (requires { seg.add; }) k.add.push_back(seg.add ? 1u : 0u);
    else k.add.push_back(0);
  }
  ++s.calls[k];
}
inline void append(const char* destination, const nlohmann::json& j) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (FILE* f = fopen(destination, "a")) { fprintf(f, "%s\n", j.dump().c_str()); fclose(f); }
  else { perror(destination); std::abort(); }
}
inline void finish(nlohmann::json j) {
  if (!enabled()) return;
  auto& s = state();
  j["trace_kind"] = "host-metadata-and-natural-command-buffer-times";
  j["timing_for_acceptance"] = false;
  j["call_count_kind"] = "host-selected encoded dispatches, not shader-internal counters";
  j["cycle_time_kind"] = "completed natural command buffers after existing wait";
  j["calls"] = nlohmann::json::array();
  for (const auto& [k, count] : s.calls) {
    j["calls"].push_back({{"request_stage",k.stage},{"role",k.role},{"phase",k.phase},{"kernel",k.kernel},{"projection",k.names},
      {"N",k.N},{"storage_N",k.storage_N},{"K",k.K},{"R_capacity",k.R},{"segments",k.N.size()},{"bits",k.bits},{"y_ld",k.y_ld},{"add",k.add},
      {"NF",k.nf},{"CS",k.cs},{"KS",k.ks},{"threadgroups",k.tg},{"threads",k.threads},{"packed",k.packed},{"calls",count}});
  }
  const auto& xp = s.canonical_xp;
  j["canonical_xp"] = {{"scope", "immutable target-layer prefill owners only; encoded dispatch counters"},
      {"plain_ops", xp.plain}, {"GU_ops", xp.gu}, {"XP_pack_dispatches", xp.xp_packs},
      {"W_halfSB_producers", xp.wsb_producers}, {"XP_GEMM_dispatches", xp.gemms},
      {"raw_C1_fallback_weight_unpacks", xp.raw_fallback_unpacks},
      {"serialized_projection_groups", xp.serialized_groups}, {"serialized_segments", xp.serialized_segments},
      {"lost_grouped_overlap_sets", xp.lost_grouped_overlap_sets}};
  j["cycles"] = nlohmann::json::array();
  for (const auto& c : s.cycles) j["cycles"].push_back({{"cycle",c.cycle},{"position",c.position},
    {"accepted_drafts",c.accepted_drafts},{"emitted",c.emitted},{"verified_rows",c.verified_rows},
    {"draft_including_commit_gpu_ms",c.draft_end_ms-c.draft_start_ms},
    {"verify_including_acceptance_compact_gpu_ms",c.verify_end_ms-c.verify_start_ms},
    {"draft_start_ms",c.draft_start_ms},{"draft_end_ms",c.draft_end_ms},
    {"verify_start_ms",c.verify_start_ms},{"verify_end_ms",c.verify_end_ms}});
  j["final_flush_gpu_ms"] = s.final_flush_ms;
  j["terminal_extra_draft_gpu_ms"] = s.terminal_extra_draft_ms;
  j["geometry_note"] = "NF is per matrix for fused gate/up; NF=0 marks a non-qmm8 fragment geometry. Sparse rescore N is selected rows per token, storage_N is the full reduced head. threadgroups is total grid product for rows kernels.";
  j["commit_gpu_ms"] = nullptr;
  j["commit_note"] = "Folded into natural drafting command buffer; not isolated because that would perturb scheduling.";
  append(path(), j); s.active = false;
}
}  // namespace ni::projection_trace
