// CPU fixtures by default. --gpu also checks the lookup kernels against a CPU suffix search, with guards,
// both host/GPU anchor positions and the completed-cycle flag surviving the next lookup. No model is loaded.
// Run the GPU cases with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1; the global GPU lock is held by this tool.
#include "bench/bench.h"
#include "kernels/kflash.h"
#include "metal/gpu.h"
#include "model/kparams.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace ni;

namespace {
struct Case {
  std::vector<int32_t> tokens;
  kp::DflLookup p;
  uint32_t live = 8;
  int want_len = -1, want_end = -1;
};

// Independent search order: match lengths from longest to shortest, endpoints from newest to oldest.
std::pair<int, int> lookup(const Case& c) {
  const int P = int(c.p.P), nmin = int(c.p.min_match);
  if (nmin < 4 || nmin > 32 || c.p.cap <= 8 || c.p.P >= c.p.cap - 8 || P < nmin + 6 || c.live != 8)
    return {0, -1};
  const int first = std::max(nmin - 1, P - int(c.p.scan));
  for (int n = std::min(32, P + 1); n >= nmin; --n)
    for (int end = P - 7; end >= std::max(first, n - 1); --end)
      if (std::equal(c.tokens.begin() + P - n + 1, c.tokens.begin() + P + 1, c.tokens.begin() + end - n + 1))
        return {n, end};
  return {0, -1};
}

Case unique_case(int P, int nmin = 8, int scan = 32768) {
  Case c;
  c.p = {uint32_t(P), uint32_t(nmin), uint32_t(scan), uint32_t(P + 16)};
  c.tokens.resize(c.p.cap);
  for (size_t i = 0; i < c.tokens.size(); ++i) c.tokens[i] = int32_t(i + 100);
  return c;
}

std::vector<Case> cases() {
  std::vector<Case> out;
  for (int P : {0, 1, 7, 8, 14, 15, 31, 32, 127, 512}) {
    auto c = unique_case(P);
    c.want_len = 0;
    out.push_back(c);
  }
  auto repeated = unique_case(79);
  std::copy_n(repeated.tokens.begin(), 32, repeated.tokens.begin() + 48);
  repeated.want_len = 32;
  repeated.want_end = 31;
  out.push_back(repeated);
  auto tie = unique_case(159);
  std::copy_n(tie.tokens.begin(), 32, tie.tokens.begin() + 48);
  std::copy_n(tie.tokens.begin(), 32, tie.tokens.begin() + 128);
  tie.want_len = 32;
  tie.want_end = 79;
  out.push_back(tie);
  auto overlap = unique_case(95);
  std::fill(overlap.tokens.begin(), overlap.tokens.end(), 42);
  overlap.want_len = 32;
  overlap.want_end = 88;
  out.push_back(overlap);
  for (int scan : {95, 96}) {
    auto c = unique_case(127, 8, scan);
    std::copy_n(c.tokens.begin(), 32, c.tokens.begin() + 96);
    c.want_len = scan == 96 ? 32 : 0;
    c.want_end = scan == 96 ? 31 : -1;
    out.push_back(c);
  }
  for (int kind = 0; kind < 5; ++kind) {
    auto c = repeated;
    c.want_len = 0;
    c.want_end = -1;
    if (kind == 0) c.p.min_match = 0;
    if (kind == 1) c.p.min_match = 33;
    if (kind == 2) c.live = 4;
    if (kind == 3) c.p.cap = 8;
    if (kind == 4) c.p.cap = c.p.P + 8;
    out.push_back(c);
  }
  std::mt19937 rng(20261028);
  for (int P : {31, 64, 128, 511, 1024, 4096, 32767, 32800})
    for (int nmin : {4, 8, 12, 24, 32})
      for (int mode = 0; mode < 4; ++mode) {
        auto c = unique_case(P, nmin);
        for (auto& v : c.tokens) v = int32_t(rng() % (mode == 3 ? 3 : 248320));
        if (mode == 1 || mode == 2) {
          const int n = std::min(nmin + (mode == 2 ? 7 : 0), P / 3);
          std::copy_n(c.tokens.begin(), n, c.tokens.begin() + P - n + 1);
        }
        out.push_back(std::move(c));
      }
  return out;
}

constexpr int kGuard = 16;
constexpr int32_t kPoison = 0x5a17ca11;

struct Guarded {
  Buffer all, body;
  std::vector<int32_t> expected;
  Guarded(Gpu& g, const std::vector<int32_t>& data, const char* name) {
    expected.assign(data.size() + 2 * kGuard, kPoison);
    std::copy(data.begin(), data.end(), expected.begin() + kGuard);
    all = g.alloc(expected.size() * 4, name);
    body = all.view(kGuard * 4, data.size() * 4);
    std::memcpy(all.ptr(), expected.data(), expected.size() * 4);
  }
  int32_t* exp() { return expected.data() + kGuard; }
  bool check(const char* name, int id, uint64_t& bytes) const {
    bytes += expected.size() * 4;
    const int32_t* got = all.as<int32_t>();
    for (size_t i = 0; i < expected.size(); ++i)
      if (got[i] != expected[i]) {
        std::fprintf(stderr, "case %d %s word %zu: got %08x expected %08x\n", id, name, i,
                     uint32_t(got[i]), uint32_t(expected[i]));
        return false;
      }
    return true;
  }
};

void expected_lookup(Guarded& tok, Guarded& tree, Guarded& live, const Case& c, std::pair<int, int> match) {
  live.exp()[1] = match.first;
  if (!match.first) return;
  for (int r = 0; r < 8; ++r) {
    if (r) tok.exp()[c.p.P + r] = c.tokens[match.second + r];
    tree.exp()[kf::kPar + r] = r - 1;
    tree.exp()[kf::kDep + r] = r;
    tree.exp()[kf::kMask + r] = (1 << (r + 1)) - 1;
    tree.exp()[kf::kLq + r] = 0;
    tree.exp()[kf::kScore + r] = 0;
  }
  for (int n = 0; n <= kf::kMaxRows; ++n)
    tree.exp()[kf::kE + n] = std::bit_cast<int32_t>(n <= 8 ? float(n) : 0.0f);
  tree.exp()[kf::kN] = tree.exp()[kf::kNodes] = tree.exp()[kf::kCap] = 8;
  tree.exp()[kf::kJump] = 0;
}
}  // namespace

int main(int argc, char** argv) {
  const bool gpu = argc == 2 && std::string(argv[1]) == "--gpu";
  if (argc > 1 && !gpu) {
    std::fprintf(stderr, "usage: dflash_lookup_check [--gpu]\n");
    return 1;
  }
  static_assert(sizeof(kp::DflLookup) == 16);
  auto all = cases();
  int failures = 0, known = 0;
  for (const auto& c : all) {
    const auto match = lookup(c);
    if (c.want_len >= 0) {
      ++known;
      if (match != std::make_pair(c.want_len, c.want_end)) ++failures;
    }
    if (match.first) {
      if (match.second + 7 > int(c.p.P) || match.first < int(c.p.min_match)) ++failures;
      for (int j = 0; j < match.first; ++j)
        if (c.tokens[c.p.P - j] != c.tokens[match.second - j]) ++failures;
    }
  }
  std::printf("CPU suffix oracle: %zu cases, %d explicit expected results, %d failures\n", all.size(), known, failures);
  if (failures || !gpu) return failures ? 1 : 0;
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire([](const std::string& who) {
        std::fprintf(stderr, "waiting for GPU lock: %s\n", who.c_str());
      })) {
    std::fprintf(stderr, "cannot acquire GPU lock: %s\n", std::strerror(errno));
    return 1;
  }
  Gpu& g = Gpu::get();
  const Pipeline select = g.pipeline("dfl_lookup_tree"), record = g.pipeline("dfl_lookup_record");
  int id = 0;
  uint64_t bytes = 0;
  for (const auto& c : all)
    for (bool gpu_anchor : {false, true}) {
      Guarded tok(g, c.tokens, "check_lookup_tokens");
      Guarded tree(g, std::vector<int32_t>(kf::kWords, 0x12345678), "check_lookup_tree");
      Guarded live(g, {int32_t(c.live), 13, 29, 47}, "check_lookup_live");
      Guarded gst(g, {int32_t(c.p.P), 0, 0, 0}, "check_lookup_gst");
      const auto match = lookup(c);
      auto p = c.p;
      if (gpu_anchor) p.P = 0xffffffffu;
      auto e = g.begin();
      e->use(select);
      e->buf(0, tok.body); e->buf(1, tree.body); e->buf(2, live.body); e->val(3, p); e->buf(4, gst.body);
      e->dispatch(1, 1, 1, 128);
      e->commit_and_wait();
      expected_lookup(tok, tree, live, c, match);
      failures += !tok.check("tokens", id, bytes);
      failures += !tree.check("tree", id, bytes);
      failures += !live.check("live before record", id, bytes);
      failures += !gst.check("GPU anchor", id, bytes);
      // The next (disabled/sampled) drafting half must not erase the completed cycle's observation.
      e = g.begin();
      e->use(record); e->buf(0, live.body); e->dispatch(1, 1, 1, 32);
      p.min_match = 0;
      e->use(select);
      e->buf(0, tok.body); e->buf(1, tree.body); e->buf(2, live.body); e->val(3, p); e->buf(4, gst.body);
      e->dispatch(1, 1, 1, 128);
      e->commit_and_wait();
      live.exp()[2] = match.first;
      live.exp()[1] = 0;
      failures += !live.check("live after next draft", id, bytes);
      failures += !tok.check("tokens after next draft", id, bytes);
      failures += !tree.check("tree after next draft", id, bytes);
      failures += !gst.check("GPU anchor after next draft", id, bytes);
      ++id;
    }
  std::printf("GPU lookup: %d cases, %llu checked data/guard bytes, %d failures\n", id,
              (unsigned long long)bytes, failures);
  return failures ? 1 : 0;
}
