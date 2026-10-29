// Opt-in, real Metal/model failure test. Run under the shared GPU lock, as sdk_model_test.
// Test-only objects select a missing kernel at real creation sites; no production fault flags.
#include "kapitsa.h"
#include "engine/engine.h"
#include "bench/bench.h"
#include "util/log.h"
#include <atomic>
#include <barrier>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

using Json = ni::ojson;
using Engine = std::unique_ptr<kapitsa_engine, decltype(&kapitsa_engine_destroy)>;
using Request = std::unique_ptr<kapitsa_request, decltype(&kapitsa_request_destroy)>;
using String = std::unique_ptr<char, decltype(&kapitsa_string_free)>;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
namespace {
std::atomic<bool> armed{false}, in_batch{false}, in_leave{false}, want_queued{false};
std::atomic<int> injected{0};
ni::Buffer submitted, discarded;
ni::Pipeline marker;
}
namespace ni {
void pipeline_test_batch() {
  if (in_batch.exchange(false)) { armed = true; Gpu::get().pipeline("sdk_batch_late_pipeline"); }
}
void pipeline_test_queued() {
  if (want_queued.exchange(false)) { armed = true; Gpu::get().pipeline("sdk_queued_late_pipeline"); }
}
void pipeline_test_leave() {
  if (in_leave.exchange(false)) { armed = true; Gpu::get().pipeline("sdk_leaving_late_pipeline"); }
}
std::string pipeline_test_name(const std::string& name) {
  if (!armed.exchange(false)) return name;
  ++injected;
  // A submitted prefix and an uncommitted tail: recovery must wait for the first
  // and discard the second, without freeing either buffer prematurely.
  if (marker.pso) {
    auto& g = Gpu::get();
    auto e = g.begin();
    e->use(marker); e->buf(0, submitted); e->dispatch(1, 1, 1, 1); e->split();
    e->use(marker); e->buf(0, discarded); e->dispatch(1, 1, 1, 1);
  }
  return "sdk_deferred_missing_kernel";
}
}
namespace {
Engine create(const Json& options) {
  kapitsa_engine* raw = nullptr; char* error = nullptr;
  const auto status = kapitsa_engine_create(options.dump().c_str(), &raw, &error);
  String message(error, kapitsa_string_free);
  if (message) std::cerr << message.get() << '\n';
  Engine result(raw, kapitsa_engine_destroy);
  CHECK(status == KAPITSA_OK && result && !message);
  return result;
}
Request request(kapitsa_engine* e, int id, bool cacheable = false) {
  std::string prompt;
  if (cacheable) for (int i = 0; i < 20; ++i) prompt += "The sky is blue. ";
  prompt += "List many facts about the number " + std::to_string(id) + ".";
  Json json{{"messages", Json::array({{{"role", "user"},
             {"content", prompt}}})},
            {"max_tokens", 128}, {"temperature", 0}, {"template_kwargs", {{"enable_thinking", false}}}};
  kapitsa_request* raw = nullptr; char* error = nullptr;
  auto status = kapitsa_request_create(e, json.dump().c_str(), &raw, &error);
  String message(error, kapitsa_string_free);
  Request result(raw, kapitsa_request_destroy);
  CHECK(status == KAPITSA_OK && result && !message);
  return result;
}
std::string generate(kapitsa_request* r, kapitsa_status expected, kapitsa_delta_callback cb = nullptr, void* user = nullptr) {
  char* output = nullptr; char* error = nullptr;
  auto status = kapitsa_generate(r, cb, user, &output, &error);
  String result(output, kapitsa_string_free), message(error, kapitsa_string_free);
  CHECK(status == expected);
  if (expected == KAPITSA_ERROR) {
    CHECK(message && !result);
    return message.get();
  }
  CHECK(result && !message && Json::parse(result.get())["completion_tokens"] > 0);
  return result.get();
}
bool lock_available(const std::string& path) {
  int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
  CHECK(fd >= 0);
  const bool ok = flock(fd, LOCK_EX | LOCK_NB) == 0;
  close(fd); return ok;
}
size_t entries(const std::string& dir) {
  size_t n = 0;
  for (const auto& f : std::filesystem::recursive_directory_iterator(dir))
    n += f.is_regular_file() && f.path().parent_path().filename() == "e";
  return n;
}
void check_failure(const std::string& message) {
  CHECK(message.find("sdk_deferred_missing_kernel") != std::string::npos || message.find("unusable") != std::string::npos);
  CHECK(submitted.as<int>()[0] == 17 && discarded.as<int>()[0] == 0);
}
void small_gpu() {
  auto& g = ni::Gpu::get();
  ni::Gpu::PipelineRecovery recover;
  try { g.compile_extra("deferred-test", "invalid Metal"); CHECK(false); }
  catch (const ni::FatalError& e) { CHECK(std::string(e.what()).find("Metal compile failed") != std::string::npos); }
  CHECK(recover.failed && !ni::FatalRecovery::enabled());
  g.compile_extra("deferred-test", R"(#include <metal_stdlib>
    kernel void sdk_deferred_mark(device int* dst [[buffer(0)]]) { dst[0] = 17; })");
  marker = g.pipeline("sdk_deferred_mark");
  CHECK(!g.pipeline("sdk_optional_missing", false).pso);
  auto missing = marker; missing.name = "sdk_missing_specialization";
  try { g.pipeline_threads(missing, 32); CHECK(false); }
  catch (const ni::FatalError& e) { CHECK(std::string(e.what()).find("sdk_missing_specialization") != std::string::npos); }
  auto b = g.alloc(16384);
  auto e = g.begin(); e->use(marker); e->buf(0, b); e->dispatch(1, 1, 1, 1); e->commit_and_wait();
  CHECK(b.as<int>()[0] == 17); g.release(b);
  std::cout << "PASS deferred compilation, optional pipeline, specialization failure, valid dispatch\n";
}
}
int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: sdk_deferred_test --small-gpu | --fatal-gpu | MODEL_DIR\n"; return 2; }
  alarm(300);
  try {
    ni::bench::GpuLock global;
    CHECK(ni::bench::GpuLock::held_by_caller() || global.acquire());
    if (std::string(argv[1]) == "--fatal-gpu") { ni::Gpu::get().pipeline("sdk_deferred_missing_kernel"); return 2; }
    if (std::string(argv[1]) == "--small-gpu") { small_gpu(); return 0; }
    // Take the common lock above, then exercise SDK ownership on a separate test lock.
    char path[] = "/tmp/kapitsa-deferred-XXXXXX"; CHECK(mkdtemp(path));
    const std::string dir = path, lock = dir + "/gpu.lock";
    setenv("KAPITSA_HOME", dir.c_str(), 1);  // calibration/cache files belong to this test, not the user's home
    setenv("NI_DISK_CACHE_DIR", (dir + "/prefix-cache").c_str(), 1);
    setenv("NI_DISK_CACHE_MIN", "64", 1); setenv("NI_DISK_CACHE_IDLE_MS", "3600000", 1);
    setenv("BENCH_GPU_LOCK", lock.c_str(), 1);
    unsetenv("NI_LOCK_HELD"); unsetenv("NACE_GPU_LOCK_HELD");
    setenv("NI_PREWARM", "0", 1); setenv("NI_ANE", "0", 1);
    setenv("NI_PARK_TEST", "1", 1); setenv("NI_BATCH", "1", 1);
    small_gpu();
    auto& g = ni::Gpu::get();
    submitted = g.alloc(16384); discarded = g.alloc(16384);
    Json options{{"model_dir", argv[1]}, {"context", 1024}, {"prefill_chunk", 128},
                 {"slots", 2}, {"dflash", "none"}, {"spec_depth", 0}, {"disk_cache", "0"}};
    for (int mode = 0; mode < 4; ++mode) {
      options["dflash"] = mode >= 2 ? "auto" : "none";
      options["spec_depth"] = mode >= 2 ? -1 : 0;
      options["disk_cache"] = mode == 1 ? "1G" : "0";
      submitted.as<int>()[0] = discarded.as<int>()[0] = 0;
      injected = 0;
      auto engine = create(options);
      if (mode == 0) {
        armed = true;
        auto r = request(engine.get(), 7);
        check_failure(generate(r.get(), KAPITSA_ERROR));
      } else {
        auto a = request(engine.get(), 11, mode == 1), b = request(engine.get(), 13, mode == 1),
             c = request(engine.get(), 17);
        std::barrier start(3);
        std::atomic<int> first_deltas{0};
        struct Stream { std::atomic<int>* seen; bool first = true; int mode; };
        auto cb = [](const char*, size_t, void* p) -> int {
          auto& s = *static_cast<Stream*>(p);
          if (s.first) {
            s.first = false;
            if (++*s.seen == 2) {
              if (s.mode == 2) in_batch = true;
              else if (s.mode == 3) in_leave = true;
              else want_queued = true;
            }
          }
          return 1;
        };
        Stream sa{&first_deltas, true, mode}, sb{&first_deltas, true, mode};
        auto fa = std::async(std::launch::async, [&] { start.arrive_and_wait(); return generate(a.get(), KAPITSA_ERROR, cb, &sa); });
        auto fb = std::async(std::launch::async, [&] { start.arrive_and_wait(); return generate(b.get(), KAPITSA_ERROR, cb, &sb); });
        start.arrive_and_wait();
        std::future<std::string> fc;
        if (mode == 1) {
          while (first_deltas < 2) std::this_thread::yield();
          fc = std::async(std::launch::async, [&] { return generate(c.get(), KAPITSA_ERROR); });
        }
        check_failure(fa.get()); check_failure(fb.get());
        CHECK(first_deltas == 2);
        check_failure(mode == 1 ? fc.get() : generate(c.get(), KAPITSA_ERROR));
        CHECK(!want_queued);
        CHECK(!in_batch);  // mode 2 must have entered a real batch cycle
        CHECK(!in_leave);  // mode 3 failed while committing a member's staged rows
      }
      CHECK(injected == 1);
      auto later = request(engine.get(), 19);
      CHECK(generate(later.get(), KAPITSA_ERROR).find("unusable") != std::string::npos);
      engine.reset(); CHECK(!lock_available(lock));  // failed requests retain the engine and lock
      later.reset(); CHECK(lock_available(lock));
      if (mode == 1) {
        CHECK(std::filesystem::is_directory(dir + "/prefix-cache") && entries(dir) == 0);
      }
      std::cout << "PASS deferred error case " << mode << ": discard/drain, request failure, retained lock\n";
    }
    options["dflash"] = "none";
    options["spec_depth"] = 0;
    options["disk_cache"] = "1G";
    { auto engine = create(options); auto r = request(engine.get(), 23, true); generate(r.get(), KAPITSA_OK); }
    CHECK(lock_available(lock));
    CHECK(entries(dir) > 0);  // positive control: healthy state is saved by the same cache configuration
    g.release(submitted); g.release(discarded);
    std::filesystem::remove_all(dir);
    std::cout << "PASS destroy failed engine, reload and generate in the same process\n";
  } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
  alarm(0);
}
