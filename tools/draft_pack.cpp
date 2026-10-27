// Save the existing CPU affine quantizer's output once, into a new drafter folder.
//   draft_pack SOURCE NEW_DIRECTORY [4|8] [away|even]
// Reads checkpoint bytes on the CPU under the GPU/model lock. Does not initialize Metal.
#include "bench/bench.h"
#include "loader/safetensors.h"
#include "model/dflash2.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ni;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::U32: return "U32";
    case DType::I32: return "I32";
    case DType::U8: return "U8";
    case DType::I8: return "I8";
    case DType::I64: return "I64";
    case DType::F8E4M3: return "F8_E4M3";
    default: throw std::runtime_error("unrecognized source dtype");
  }
}

struct Job {
  std::string name;
  const SafetensorsIndex::Entry* source = nullptr;
  bool quantize = false;
  size_t N = 0, K = 0;
};

int pack(const fs::path& source, const fs::path& dest, int bits, bool even) {
  if (fegetround() != FE_TONEAREST) throw std::runtime_error("requires round-to-nearest floating-point mode");
  if (fs::exists(dest)) throw std::runtime_error("output directory already exists; choose a new directory");
  std::ifstream cf(source / "config.json");
  json config = json::parse(cf);
  const json arch = config.value("architectures", json::array());
  if ((arch != json::array({"DFlashDraftModel"}) && arch != json::array({"DFlash2DraftModel"})) ||
      !config.contains("dflash_config") || config.contains("kapitsa_draft_pack"))
    throw std::runtime_error("source must be an unpacked DFlash/DFlash2 checkpoint");

  SafetensorsIndex index(source.string());
  std::map<std::string, const SafetensorsIndex::Entry*> ordered;
  for (const auto& [name, entry] : index.all()) ordered[name] = &entry;
  std::vector<Job> jobs;
  json header = {{"__metadata__", {{"format", "pt"}, {"kapitsa_draft_pack", "1"}}}};
  uint64_t bytes = 0, source_bytes = 0;
  size_t projections = 0;
  auto add = [&](const std::string& name, const char* dtype, const std::vector<int64_t>& shape, uint64_t size) {
    if (header.contains(name)) throw std::runtime_error("duplicate output tensor: " + name);
    header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {bytes, bytes + size}}};
    bytes += size;
  };
  for (const auto& [name, p] : ordered) {
    Job job{name, p};
    size_t expected = dtype_size(p->dtype);
    if (!expected) throw std::runtime_error("unsupported source dtype: " + name);
    for (int64_t dim : p->shape) {
      if (dim < 0 || (dim && expected > std::numeric_limits<size_t>::max() / uint64_t(dim)))
        throw std::runtime_error("source tensor shape overflows: " + name);
      expected *= size_t(dim);
    }
    const uint64_t file_size = fs::file_size(index.path(*p));
    if (p->len != expected || p->off > file_size || p->len > file_size - p->off)
      throw std::runtime_error("source tensor byte extent does not match its shape: " + name);
    source_bytes += p->len;
    job.quantize = name.ends_with(".weight") && p->shape.size() == 2;
    if (job.quantize) {
      if (p->dtype != DType::BF16 || p->shape[0] < 1 || p->shape[1] < 1 ||
          p->shape[0] % 8 || p->shape[1] % 64)
        throw std::runtime_error("projection must be bf16 with N % 8 = K % 64 = 0: " + name);
      job.N = size_t(p->shape[0]);
      job.K = size_t(p->shape[1]);
      if (job.N > UINT32_MAX || job.K > UINT32_MAX / 8)
        throw std::runtime_error("projection dimensions exceed the native quantizer's range");
      const std::string prefix = name.substr(0, name.size() - 7);
      add(name, "U32", {int64_t(job.N), int64_t(job.K * bits / 32)}, job.N * job.K * bits / 8);
      add(prefix + ".scales", "BF16", {int64_t(job.N), int64_t(job.K / 64)}, job.N * job.K / 64 * 2);
      add(prefix + ".biases", "BF16", {int64_t(job.N), int64_t(job.K / 64)}, job.N * job.K / 64 * 2);
      ++projections;
    } else {
      add(name, dtype_name(p->dtype), p->shape, p->len);
    }
    jobs.push_back(job);
  }
  if (!projections) throw std::runtime_error("no projection weights in source");
  if (!fs::create_directory(dest)) throw std::runtime_error("could not create the new output directory");
  // config.json appears last, so an interrupted conversion cannot be found as a loadable drafter.
  const fs::path partial = dest / "model.safetensors.partial";
  std::ofstream output(partial, std::ios::binary);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  std::string encoded = header.dump();
  encoded.append((8 - encoded.size() % 8) % 8, ' ');
  const uint64_t header_size = encoded.size();
  output.write(reinterpret_cast<const char*>(&header_size), 8);
  output.write(encoded.data(), std::streamsize(encoded.size()));
  auto write = [&](const auto& v) {
    output.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(v[0])));
  };
  for (const Job& job : jobs) {
    if (job.quantize) {
      std::vector<uint16_t> raw(job.N * job.K);
      index.read(*job.source, 0, raw.data(), job.source->len);
      for (uint16_t value : raw)
        if ((value & 0x7f80) == 0x7f80) throw std::runtime_error("non-finite source projection: " + job.name);
      std::vector<const uint16_t*> rows(job.N);
      for (size_t r = 0; r < job.N; ++r) rows[r] = raw.data() + r * job.K;
      std::vector<uint32_t> w(job.N * job.K * bits / 32);
      std::vector<uint16_t> s(job.N * job.K / 64), b(s.size());
      dfl_quantize_cpu(rows, uint32_t(job.K), bits, even, w.data(), s.data(), b.data());
      for (const auto* values : {&s, &b})
        for (uint16_t value : *values)
          if ((value & 0x7f80) == 0x7f80)
            throw std::runtime_error("non-finite quantized scale/bias: " + job.name);
      write(w);
      write(s);
      write(b);
    } else {
      std::vector<uint8_t> chunk(std::min<size_t>(job.source->len, 4u << 20));
      for (size_t off = 0; off < job.source->len; off += chunk.size()) {
        const size_t size = std::min(chunk.size(), job.source->len - off);
        index.read(*job.source, off, chunk.data(), size);
        output.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(size));
      }
    }
  }
  output.close();
  if (fs::file_size(partial) != 8 + encoded.size() + bytes) throw std::runtime_error("output size mismatch");
  fs::rename(partial, dest / "model.safetensors");
  for (const char* name : {"LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "README.md"})
    if (fs::is_regular_file(source / name)) fs::copy_file(source / name, dest / name);
  config["kapitsa_draft_pack"] = {{"version", 1}, {"bits", bits}, {"group_size", 64},
                                  {"round", even ? "even" : "away"}};
  std::ofstream packed_config(dest / "config.json.partial");
  packed_config.exceptions(std::ios::badbit | std::ios::failbit);
  packed_config << config.dump(2) << '\n';
  packed_config.close();
  fs::rename(dest / "config.json.partial", dest / "config.json");
  std::printf("{\"projections\":%zu,\"source_bytes\":%llu,\"packed_bytes\":%llu,\"bits\":%d,\"round\":\"%s\"}\n",
              projections, (unsigned long long)source_bytes, (unsigned long long)bytes, bits, even ? "even" : "away");
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr, "usage: draft_pack SOURCE NEW_DIRECTORY [4|8] [away|even]\n");
    return 2;
  }
  const std::string bit_text = argc >= 4 ? argv[3] : "4", round = argc >= 5 ? argv[4] : "away";
  if ((bit_text != "4" && bit_text != "8") || (round != "away" && round != "even")) {
    std::fprintf(stderr, "bits must be 4 or 8; rounding must be away or even\n");
    return 2;
  }
  bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD")) {
    if (!lock.acquire([](const std::string&) { std::fprintf(stderr, "waiting for the GPU/model lock\n"); }))
      fatal("cannot take the GPU/model lock: %s", strerror(errno));
  }
  try {
    return pack(argv[1], argv[2], bit_text == "4" ? 4 : 8, round == "even");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "draft_pack: %s\n", e.what());
    return 1;
  }
}
