# Validation

Native code tested: 8ea1723, against main 4256a5119bd093f6088bd46943aaacf5cc60d48a. Later evidence/tool-only changes do not alter src/ or CMake.

Runs: Apple M4 Pro (20 GPU cores), 48 GB, macOS 15.7.1 (24G231), Qwen3.8-27B-4bit, AC power. Default Metal math; unchanged numerical floor.

`decode_reduce_test --gpu --small` under Metal shader/API validation, then `--gpu`: all 90 reference/guard cases passed, including empty splits, exponential underflow and an FP64 reference.

Default output check: all 584 rows bitwise identical. Synthetic 64K stress check: all 17 rows bitwise identical. `logit-compare.txt` and `long-compare.txt` contain the complete reports. The 64K input repeats the first default corpus window's token IDs; it is a correctness stress case, not a throughput workload.

All 27 CPU check groups passed on the first run. The installed model tokenizer's grammar/mask test and final prek also passed.

API: candidate and main both passed 48/49. The same keepalive fixture fails: holder requests stop after two tokens, freeing slots before its expected 15-second keepalive. Full candidate/main logs are in the [archived evidence](https://github.com/chainhelen/kapitsa/tree/5554eb32398076dddbecd6b2193b57eefb56b5aa/bench/results/2026-10-29-m4pro-operators/reduce/model-validation).

## Standard request benchmark

Produced by unmodified `kapitsa bench --quick --json`, A/B/A/B, with a fresh KAPITSA_HOME per run. The benchmark filters NI_DFLASH/NI_NO_MTP, so all these runs use default KFlash/DFlash2. The [archived benchmark JSONs](https://github.com/chainhelen/kapitsa/tree/5554eb32398076dddbecd6b2193b57eefb56b5aa/bench/results/2026-10-29-m4pro-operators/reduce/model-validation) include all individual samples, hardware, version, flags and conditions. All were valid, tools passed, swapouts were zero, and contention lists were empty. ChatGPT was retained; its detected GPU activity is shown. The standard legs below are all at fair thermal state; no reproducible whole-request speedup is claimed.

| Run | Decode T=0.6 tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Thermal | Other GPU mean % |
|---|---:|---:|---:|---:|---|---:|
| A1 | 66.82 | 65.67 | 10.4974 | 0.2306 | fair | 0.0 |
| B1 | 63.52 | 65.67 | 10.5542 | 0.2307 | fair | 0.0 |
| A2 | 58.35 | 64.24 | 10.5444 | 0.2321 | fair | 0.0 |
| B2 | 58.53 | 65.73 | 10.5402 | 0.2296 | fair | 0.0 |

## Ordinary decode supplement

`tools/bench_plain_decode.py` starts the real HTTP server with `--spec 0 --slots 1 --context 8192`, holds the GPU lock, warms up, and measures one 2957-token prompt / 512-token greedy answer per leg. All four outputs were nonempty, ended at the length limit, had identical SHA-256, and reported draft source `none`. Throughput does not show a reproducible request-level gain.

The first attempt is retained in the archive at [plain-v1](https://github.com/chainhelen/kapitsa/tree/5554eb32398076dddbecd6b2193b57eefb56b5aa/bench/results/2026-10-29-m4pro-operators/reduce/model-validation/plain-v1): B2 compiled its ANE program asynchronously for ~35 s, contaminating startup/prefill timing. The tool now waits for ANE readiness; plain-v2 is the rerun. Its four JSON records remain here; prompts and generated text are in the [archive](https://github.com/chainhelen/kapitsa/tree/5554eb32398076dddbecd6b2193b57eefb56b5aa/bench/results/2026-10-29-m4pro-operators/reduce/model-validation/plain-v2). Thermal values below use NSProcessInfo: 0 nominal, 1 fair. B2 warms to fair, so sub-percent throughput differences must not be interpreted as a speedup.

| Run | Decode tok/s | Prefill ms | Thermal before → after |
|---|---:|---:|---|
| A1 | 15.4101 | 7736.4 | 0 → 0 |
| B1 | 15.3926 | 7744.9 | 0 → 0 |
| A2 | 15.3794 | 7752.9 | 0 → 0 |
| B2 | 15.2603 | 7730.5 | 0 → 1 |

Reproduce (OUT must not exist):

```sh
curl -fL https://raw.githubusercontent.com/chainhelen/kapitsa/5554eb32398076dddbecd6b2193b57eefb56b5aa/bench/results/2026-10-29-m4pro-operators/reduce/model-validation/plain-v2/prompt.txt -o /tmp/kapitsa-reduce-prompt.txt
python3 tools/bench_plain_decode.py --reference /path/to/main/build/kapitsa --candidate build/kapitsa --model /path/to/qwen3.8-27b-4bit --prompt /tmp/kapitsa-reduce-prompt.txt --out /tmp/kapitsa-plain-new
```

## Not verified

Qwen3.5-4B and Qwen3.6-35B-A3B/MoE (not installed), other chips/macOS versions, and real multimodal end-to-end inputs. Named build targets were used as required on the older SDK; M5/Metal-4-only tools were not built.
