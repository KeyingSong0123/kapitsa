# ANE segmented input within the existing 8192-channel rotation limit

The PR makes supported dense Qwen hidden sizes **2048, 4096, 6144 and 8192** eligible for ANE prefill by default. The 9B uses two 2048-channel inputs. Existing 4B/27B hidden sizes 2560/5120 prefer 2560-channel segments and keep their rotation geometry. `NI_ANE=0` disables the split; hardware, format, intermediate-size, memory, chunk, startup-check and fallback conditions remain.

**Default 9B outputs intentionally change relative to main because this uses the existing W8A8 approximation. This is not a lossless optimization or a claim of passing a 9B noise floor.** The quantization, 128-channel Hadamard blocks, token-wide scale, down segmentation and wavefront schedule are unchanged.

The unused `ane_rotate_wide` is removed. `src/kernels/ane_ffn.metal` is byte-for-byte main's file; every accepted width uses the existing `ane_rotate`. Input selection rejects hidden > 8192, rotation keeps the original at-most-eight-blocks check, and unsupported wide inputs reserve no split memory. The SSD prefix marker now follows the same input selector, so rejected wide models do not get the new numerical-mode namespace. Tests retain the six supported GPU widths and assert rejection/no-dispatch/no-budget for wider widths, including 10240 (also divisible by 2560) and `UINT32_MAX`.

## Measured source

Runs: MacBook Pro (14-inch, 2024), M4 Pro / 20 GPU cores / 48 GiB, macOS 26.4.1 (25E253), AC power for every timed leg. Main A: `16ead4cf201acd8f321090db7c0c3aacbe7de95f`. Current B: `93b2bd7c078ec37f767b1ee6753683811db8045c`. Previous PR head: `fac53c043ea5bcca7e3a40330ef40ca3c96e3dc8`. The current head contains main. Kapitsa version: 0.2.24; all three binary versions and hashes are recorded.

Published source: `a6e2d3138d0c48dea085fe6815a9357c063f115e`. The only follow-up to the measured `93b2bd7c` is one stale header comment. A Release rebuild retaining the recorded build ID reproduced all 27 tested executables byte for byte (including `kapitsa`, `logit_dump` and `ane_segment_check`); the hashes and follow-up checks are in `final-source-validation.json`. Printed runtime versions below deliberately retain the actual measured ID.

Models: lmstudio-community/Qwen3.5-4B-MLX-4bit at `c43ee1d65576a5d98de1e8405cac93c371a655c1`, lmstudio-community/Qwen3.5-9B-MLX-4bit at `b455506b0f574c74616dbcd56879bde38fafcff3`, mlx-community/Qwen3.8-27B-4bit at `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`; 4-bit group-64. Tokenizer hashes match. Every model/GPU command ran sequentially under the exclusive GPU lock with caller-held markers. CPU checks completed before timing. The existing service was restored and its health, model listing and actual generation checked.

## Current fixed-input A/B

This supplementary timing uses the repository's unchanged Python prefill harness, with RNG seed 20261029 fixed by the supplied wrapper. It measures loaded prefill latency; it is not a `kapitsa bench` submission or a decode-throughput experiment. Each leg starts a fresh server/home, uses the same warmup inputs, then the **same complete timed request body**, SHA256 `14d716a878d157133545aeb78864ad34839727f0f951f6c4045346c7a896f424`. The raw bodies and their hashes are retained. There are no cached follow-ups or decode-rate claims in this check.

Flags: `kapitsa serve --private --model MODEL_9B --model-id qwen3.5-9b --ctx 32768 --slots 1 --prefill-chunk 2048 --disk-cache 0 --dflash none`. Shipping ANE defaults; no `NI_ANE_H4096`, no forced share or schedule. Harness flags: `--sizes 4000 --skip decode,greedy,tools,agent --cached-samples 0 --warmup full`; its ordinary sampling settings are retained. Fresh server warmup pays compilation/startup costs. Thus these are loaded-service timings, not cold-start latency.

| Leg | Build | Prompt tokens | TTFT (s) | Prefill tok/s | Cold cached tokens | Other GPU mean/max % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 16ead4cf | 4134 | 9.2978 | 444.6 | 0 | 2.74 / 3.73 |
| B1 | 93b2bd7c | 4134 | 3.4318 | 1204.6 | 0 | 3.5 / 4.52 |
| A2 | 16ead4cf | 4134 | 8.6208 | 479.5 | 0 | 3.26 / 6.73 |
| B2 | 93b2bd7c | 4134 | 3.1567 | 1309.6 | 0 | 2.44 / 3.42 |

Every A B A B leg is reported and has empty invalid/flag arrays. Power is recorded before and after each leg. The wrapper uses the repository's quiet-GPU wait and contention monitor; raw samples, swap/memory snapshots and thermal information are retained. Mean TTFT: **8.9593 -> 3.2943 s (-63.23%)**. Two pairs are not a confidence interval. Logs confirm the timed B legs use ANE and A remains GPU-only.

## Current output checks

`build/logit_dump` and `tools/logit_compare.py` compare eight fixed corpus windows (prefixes 512-3200), 24 teacher-forced rows per window, chunk 2048, with `--from` preserving identical prefixes. `NI_ANE_ASYNC=0 NI_ANE_KEEP_S=0` for logit tools; no balance/share override. Step/verify/batch score the same positions, not independent datasets.

The removal itself is compared directly against the saved `fac53c04` binary: **4B, 9B and 27B each have 584/584 rows bitwise identical** (8 prefill + 192 per decode path). This supports preserving the existing 2560 segmentation and the 9B path on these samples. Current 9B `NI_ANE=0` is also 584/584 bitwise identical to main.

Current 9B default versus main still changes outputs:

| Path | Rows | Top-1 flips | Mean KL | p99 KL | Maximum KL | Mean NLL change (nats) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 8 | 0 | 0.0004784752 | 0.001989358 | 0.002064196 | +7.509513e-05 |
| step | 192 | 2 | 0.000369141 | 0.007209772 | 0.009696857 | +0.001680195 |
| verify | 192 | 2 | 0.0003767444 | 0.007340712 | 0.01012497 | +0.001696579 |
| batch | 192 | 2 | 0.0003638998 | 0.00697298 | 0.009528185 | +0.001631199 |

No nonfinite values. **The repository has no 9B noise-floor file.** The ordinary comparison reports drift without applying a model floor; its exit 0 is not a gate pass. Explicit `--min-top1 1 --max-mean-kl 0 --max-p99-kl 0 --max-nll-change 0` fails all three decode paths, exit 1, as expected. That zero-drift check does not substitute for a formally measured 9B floor. No floor was changed and `--reorder` was not used. Evaluation against a separately established 9B noise floor remains outstanding.

## Current functional validation

- Release build; `vk cpu-tests` 18/18; 14/14 supplemental CPU/tokenizer checks, including `qwen4_config_test`; `prek run --all-files`; `git diff --check`: passed.
- `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/ane_segment_check --gpu`: both layers active. GPU widths **2048, 2560, 4096, 5120, 6144, 8192**; rows 0, 1, 31, 32, 33, 127; fp64 rotation reference, token-wide scale, segmented/full packing, weight staging, padding and guards. Wider widths are rejection tests on CPU, not GPU dispatches.

```text
CPU input selection: 50 checks, 0 failures
GPU input segments: 2098 comparisons, 34105984 data/guard bytes checked, 0 failures
```

- Default 9B `test_ane_split_abort.py`: 17/17 passed, including injected failure, the 250 ms bound, GPU replay and the following GPU-only request.
- Default 9B native API with two slots and structured API: passed. Servers use shipping ANE defaults, chunk 2048, disk cache disabled. Prior one-slot cache-retention failures and their upstream reproduction remain in the older archive.
- Full-model runs omit the Metal debug layer because unchanged `attn_prefill11g` has 20,480 B static storage, counted by the layer as 40,960 B against a 32,768 B cap. The segmentation checks above ran with both layers.

## Previously measured performance and numerical context

The [fac53c04 evidence](https://github.com/KeyingSong0123/kapitsa/tree/1ab0f76e217f40bdebae95d56c79bc75fa2b6cab/bench/results/2026-10-29-m4pro-ane-default2048) stays pinned to that source. Its four `kapitsa bench` legs on this M4 Pro measured loaded ~4K TTFT 9.00825 -> 3.62055 s (-59.81%), decode effectively unchanged, peak memory 5.76 -> 6.20 GB. Those CLI runs used the same frozen corpus but different cache-busting nonces (4132-4134 tokens); they are not the fixed-input measurements above. The split reported 374.7 MB Metal allocations; cold program compilation was 31.635 s + 93 ms load. Normal servers prefill on the GPU until asynchronous compilation finishes. Those memory/startup measurements were not repeated for the removal-only commit.

The original three-model code study at `92e8b88` compared each model with its own GPU reference: mean KL 4B 3.9396e-4, 9B 1.7953e-4, 27B 8.2487e-5; flips 3/768, 3/768, 2/768. Sampled 9B mean drift was 0.456x 4B and 2.176x 27B. [Original study and document-bootstrap limits](https://github.com/KeyingSong0123/kapitsa/blob/9f8dd15a81aa67a69b4e1a86a5c4e2c67495194d/bench/results/2026-10-28-m4pro-9b-ane/README.md).

At `fac53c04`, the broader C++/Python/prose/chat sample (4092 positions, model-native templates) gave:

| Model | Mean KL lower bound | p99 | Maximum | Top-1 agreement | GPU PPL -> ANE PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 0.00185363 | 0.0261503 | 0.32020 | 98.412% | 6.90712 -> 6.89651 |
| 9B | 0.00315867 | 0.0203410 | 3.61984 | 98.778% | 6.41118 -> 6.41682 |
| 27B, native template | 0.00196361 | 0.0191110 | 1.90975 | 98.900% | 4.21701 -> 4.22375 |

This KL uses the reference's top 256 plus remaining mass (a lower bound), with each model's own 4-bit GPU reference, not bf16. 4B/9B use `ids_qwen35`, 27B uses `ids_qwen38`; code/prose IDs match, chat/math renderings differ. These samples include early prompt positions, unlike the standard full-vocabulary logit check. PPL is teacher-forced likelihood, not reasoning accuracy. 9B mean drift was 1.70x/1.61x 4B/27B, but maxima were 11.30x/1.90x: similar sampled means do not imply uniform or worst-case equivalence.

The separately reported math sample (1023 positions) had 4B/9B/native-27B mean KL 0.00289598/0.00110341/0.00118260 and PPL 3.39864 -> 3.38700 / 3.11591 -> 3.11461 / 6.76221 -> 6.70327. The fixed-token 27B control with `ids_qwen35` remains separate: four-domain mean 0.00274766, max 6.57894, PPL 4.38568 -> 4.38674; math mean 0.00769977, max 1.88452, PPL 62.48712 -> 61.97551. All per-domain results, previous long-context checks and original printed reports remain in the pinned archive. None is relabeled as a new quality run on `93b2bd7c`.

## Not verified

A formally established 9B noise-floor gate; model-level quality/speed at the additional 2048/6144/8192 widths; M5 or other Macs; other weight formats, MoE/35B-A3B, partial ANE shares, bf16-reference quality, full math500-quick reasoning accuracy, current-head long-context quality, saved-prefix replay, long cold-compile cancellation and memory-pressure transitions. New current-head speed was measured on 9B only; the 4B/27B checks verify output preservation. Prior cache-namespace checks and startup/memory observations remain historical. No default use or performance claim is made for hidden > 8192, which is rejected.

## Evidence

The source PR has 10 files and 397 counted changed lines, with its existing 29-line historical summary untouched. These new results are published separately from the source branch. Local validation does not imply remote CI or merge approval.

- `provenance.json`: source versions, binary hashes, model revisions, tokenizer identities, CPU results.
- `final-source-validation.json`: the comment-only follow-up and identical executable hashes after rebuilding.
- `commands.json`: exact GPU commands, environments and experiment server launches (local paths normalized).
- `comparisons.json` and `check-output.txt`: full comparison results and printed checks, including the expected failed zero gate.
- `fixed-prefill.json` and `fixed_prefill.py`: all timed legs, request bodies/hashes, telemetry and the portable fixed-nonce harness wrapper.
- `runtime-excerpts.txt`: selected ANE engagement/availability/allocation lines from each fresh server.
- `logit-inputs.json` and `logit-metadata.json`: exact token IDs/records, original raw-logit hashes and normalized metadata. No raw logits, model weights or binaries are committed.

Reproduction: build main, `fac53c04` and `93b2bd7c` separately in Release, including `kapitsa`, `logit_dump` and `ane_segment_check`. Use the model revisions above, take the exclusive GPU lock and set caller-held markers. Reconstruct the numeric text files from `logit-inputs.json`'s `files`/`text_by_sha256` mapping in a fresh directory, checking each hash; use them with `logit_dump --from`. Run the recorded comparison commands and expected zero gate without editing a floor. For fixed-input timing, start a fresh server/home per recorded A B A B leg and run `python3 fixed_prefill.py REPO PORT SERVER_PID OUT LABEL` under the lock. Keep every result and flag; the wrapper fixes input nonces, not generated answers. `SHA256SUMS` covers the archive files.
