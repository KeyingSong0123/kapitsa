# Default ANE prefill for hidden sizes divisible by 2048

The supported dense Qwen backend now makes positive multiples of 2048 eligible for ANE prefill by default. Qwen3.5-9B (hidden 4096) uses two 2048-channel input segments. Existing 2560-channel segmentation is preferred where it fits, so the 4B (2560) and 27B (5120) keep their existing geometry and rotation. `NI_ANE_H4096` is removed; `NI_ANE=0` disables the split. Hardware availability, weight-format, intermediate-size, memory, startup-check and failure-fallback conditions still apply; this is not support for arbitrary architectures or MoE.

**This is an intentional numerical change for newly supported models. The existing W8A8 path changes logits and can change generated tokens.** It improves the measured 9B prefill, adds memory and compilation costs, and does not establish quality or speed at every newly eligible width.

Gate/up partial sums are combined before SwiGLU. The 128-element Hadamard blocks, token-wide activation scale, weight-row scale, down-projection segmentation and wavefront scheduling remain the existing ones. Above 8192 hidden channels, a two-pass rotation finds the whole-token peak and recomputes blocks under that single scale, with 128 B of static threadgroup storage. Newly eligible widths receive a distinct SSD prefix-cache format marker. `split_bytes()`'s production caller, `Qwen35::ane_plan`, rejects unsupported rotation widths before using its result as a budget. The abort test also waits for a cold asynchronous compile to finish and sends the request that activates the ready split.

## Measured source and protocol

Runs: MacBook Pro (14-inch, 2024), Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 26.4.1 (25E253).

Kapitsa version (commit): A = `0.2.24 (16ead4cf201acd8f321090db7c0c3aacbe7de95f)`; B = `0.2.24 (fac53c043ea5bcca7e3a40330ef40ca3c96e3dc8)`. B contains A, including the upstream ANE compiler-thread lifetime fix. All current measurements below use these binaries; historical results have their own attribution.

Models (4-bit, group 64):

| Model | Repository | Revision |
| --- | --- | --- |
| 4B | lmstudio-community/Qwen3.5-4B-MLX-4bit | `c43ee1d65576a5d98de1e8405cac93c371a655c1` |
| 9B | lmstudio-community/Qwen3.5-9B-MLX-4bit | `b455506b0f574c74616dbcd56879bde38fafcff3` |
| 27B | mlx-community/Qwen3.8-27B-4bit | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |

All three tokenizer files have SHA256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`. Binary hashes, model identities, raw comparison reports, commands and input IDs are linked below. Model/GPU jobs ran sequentially under the exclusive `/tmp/kapitsa-gpu.lock`; callers held both `NI_LOCK_HELD=1` and `NACE_GPU_LOCK_HELD=1`. The existing local service was stopped for measurement and restored afterward, with health, model listing and generation checked.

## Interleaved performance

`kapitsa bench` produced every result below, in **A1 B1 A2 B2** order, after CPU validation finished. Each leg used a fresh server and isolated `KAPITSA_HOME`, with the 9B model loaded and ANE program compilation cached. The benchmark clears `NI_*` overrides before starting its server, so the candidate's ANE path is the shipping default, not an environment opt-in. Logs confirm actual split evaluations for B and GPU-only prefill for A.

Flags: `kapitsa bench --model MODEL_9B --quick --prefill-chunk 2048 --json --out RUN.json`. Plain decoding, no drafter; 512 decode tokens, three temperature-0.6 and three temperature-0 decode samples per leg. The explicit 2048 chunk equals this dense model's automatic choice on this Mac. The frozen corpus SHA256 is `58a7756fd94fc48a7b7306d330b2c44663bdf22faa943f82430966c2878081b2`. The harness's per-run prompt sizes are retained rather than rounded to an identical token count.

| Leg | Commit | Prompt tokens | Loaded TTFT (s) | Prefill tok/s | Decode median tok/s | Greedy median tok/s | Cached follow-up median (ms) | Reported peak GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | `16ead4cf` | 4132 | 9.0228 | 458.0 | 50.45 | 50.52 | 74.8 | 5.76 |
| B1 | `fac53c04` | 4134 | 3.6446 | 1134.3 | 51.43 | 50.44 | 74.1 | 6.20 |
| A2 | `16ead4cf` | 4133 | 8.9937 | 459.5 | 50.79 | 50.40 | 75.3 | 5.76 |
| B2 | `fac53c04` | 4133 | 3.5965 | 1149.2 | 50.65 | 50.39 | 91.7 | 6.20 |

Mean loaded TTFT: **9.00825 -> 3.62055 s (-59.81%)**; mean prefill: **458.75 -> 1141.75 tok/s (2.49x)**. Decode medians average 50.62 -> 51.04 tok/s (+0.83%); greedy medians average 50.46 -> 50.415 (-0.09%). These two pairs support a prefill benefit, not a demonstrated decode benefit or a confidence interval. Cached follow-up performance is not improved consistently. Peak memory reported by the benchmark rises **0.44 GB**.

Every leg has empty `flags`, `invalid` and `contention` arrays; AC power, automatic power mode, fair thermals, zero swapouts, 18.1 s quiet-GPU wait and zero mean ambient GPU activity. Other-GPU mean/max percentages in A1/B1/A2/B2 are 1.02/7.80, 0.52/2.64, 0.70/5.11 and 0.31/2.10; the A legs list Codex as a GPU user. These are all the benchmark legs, with no discarded timing run. Raw JSON retains all decode and follow-up samples.

The split reports **374.7 MB of Metal allocations**. A separate functional run at B measured a cold program compile of **31.635 s + 93 ms load**, ready in 31.8 s; its startup numerical check reported 1.94%. That check is not a model-quality metric. Normal servers compile asynchronously and prefill on the GPU until ready. Therefore the loaded TTFT speedup does not describe a first request during cold compilation, and the benchmark peak is not a measurement of the ANE daemon's total memory.

## Current full-vocabulary output checks

`build/logit_dump` plus `tools/logit_compare.py`, chunk 2048, eight standard corpus windows with prefixes 512-3200, 24 teacher-forced rows each. `NI_ANE_ASYNC=0 NI_ANE_KEEP_S=0` ensures compilation completes before comparison and avoids keepalive work; normal routing/share/balance defaults are retained. Candidate dumps use `--from` the reference. Step, verify and batch score the same positions, not three independent datasets.

| 9B path | Rows | Top-1 flips | Mean KL | p99 KL | Maximum KL | Mean NLL change (nats) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 8 | 0 | 4.7847517e-4 | 1.9893584e-3 | 2.0641965e-3 | +7.5095129e-5 |
| Step | 192 | 2 | 3.6914095e-4 | 7.2097717e-3 | 9.6968571e-3 | +1.6801951e-3 |
| Verify | 192 | 2 | 3.7674437e-4 | 7.3407117e-3 | 1.0124971e-2 | +1.6965788e-3 |
| Batch | 192 | 2 | 3.6389980e-4 | 6.9729802e-3 | 9.5281845e-3 | +1.6311986e-3 |

No nonfinite values. Each decode path has 190/192 top-1 agreement; no 9B row is bitwise identical. Both repeat controls (A1/A2 and B1/B2) are **584/584 bitwise identical**, and B2 repeats the B1 comparison. **4B default, 27B default, and 9B `NI_ANE=0` are each 584/584 bitwise identical to A.** The 4B/27B result compares the old and new implementations with ANE already available in both; it does not mean ANE equals GPU for those models.

**No stored 9B noise floor exists.** The normal comparison is report-only; its exit 0 is not an output-gate pass. A fresh A/B dump pair with explicit `--min-top1 1 --max-mean-kl 0 --max-p99-kl 0 --max-nll-change 0` **fails all three decode paths, exit 1**, as expected for the intended numerical change. These explicit zero bounds demonstrate nonzero drift, not a measured or accepted 9B floor. No floor was edited and `--reorder` was not used.

The additional 9B 4K/16K check uses two offsets at each length, 64 step rows each: **0/256 top-1 flips**, mean KL **2.9773362e-4**, p99 **3.3892772e-3**, maximum **2.1341838e-2**, mean NLL change **-1.5651949e-3 nats**. The four prefill rows also have no flips (mean KL 5.5296344e-4). These are teacher-forced rows, not proof that free-running continuations stay identical.

## Three-model numerical context

The original same-prefix code study at experimental commit `92e8b882b003971e6569fb58d1147b49e69c5d8a`, on this M4 Pro, compared each model's ANE against its own 4-bit GPU reference. Eight code documents x 96 step rows = 768 rows; `NI_BAL=0`; A B A B repeat hashes matched. The separate code PPL sample contained 8192 positions.

| Historical model | Mean full-vocabulary KL | Top-1 flips / 768 | GPU PPL -> ANE PPL |
| --- | ---: | ---: | ---: |
| 4B | 3.9395954e-4 | 3 | 2.26564 -> 2.26824 |
| 9B | 1.7952624e-4 | 3 | 2.01072 -> 2.01052 |
| 27B | 8.248671e-5 | 2 | 1.44690 -> 1.44765 |

Historical 9B/4B mean-KL ratio: **0.456** (paired document-bootstrap 95% [0.242, 1.021]); 9B/27B: **2.176** [1.685, 2.820]. At 4K/16K the latter ratios were 2.904/2.039. The bootstrap resampled eight documents 20,000 times, not independent tokens. The 9B code-PPL delta was -0.00995%, with an interval crossing zero. [Pinned original study, controls and limitations](https://github.com/KeyingSong0123/kapitsa/blob/9f8dd15a81aa67a69b4e1a86a5c4e2c67495194d/bench/results/2026-10-28-m4pro-9b-ane/README.md).

Those data support comparable sampled **mean** drift, not an error bound from the segmentation geometry. To broaden the sample, the current binaries also ran `ni_eval`, chunk 1024, skip 0, on C++, Python, prose and chat (4092 positions), with math separately (1023 positions). The reference is each model's own 4-bit GPU model, **not bf16**. KL below uses the reference's top 256 probabilities plus the remaining probability mass: it is a **lower bound** on full-vocabulary KL. PPL is teacher-forced next-token likelihood, not task or reasoning accuracy. These samples include early prompt positions, unlike the 512+-prefix full-vocabulary test above.

The primary table uses model-native templates: `ids_qwen35` for 4B/9B and `ids_qwen38` for 27B. Code/prose IDs match between directories; chat/math renderings differ.

| Current four-domain model | Mean KL lower bound | p99 | Maximum | Top-1 agreement | GPU PPL -> ANE PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 0.00185363 | 0.0261503 | 0.32020 | 98.412% | 6.90712 -> 6.89651 |
| 9B | 0.00315867 | 0.0203410 | 3.61984 | 98.778% | 6.41118 -> 6.41682 |
| 27B, native template | 0.00196361 | 0.0191110 | 1.90975 | 98.900% | 4.21701 -> 4.22375 |

9B's mean is **1.70x 4B / 1.61x native 27B**, while its maximum is **11.30x / 1.90x**. The 9B prose maximum is 2.65976 and chat maximum 3.61984. Aggregate 9B PPL increases about 0.088%. Average drift remains in a similar range, but individual tails can be substantially larger; the evidence does not support uniform error equivalence or a guarantee of unchanged answers.

| Current math model | Mean KL lower bound | Maximum | Top-1 agreement | GPU PPL -> ANE PPL |
| --- | ---: | ---: | ---: | ---: |
| 4B | 0.00289598 | 0.641531 | 98.925% | 3.39864 -> 3.38700 |
| 9B | 0.00110341 | 0.259390 | 99.511% | 3.11591 -> 3.11461 |
| 27B, native template | 0.00118260 | 0.223365 | 99.413% | 6.76221 -> 6.70327 |

The separate identical-input 27B control used `ids_qwen35`, preserving a fixed-token comparison across all models: four-domain mean KL 0.00274766, p99 0.0143471, maximum 6.57894, top-1 98.900%, PPL 4.38568 -> 4.38674; math mean KL 0.00769977, maximum 1.88452, PPL 62.48712 -> 61.97551. These results remain in the archive, but are not substituted for the native-template 27B comparison. Absolute PPL should not be used here to rank different model/template combinations.

## Current validation

- Release build of CLI and required tools; `vk cpu-tests` **18/18**; **13/13** supplemental checks (11 C++ including parser/tokenizer checks, logit-dump CLI and structured tokenizer checks); `prek run --all-files`; `git diff --check`: passed.
- `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/ane_segment_check --gpu`: **146 CPU checks, 3872 GPU comparisons, 83,542,976 data/guard bytes, zero failures**. Both validation layers are logged active. Widths 2048, 2560, 4096, 5120, 6144, 8192, 10240, 12288, 18432; rows 0, 1, 31, 32, 33, 127. Independent fp64 rotation reference, token-wide scale, padding, packing, weight staging and guards.
- 9B `tests/test_ane_split_abort.py`, with no `NI_ANE_H4096`: **17/17 passed**. Normal split, injected failure, 250 ms bound, retirement, GPU replay and the following GPU-only request. These timings are functional evidence, not additional benchmark samples.
- Native API suite passed with two slots; structured API passed. Initial one-slot native API had three cache-retention assertion failures, reproduced exactly on upstream with one slot: `main slot survives side request`, `one-offs keep the conversation in progress`, `a side request sent beside a first request does not evict it`. The initial failed run is retained.
- Cache namespace checks passed: new default differs from `NI_ANE=0`; opt-out matches upstream; a default restart selects the same default namespace. Actual saved-prefix replay was **not tested**, because free disk space was below the engine's 10 GiB reserve. No reserve was bypassed.

## Not verified

Whole-model quality/performance at newly eligible widths other than 9B; M5 or other Macs; other weight formats, MoE/35B-A3B, partial ANE shares, bf16-reference quality, full math500-quick reasoning accuracy, saved-prefix replay, cancellation during a long uncached background compile, and memory-pressure transitions. A synthetic shape check does not establish model quality or speed. The general eligibility proposal deliberately extends beyond the measured 9B model and remains a review decision.

The full-model runs omit the Metal debug layer because unchanged `attn_prefill11g` has 20,480 B static threadgroup storage, counted by this layer as 40,960 B against a 32,768 B cap. The changed rotation/segmentation kernels were checked at small sizes with both layers enabled. Local checks do not establish remote CI or merge approval.

Earlier Mac mini M4 measurements and functional checks remain separately attributed in the [pinned M4 archive](https://github.com/KeyingSong0123/kapitsa/tree/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review). They are not relabeled as measurements of `fac53c04` on this M4 Pro.

## Evidence files and reproduction

- [provenance.json](provenance.json): exact source versions, binary hashes, model revisions, tokenizer hash and quality-reference hashes.
- [commands.json](commands.json): executed argument vectors, explicit environment overrides, times and exit codes, including the initial API failure and explicit zero-gate failure.
- [benchmarks.json](benchmarks.json): all four raw `kapitsa bench` objects and the arithmetic used for the summary.
- [comparisons.json](comparisons.json) and [comparison-output.txt](comparison-output.txt): full comparison JSON and printed tool output, including every flip and failure.
- [quality.json](quality.json): every per-domain and aggregate current quality result, with native-template and fixed-input controls distinguished.
- [validation.json](validation.json) and [validation-output.txt](validation-output.txt): CPU, kernel, abort, API and cache results, including failed one-slot checks and upstream controls.
- [runtime-excerpts.txt](runtime-excerpts.txt): selected ANE engagement/compile/allocation/fallback lines, identified by original run label.
- [inputs.json](inputs.json) and [logit-metadata.json](logit-metadata.json): exact numeric input text, record mappings, original hashes and normalized metadata. Large raw logits were hashed and removed after comparison; they are not included. Metadata paths alone are normalized for publication, as documented in that file.
- [SHA256SUMS](SHA256SUMS): archive file hashes.

Local paths are normalized to `$REFERENCE` (checkout of A), `$CANDIDATE` (checkout of B), `$RESULTS` (fresh output root), `$MODEL_4B`, `$MODEL_9B`, `$MODEL_27B`, and `$HISTORICAL_RESULTS`. Commands using private orchestration helpers are records of execution, not supplied executables; their underlying public test commands and settings are given here and in `validation.json`. No controller, model, binary, service configuration or process inventory is published.

Build A and B in separate folders using the repository's Release commands, including targets `kapitsa`, `logit_dump`, `ni_eval`, and `ane_segment_check`. Take the exclusive GPU lock and use caller-held markers for model tools. Run changed kernels with both Metal validation variables before full-model tests. Use a fresh results directory, so no published result is overwritten.

For the standard logit comparison, under the lock:

```sh
NI_ANE_ASYNC=0 NI_ANE_KEEP_S=0 "$REFERENCE/build/logit_dump" "$MODEL_9B" "$RESULTS/A" --chunk 2048
NI_ANE_ASYNC=0 NI_ANE_KEEP_S=0 "$CANDIDATE/build/logit_dump" "$MODEL_9B" "$RESULTS/B" --chunk 2048 --from "$RESULTS/A"
python3 "$CANDIDATE/tools/logit_compare.py" "$RESULTS/A" "$RESULTS/B" --json "$RESULTS/compare.json"
python3 "$CANDIDATE/tools/logit_compare.py" "$RESULTS/A" "$RESULTS/B" --min-top1 1 --max-mean-kl 0 --max-p99-kl 0 --max-nll-change 0
```

The last command is expected to return 1. Repeat A and B in that order for controls; compare candidate `NI_ANE=0` against A; run 4B and 27B defaults similarly. Keep the same input mapping via `--from`. Original standard and long input files can be restored from `inputs.json` without writing any existing file:

```python
import hashlib, json
from pathlib import Path
j = json.loads(Path('inputs.json').read_text())
out = Path('restored-inputs')
out.mkdir(exist_ok=False)
for name, sha in j['files'].items():
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    data = j['text_by_sha256'][sha].encode()
    assert hashlib.sha256(data).hexdigest() == sha
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data)
```

For long checks, pass the restored `dumps/9b-long-A` to `--from` and request `--paths step`. For quality, use the appropriate `bench/quality/ids_qwen35` or `ids_qwen38` files: A `ni_eval --dump-ref REF.bin --topk 256 --chunk 1024 --skip 0` (with `NI_ANE=0` for 4B/27B), then B `ni_eval --ref REF.bin --chunk 1024 --skip 0` on the same `@file` arguments. Reference files are runtime outputs, not committed binaries. See the exact command vectors for all domain groupings and env settings.

For functional testing use `python3 tests/test_ane_split_abort.py build/kapitsa MODEL_9B --port PORT --keep`, then fresh native servers with `--model-id qwen3.5-9b --ctx 32768 --disk-cache 2G --dflash none`, first with `--slots 1` and then `--slots 2`. Run `tests/test_server.py URL --model-id qwen3.5-9b` and `tests/test_structured.py URL`. The API runs used `NI_ANE_ASYNC=0 NI_ANE_KEEP_S=0` and a shared isolated home for namespace checks; `validation.json` records the command template. Preserve initial failures and the upstream control. For performance, run the four recorded `kapitsa bench` commands in A B A B order, with no concurrent CPU/GPU job, and retain every JSON leg and any validity flag.
