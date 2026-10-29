# Shared decode reduction: results

Runs: main `4256a51` vs native candidate `8ea1723`; Apple M4 Pro, 20 GPU cores, 48 GB, macOS 15.7.1 (24G231), AC power.

`bench.jsonl` and `model-validation/clean-bench.jsonl` retain both isolated-kernel sessions, six interleaved A/B pairs per shape. These timings do not establish a whole-request speedup.

[Validation and reproduction](model-validation/VALIDATION.md) includes all four standard request legs, output checks, and the ordinary-decode supplement. Numerical records retained here are unchanged.

[Complete original evidence](https://github.com/chainhelen/kapitsa/tree/5554eb32398076dddbecd6b2193b57eefb56b5aa/bench/results/2026-10-29-m4pro-operators/reduce) is preserved on `chainhelen/kapitsa`, branch `archive/decode-reduce-full-evidence-20261029` (pinned commit `5554eb32398076dddbecd6b2193b57eefb56b5aa`). It contains the full build/CPU/API logs, standard benchmark JSONs, both ordinary-decode attempts, prompts and generated outputs.
