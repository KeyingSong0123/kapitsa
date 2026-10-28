# Latest-main M4 ANE 4K rerun

Loaded-model HTTP TTFT on this M4 only. Explicit W8A8 approximation changes replies. Full HTTP time includes possibly different continuation work; no decode-speed claim.

Primary protocol confirmed: **True**. Full first-pair schedule: True.

| Workload | Eligible pairs | TTFT A/B ratio (95% interval) | TTFT time reduction | Identical replies |
| --- | ---: | ---: | ---: | ---: |
| code-scheduler-4096 | 4 | 2.765 (2.734–2.797) | 63.83% | 4/4 |
| prose-zh-4096 | 4 | 2.736 (2.693–2.781) | 63.45% | 0/4 |

Every attempt, including rejected attempts:

| Workload | Pair | Attempt | TTFT A / B (s) | HTTP A / B (s) | TTFT other GPU peak including desktop A / B (%) | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| code-scheduler-4096 | 0 | 0 | 17.6476 / 6.4990 | 27.1367 / 16.5103 | 26.473 / 0.000 | False |
| code-scheduler-4096 | 0 | 1 | 17.7340 / 6.4370 | 27.0844 / 15.7353 | 0.000 / 0.000 | True |
| code-scheduler-4096 | 1 | 0 | 17.7537 / 6.4557 | 27.8025 / 15.8733 | 0.000 / 0.000 | True |
| code-scheduler-4096 | 2 | 0 | 17.5385 / 6.3779 | 27.1238 / 15.7840 | 26.193 / 0.000 | False |
| code-scheduler-4096 | 2 | 1 | 17.5349 / 6.3502 | 26.8934 / 15.8395 | 0.000 / 0.000 | True |
| code-scheduler-4096 | 3 | 0 | 17.7484 / 6.3524 | 27.1144 / 15.7599 | 0.000 / 0.000 | True |
| prose-zh-4096 | 0 | 0 | 17.7597 / 6.5887 | 37.4753 / 25.7260 | 0.000 / 0.000 | True |
| prose-zh-4096 | 1 | 0 | 18.3158 / 6.4382 | 37.6952 / 24.9596 | 35.894 / 0.000 | False |
| prose-zh-4096 | 1 | 1 | 17.5897 / 6.3900 | 36.6953 / 25.3396 | 0.000 / 0.000 | True |
| prose-zh-4096 | 2 | 0 | 17.7321 / 6.4438 | 36.9367 / 24.9109 | 0.247 / 0.000 | True |
| prose-zh-4096 | 3 | 0 | 17.7428 / 6.4624 | 36.9427 / 24.7407 | 0.038 / 0.000 | True |
