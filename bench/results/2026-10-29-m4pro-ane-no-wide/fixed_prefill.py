"""Run the unchanged repository prefill harness with fixed input nonces and telemetry.

Usage: fixed_prefill.py REPO PORT SERVER_PID OUT LABEL
The caller must hold the exclusive GPU lock. Each leg needs a fresh server/home.
Only loaded prefill latency is compared; this does not measure decode throughput.
"""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

code, port, server_pid, output, label = sys.argv[1:]
output = Path(output)
sys.path.insert(0, str(Path(code) / 'bench/harness'))
import bench
import gpu_hogs
import agentic

assert gpu_hogs.held_by_caller(), 'Run under vk gpu run with caller-held lock markers'
assert not output.exists(), 'Never overwrite a result'
requests = []
stream = bench.stream_chat


def captured(base, body, session=None, timeout=1800):
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    requests.append(dict(body=body, body_sha256=hashlib.sha256(encoded).hexdigest(), session=session))
    return stream(base, body, session=session, timeout=timeout)


bench.stream_chat = captured
# Keep the repository corpus and protocol unchanged. Prevent loading machine-local harness overrides.
bench.LOCAL_ENV = str(output.parent / 'no-local-overrides.env')
assert not Path(bench.LOCAL_ENV).exists()
random.seed(20261029)
sys.argv = [str(Path(code) / 'bench/harness/bench.py'), '--base-url', f'http://127.0.0.1:{port}/v1',
    '--model', 'qwen3.5-9b', '--label', label, '--pid', server_pid, '--sizes', '4000',
    '--skip', 'decode,greedy,tools,agent', '--agent-trace', 'none', '--cached-samples', '0',
    '--warmup', 'full', '--out', str(output)]
power_before = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
assert "'AC Power'" in power_before, 'Fixed-input timing requires AC power'
thermal_before = agentic.thermal_state()
quiet, waited, busy = gpu_hogs.wait_quiet(quiet_s=15, timeout_s=180,
    exclude_roots=[int(server_pid)], log=lambda msg: print(msg, flush=True))
assert quiet, 'Foreign GPU activity did not become quiet'
monitor = gpu_hogs.ContentionMonitor([int(server_pid)])
monitor.start()
try:
    bench.main()
finally:
    monitor.stop()
power_after = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
result = json.loads(output.read_text())
phase = result['phases']['prefill_4000']
contention = gpu_hogs.summarize_samples(monitor.samples, *phase)
result['fixed_input_check'] = dict(seed=20261029, server_fresh=True, power_before=power_before,
    power_after=power_after, thermal_before=thermal_before, thermal_after=agentic.thermal_state(),
    quiet_wait_s=waited, contention=contention, telemetry=monitor.samples,
    measured_at=time.time(), requests=requests)
if "'AC Power'" not in power_after:
    result['invalid'].append('AC power disconnected during the leg')
if contention['flagged']:
    result['invalid'].append('foreign GPU contention during prefill')
assert len(requests) == 3, 'Expected two warmup requests and one timed prefill request'
assert result['prefill'][0]['answer'].strip(), 'Empty measured response'
output.write_text(json.dumps(result, indent=2) + '\n')
print('FIXED PREFILL', label, result['prefill'][0]['ttft_s'], 'invalid', result['invalid'],
      'flags', result['flags'], flush=True)
