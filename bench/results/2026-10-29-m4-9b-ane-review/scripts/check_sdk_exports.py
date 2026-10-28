from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[1]
repo = Path('$REPO')
actual = sorted(subprocess.check_output(['nm', '-gUj', str(root / 'sdk-build/libkapitsa_sdk.dylib')], text=True).splitlines())
expected = sorted((repo / 'sdk/c/exports.txt').read_text().splitlines())
assert actual == expected, (actual, expected)
print('PASS SDK exports match the C ABI contract')
