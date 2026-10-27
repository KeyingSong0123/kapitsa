#!/usr/bin/env python3
"""Offline affine checkpoint format: exact known codes, metadata, copying, and refusals. CPU only."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return struct.pack('<H', (bits + 0x7fff + ((bits >> 16) & 1)) >> 16)


def write_safetensors(path, values):
    header = {}
    raw = bytearray()
    for name, (dtype, shape, data) in values.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(raw), len(raw) + len(data)]}
        raw.extend(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + raw)


def read_safetensors(path):
    raw = path.read_bytes()
    n = struct.unpack_from('<Q', raw)[0]
    header = json.loads(raw[8:8 + n])
    values = {}
    for name, v in header.items():
        if name == '__metadata__':
            continue
        a, b = v['data_offsets']
        values[name] = (v['dtype'], v['shape'], raw[8 + n + a:8 + n + b])
    return values


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/draft_pack').resolve()
    checks = 0

    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)

    with tempfile.TemporaryDirectory(prefix='kapitsa-draft-pack-test-') as td:
        root = Path(td)
        source = root / 'source'
        source.mkdir()
        config = {'architectures': ['DFlashDraftModel'], 'dflash_config': {}, 'keep_this': 'unchanged'}
        (source / 'config.json').write_text(json.dumps(config))
        (source / 'LICENSE').write_text('test attribution\n')
        # min=-15, max=15 gives scale=-15/8 and bias=15 in af4. The other 62 values
        # are exactly half a code from the bias: away -> 1, even -> 0.
        half_row = b''.join(map(bf16, [-15, 15] + [14.0625] * 62))
        ramp_row = b''.join(map(bf16, list(range(-8, 8)) * 8))
        original = {
            'fc.weight': ('BF16', [8, 64], half_row * 8),
            'layers.0.self_attn.k_proj.weight': ('BF16', [16, 128], ramp_row * 16),
            'hidden_norm.weight': ('BF16', [4], b''.join(map(bf16, [0, 1, 2, 3]))),
            'candidate_selector.predecessor_codebook': ('BF16', [8, 64], half_row * 8),
        }
        write_safetensors(source / 'model.safetensors', original)
        source_bytes = (source / 'model.safetensors').read_bytes()
        env = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_'))}
        env['BENCH_GPU_LOCK'] = str(root / 'gpu.lock')

        def run(src, dst, *args, expected=0):
            result = subprocess.run([str(binary), str(src), str(dst), *args], env=env,
                                    capture_output=True, text=True, timeout=30)
            check(result.returncode == expected,
                  f'return code {result.returncode}, wanted {expected}: {result.stderr}')
            return result

        for bits in (4, 8):
            for rounding in ('away', 'even'):
                dest = root / f'pack-{bits}-{rounding}'
                result = run(source, dest, str(bits), rounding)
                stats = json.loads(result.stdout)
                check(stats['projections'] == 2 and stats['bits'] == bits, 'conversion accounting')
                loaded = read_safetensors(dest / 'model.safetensors')
                stored = json.loads((dest / 'config.json').read_text())
                check(stored.pop('kapitsa_draft_pack') ==
                      {'version': 1, 'bits': bits, 'group_size': 64, 'round': rounding}, 'format metadata')
                check(stored == config, 'preserve other configuration fields')
                check((dest / 'LICENSE').read_bytes() == (source / 'LICENSE').read_bytes(), 'preserve attribution')
                check(not list(dest.glob('*.partial')), 'only complete output files')
                for name in ('hidden_norm.weight', 'candidate_selector.predecessor_codebook'):
                    check(loaded[name] == original[name], 'non-projection values copied unchanged')
                w, s, b = [loaded['layers.0.self_attn.k_proj.' + suffix]
                           for suffix in ('weight', 'scales', 'biases')]
                check(w[:2] == ('U32', [16, 128 * bits // 32]), 'packed weight dimensions')
                check(s[:2] == b[:2] == ('BF16', [16, 2]), 'scale and bias dimensions')
                if bits == 4:
                    words = [0x76543210, 0xfedcba98] * 8
                    check(w[2] == struct.pack('<16I', *words) * 16, 'integer ramp has exact 4-bit codes')
                    check(s[2] == bf16(1) * 32 and b[2] == bf16(-8) * 32, '4-bit ramp coefficients')
                    first = 0x1111110f if rounding == 'away' else 0xf
                    rest = 0x11111111 if rounding == 'away' else 0
                    expected = struct.pack('<8I', first, *([rest] * 7)) * 8
                    check(loaded['fc.weight'][2] == expected, 'half-code ties preserve selected rounding')
                    check(loaded['fc.scales'][2] == bf16(-1.875) * 8 and
                          loaded['fc.biases'][2] == bf16(15) * 8, 'half-code exact coefficients')
                else:
                    words = [0x33221100, 0x77665544, 0xbbaa9988, 0xffeeddcc] * 8
                    check(w[2] == struct.pack('<32I', *words) * 16, 'integer ramp has exact 8-bit codes')
                    check(s[2] == bf16(1 / 17) * 32 and b[2] == bf16(-8) * 32, '8-bit ramp coefficients')
                check((source / 'model.safetensors').read_bytes() == source_bytes, 'source checkpoint unchanged')

        existing = root / 'existing'
        existing.mkdir()
        (existing / 'sentinel').write_text('keep')
        run(source, existing, expected=1)
        check(sorted(p.name for p in existing.iterdir()) == ['sentinel'], 'refuse to overwrite a directory')
        run(source, root / 'bad-bits', '3', expected=2)
        run(source, root / 'bad-round', '4', 'unknown', expected=2)
        check(not (root / 'bad-bits').exists() and not (root / 'bad-round').exists(), 'bad options create nothing')
        run(root / 'pack-4-away', root / 'twice', expected=1)
        check(not (root / 'twice').exists(), 'refuse to requantize an already packed checkpoint')
        for name, tensor in {
            'shape': ('BF16', [8, 63], bf16(1) * (8 * 63)),
            'dtype': ('F16', [8, 64], struct.pack('<e', 1) * (8 * 64)),
            'byte-extent': ('BF16', [8, 64], bf16(1) * (8 * 64 + 1)),
            'nonfinite': ('BF16', [8, 64], struct.pack('<H', 0x7fc0) * (8 * 64)),
        }.items():
            bad = root / ('source-' + name)
            bad.mkdir()
            (bad / 'config.json').write_text(json.dumps(config))
            write_safetensors(bad / 'model.safetensors', {'fc.weight': tensor})
            out = root / ('output-' + name)
            run(bad, out, expected=1)
            check(not (out / 'config.json').exists(), 'failed conversion is not a loadable drafter')
    print(f'PASS {checks} offline draft pack checks (CPU, temporary model lock)')


if __name__ == '__main__':
    main()
