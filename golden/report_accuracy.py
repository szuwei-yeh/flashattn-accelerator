"""Compare hardware fixed-point attention with float64 on the same INT8 inputs.

All scales are the encoded signed Q8.8 values, not original quantizer floats.
Canonical expected files are checked, never rewritten. Additional fixtures are
generated in memory. This measures arithmetic approximation, not model accuracy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from generate_hw_expected import (flash_attn_q_tile, load_exp_lut,
                                  load_int8_hex, load_scales, signed16)

ROOT = Path(__file__).resolve().parents[1]


def floating_attention(q, k, v, scales, causal=False):
    sq, sk, sv = (signed16(s)/256 for s in scales)
    scores = (q.astype(np.float64)*sq) @ (k.astype(np.float64)*sk).T / np.sqrt(q.shape[1])
    if causal:
        scores[np.triu_indices(q.shape[0], 1)] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ (v.astype(np.float64)*sv)


def error_metrics(actual, reference):
    error = actual-reference
    denominator = np.linalg.norm(reference)
    return dict(relative_l2=float(np.linalg.norm(error)/denominator) if denominator else None,
                rmse=float(np.sqrt(np.mean(error**2))),
                max_abs=float(np.max(np.abs(error))))


def fixed_attention(q, k, v, scales, lut, causal):
    result = []
    for tile in range(q.shape[0]//16):
        result.extend(flash_attn_q_tile(q[tile*16:(tile+1)*16], k, v,
                                       *scales, lut, 16, causal, tile))
    return np.asarray(result, dtype=np.int64)


def measure(name, q, k, v, scales, lut, causal, expected=None):
    fixed = fixed_attention(q, k, v, scales, lut, causal)
    if expected is not None and not np.array_equal(fixed, expected):
        raise ValueError(f'{name}: stored expected output differs from fixed-point model')
    reference = floating_attention(q, k, v, scales, causal)
    product = q.astype(np.int64) @ k.astype(np.int64).T
    sq, sk, sv = map(signed16, scales)
    before_clamp = (product*sq*sk+128) >> 8
    visible = np.tril(np.ones(product.shape, dtype=bool)) if causal else np.ones(product.shape, dtype=bool)
    saturation = ((before_clamp < -32768) | (before_clamp > 32767)) & visible
    return dict(case=name, n=q.shape[0], d=q.shape[1], causal=causal,
                scale_q_q88=sq, scale_k_q88=sk, scale_v_q88=sv,
                visible_scores=int(visible.sum()), saturated_scores=int(saturation.sum()),
                **error_metrics(fixed/256, reference))


def quantize(values):
    scale = float(np.max(np.abs(values)))/127
    if scale == 0:
        return np.zeros(values.shape, dtype=np.int8), 256
    encoded = max(1, min(32767, round(scale*256)))
    # Match the fixture generator: quantize with the original float scale, then
    # compare both algorithms using the encoded scale of those same integers.
    return np.clip(np.rint(values/scale), -128, 127).astype(np.int8), encoded


def generated_cases(dim):
    for seed in (7,42,20261003):
        for amplitude in (.25,1.,4.):
            rng = np.random.default_rng(seed)
            quantized = [quantize(rng.standard_normal((64,dim))*amplitude) for _ in range(3)]
            matrices = [entry[0] for entry in quantized]
            scales = [entry[1] for entry in quantized]
            yield f'random_seed{seed}_amp{amplitude:g}', matrices, scales
    # Deliberate precision limits, distinct from ordinary random inputs.
    q,k,v = [np.zeros((64,dim),dtype=np.int8) for _ in range(3)]
    v[:] = 1
    yield 'uniform_bias', [q.copy(),k.copy(),v.copy()], (256,256,256)
    q[:,0] = 2
    k[:,0] = np.where(np.arange(64)%2,127,64)
    v[:] = np.where((np.arange(64)%2)[:,None],127,-128)
    yield 'saturated_distinct', [q.copy(),k.copy(),v.copy()], (256,256,256)
    yield 'signed_scales', [q,k,v], (-256,256,-256)


def report():
    lut = load_exp_lut(ROOT/'data/exp_lut.hex')
    rows, sources = [], {}
    for dim in (16,64):
        for causal in (False, True):
            folder = ('causal_N64' if causal else 'N64') + ('_d64' if dim == 64 else '')
            data = ROOT/'data'/folder
            sq,sk,sv,n,d = load_scales(data/'scales.txt')
            matrices = [load_int8_hex(data/f'{name}_input.hex',n*d).reshape(n,d) for name in ('q','k','v')]
            words = np.array([int(t,16) for t in (data/'expected.hex').read_text().split()],dtype=np.int64)
            expected = np.where(words >= 2**31, words-2**32, words).reshape(n,d)
            rows.append(measure(f'canonical_{folder}',*matrices,(sq,sk,sv),lut,causal,expected))
            for file in data.iterdir():
                if file.name in ('q_input.hex','k_input.hex','v_input.hex','expected.hex','scales.txt'):
                    sources[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
            for name, matrices, scales in generated_cases(dim):
                rows.append(measure(name,*matrices,scales,lut,causal))
    for name in ('data/exp_lut.hex','golden/generate_hw_expected.py','golden/report_accuracy.py'):
        sources[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return dict(schema=1, reference='NumPy float64 stable softmax, same INT8 inputs and signed Q8.8 scales',
                output_units='stored signed INT32 output / 256',
                scope='Fixed-point arithmetic approximation; not original FP32-input or model accuracy. Generated cases use the hardware reference, not a fresh RTL run.',
                seeds=[7,42,20261003], amplitudes=[.25,1.,4.], source_sha256=sources, cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    result = report()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'accuracy.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (args.output_dir/'accuracy.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(result['cases'][0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(result['cases'])
    for row in result['cases']:
        if row['case'].startswith('canonical'):
            print(f"{row['case']}: relative L2={row['relative_l2']:.6%}, RMSE={row['rmse']:.9g}, max abs={row['max_abs']:.9g}")
    print(f"RESULT: PASS {len(result['cases'])} accuracy cases; four canonical fixed-point outputs unchanged")


if __name__ == '__main__':
    main()
