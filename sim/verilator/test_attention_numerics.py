"""Analytic and metamorphic RTL checks, without importing the hardware golden.

Requires the four core/top N64 d16/d64 binaries (`make numerical_checks`).
Uniform-score cases have closed-form expected outputs. Permutation cases use
the preserved canonical outputs and mathematical data-layout invariants.
"""
import random
import tempfile
import unittest
from pathlib import Path

from test_tb_oracles import ROOT, fixture, run_attention


def read_matrix(path, dim, bits):
    words = [int(x, 16) for x in path.read_text().split()]
    words = [x - (1 << bits) if x >= (1 << (bits - 1)) else x for x in words]
    return [words[i:i + dim] for i in range(0, len(words), dim)]


def write_matrix(path, matrix, bits):
    path.write_text(''.join(f'{x & ((1 << bits) - 1):0{bits // 4}x}\n'
                            for row in matrix for x in row))


def trunc_div(numer, denom):
    return (1 if numer >= 0 else -1) * (abs(numer) // denom)


class AttentionNumerics(unittest.TestCase):
    def check_rtl(self, dim, q, k, v, expected, scales, causal=False):
        with tempfile.TemporaryDirectory(prefix='attention-numerics-') as temp:
            data = Path(temp)
            for name, matrix in zip(('q', 'k', 'v'), (q, k, v)):
                write_matrix(data / f'{name}_input.hex', matrix, 8)
            write_matrix(data / 'expected.hex', expected, 32)
            (data / 'scales.txt').write_text(scales)
            for kind in ('core', 'top'):
                with self.subTest(kind=kind):
                    result = run_attention(kind, dim, data, causal=causal)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn('RESULT: PASS', result.stdout)

    def test_closed_form_uniform_and_saturated_scores(self):
        # With equal visible scores: exp=256, P=255, rescale=256.
        # Each 16-row block contributes floor(255 * sum(V) * sv / 256).
        # The final Q8.8 result is trunc(sum(block contributions)/visible_rows).
        # This is a direct formula, with no online-softmax reference involved.
        cases = [('uniform', sv) for sv in (256, 7, -7, 32767, -32768)]
        cases += [('constant_v', 256), ('sat_positive', 256), ('sat_negative', 256)]
        for dim in (16, 64):
            for causal in (False, True):
                for name, sv in cases:
                    with self.subTest(dim=dim, causal=causal, case=name, sv=sv):
                        q = [[0] * dim for _ in range(64)]
                        k = [[0] * dim for _ in range(64)]
                        v = [[((r * 37 + c * 19) % 256) - 128 for c in range(dim)]
                             for r in range(64)]
                        sq = -256 if name == 'sat_negative' else 256
                        if name.startswith('sat_'):
                            # QK=128 or 254: both saturate BEFORE sqrt(d) scaling.
                            for r in range(64):
                                q[r][0] = 2
                                k[r][0] = 64 if r % 2 == 0 else 127
                        if name == 'constant_v':
                            v = [[1] * dim for _ in range(64)]
                        expected = []
                        for r in range(64):
                            visible = r + 1 if causal else 64
                            row = []
                            for c in range(dim):
                                accum = sum((255 * sv * sum(v[j][c] for j in
                                    range(start, min(start + 16, visible)))) // 256
                                    for start in range(0, visible, 16))
                                row.append(trunc_div(accum, visible))
                            expected.append(row)
                        if name == 'constant_v':
                            self.assertEqual(expected, [[255] * dim for _ in range(64)])
                        scales = ''.join(f'scale_{n}_q88 = 0x{s & 65535:04X}\n'
                                         for n, s in zip(('q', 'k', 'v'), (sq, 256, sv)))
                        self.check_rtl(dim, q, k, v, expected, scales, causal)

    def test_layout_permutations(self):
        for dim in (16, 64):
            for name in ('query_rows', 'kv_rows_within_tiles', 'features', 'causal_features'):
                with self.subTest(dim=dim, case=name):
                    causal = name == 'causal_features'
                    data = (ROOT / 'data' / ('causal_N64' + ('_d64' if dim == 64 else ''))
                            if causal else fixture(dim))
                    q, k, v = [read_matrix(data / f'{x}_input.hex', dim, 8) for x in ('q', 'k', 'v')]
                    expected = read_matrix(data / 'expected.hex', dim, 32)
                    rng = random.Random(30930)
                    if name == 'query_rows':
                        order = list(range(64)); rng.shuffle(order)
                        q, expected = ([matrix[i] for i in order] for matrix in (q, expected))
                    elif name == 'kv_rows_within_tiles':
                        order = []
                        # Across-tile permutations need not be bit-identical:
                        # running-max rescale and truncation are order dependent.
                        for base in range(0, 64, 16):
                            block = list(range(base, base + 16)); rng.shuffle(block)
                            order.extend(block)
                        k, v = ([matrix[i] for i in order] for matrix in (k, v))
                    else:
                        order = list(range(dim)); rng.shuffle(order)
                        q, k, v, expected = ([[row[i] for i in order] for row in matrix]
                                             for matrix in (q, k, v, expected))
                    self.check_rtl(dim, q, k, v, expected,
                                   (data / 'scales.txt').read_text(), causal)


if __name__ == '__main__':
    unittest.main()
