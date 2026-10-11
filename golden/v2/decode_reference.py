"""Independent scalar fixed-point oracle for the M2B d16/L16 slice.

No RTL, loader indices, numpy matmul, padded rows or behavioral DUT calculation.
Only the immutable exponential LUT is shared with V1. Other lengths/positions
are rejected rather than silently extending the supported arithmetic scope.
"""
from pathlib import Path


def signed(value, bits):
    value = int(value) & ((1 << bits) - 1)
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def trunc_div(numer, denom):
    if denom <= 0:
        raise ValueError('normalization requires a positive denominator')
    return -(abs(numer) // denom) if numer < 0 else numer // denom


def dequant_score(acc, sq, sk):
    combined = signed(signed(sq, 16) * signed(sk, 16), 32)
    product = signed(signed(acc, 32) * combined, 64)
    rounded = signed(product + 128, 64) >> 8
    return max(-32768, min(32767, rounded)) >> 2


def lut_address(score, maximum):
    difference = signed(score - maximum, 17)
    return max(0, min(255, (difference * 255 + 523264) >> 11))


def load_lut(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[2]/'data/exp_lut.hex'
    values = [int(token, 16) for token in path.read_text().split()]
    if len(values) != 256 or values[0] != 0 or values[255] != 256:
        raise ValueError('unexpected immutable LUT geometry/endpoints')
    return values


def decode(q, k, v, scales, query_position=15, causal=False, lut=None):
    if len(q) != 16 or len(k) != 16 or len(v) != 16 or query_position != 15:
        raise ValueError('M2B oracle supports only d16/L16, absolute position15')
    if any(len(row) != 16 for row in list(k)+list(v)):
        raise ValueError('invalid K/V dimension')
    if len(scales) != 3:
        raise ValueError('three scale encodings required')
    if any(int(x) < -128 or int(x) > 127 for x in list(q)+[x for row in list(k)+list(v) for x in row]):
        raise ValueError('Q/K/V must be signed INT8')
    sq, sk, sv = [signed(x, 16) for x in scales]
    lut = load_lut() if lut is None else lut
    # At position15 all sixteen keys are eligible for either causal setting.
    scores = []
    for token in range(16):
        acc = 0
        for feature in range(16):
            acc = signed(acc + int(q[feature]) * int(k[token][feature]), 32)
        scores.append(dequant_score(acc, sq, sk))
    maximum = max(scores)
    raw = [lut[lut_address(score, maximum)] for score in scores]
    denominator = sum(raw) & 0xffffffff  # first-tile old sum is exactly zero
    result = []
    for feature in range(16):
        pv = 0
        for token in range(16):
            pv = signed(pv + min(raw[token], 255) * int(v[token][feature]), 32)
        wide = signed(pv * sv, 48)
        first_write = signed(wide >> 8, 32)
        numer = signed(first_write << 8, 40)
        result.append(signed(trunc_div(numer, denominator), 32))
    return result
