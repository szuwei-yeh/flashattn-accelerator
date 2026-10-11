"""Scalar logical-row oracle; tile order and widths follow frozen V1 arithmetic.

No RTL-derived outputs, padded square tensors or floating-point attention.
Immutable M2B scalar width/LUT helpers are reused; partial tokens are omitted.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_reference import signed, trunc_div, dequant_score, lut_address, load_lut


def decode_with_trace(q, k, v, scales, query_position=None, causal=False, lut=None):
    length = len(k)
    if len(q) != 16 or not 1 <= length <= 64 or len(v) != length:
        raise ValueError('d16, one query and 1..64 committed tokens required')
    query_position = length - 1 if query_position is None else query_position
    if query_position != length - 1:
        raise ValueError('only latest absolute query position is supported')
    if any(len(row) != 16 for row in list(k)+list(v)) or len(scales) != 3:
        raise ValueError('invalid vector/scale dimensions')
    if any(not -128 <= int(x) <= 127 for x in list(q)+[x for row in list(k)+list(v) for x in row]):
        raise ValueError('Q/K/V must be signed INT8')
    sq, sk, sv = [signed(x, 16) for x in scales]
    lut = load_lut() if lut is None else lut
    maximum, denominator, output = -32768, 0, [0]*16
    trace = []
    for base in range(0, length, 16):
        scores = []
        for token in range(base, min(base+16, length)):
            acc = 0
            for feature in range(16):
                acc = signed(acc + int(q[feature])*int(k[token][feature]), 32)
            scores.append(dequant_score(acc, sq, sk))
        tile_max = max(scores)  # contains eligible tokens only
        new_max = tile_max if base == 0 else max(maximum, tile_max)
        factor = lut[lut_address(maximum, new_max)]
        if base == 0:
            denominator = 0
        denominator = ((denominator*factor+128) >> 8) & 0xffffffff
        maximum = new_max
        raw = [lut[lut_address(score, maximum)] for score in scores]
        denominator = (denominator+sum(raw)) & 0xffffffff
        p = [min(value, 255) for value in raw]
        for feature in range(16):
            pv = 0
            for local, token in enumerate(range(base, min(base+16, length))):
                pv = signed(pv+p[local]*int(v[token][feature]), 32)
            pv_scaled = signed(signed(pv*sv, 48) >> 8, 32)
            prior = signed(signed(output[feature]*factor, 48) >> 8, 32)
            output[feature] = pv_scaled if base == 0 else signed(prior+pv_scaled, 32)
        trace.append({'base':base,'valid_keys':len(scores),'max':maximum,'sum':denominator,
                      'factor':factor,'p':p+[0]*(16-len(p)),'scores':scores,'output_accum':list(output)})
    return [signed(trunc_div(signed(word << 8, 40), denominator), 32) for word in output], trace


def decode(q, k, v, scales, query_position=None, causal=False, lut=None):
    return decode_with_trace(q,k,v,scales,query_position,causal,lut)[0]
