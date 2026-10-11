"""Scalar logical-row oracle; tile order and widths follow frozen V1 arithmetic.

No RTL-derived outputs, padded square tensors or floating-point attention.
Immutable M2B scalar width/LUT helpers are reused; partial tokens are omitted.
"""
from pathlib import Path
import sys
import importlib.util
_spec = importlib.util.spec_from_file_location("m2b_width_helpers", Path(__file__).resolve().parents[1] / "decode_reference.py")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
signed, trunc_div, lut_address, load_lut = (_helpers.signed, _helpers.trunc_div, _helpers.lut_address, _helpers.load_lut)


def decode_with_trace(q, k, v, scales, query_position=None, causal=False, lut=None):
    length = len(k)
    if len(q) not in (16, 64) or not 1 <= length <= 64 or len(v) != length:
        raise ValueError('d16/d64, one query and 1..64 committed tokens required')
    query_position = length - 1 if query_position is None else query_position
    if query_position != length - 1:
        raise ValueError('only latest absolute query position is supported')
    if any(len(row) != len(q) for row in list(k)+list(v)) or len(scales) != 3:
        raise ValueError('invalid vector/scale dimensions')
    if any(not -128 <= int(x) <= 127 for x in list(q)+[x for row in list(k)+list(v) for x in row]):
        raise ValueError('Q/K/V must be signed INT8')
    sq, sk, sv = [signed(x, 16) for x in scales]
    lut = load_lut() if lut is None else lut
    maximum, denominator, output = -32768, 0, [0]*len(q)
    trace = []
    for base in range(0, length, 16):
        scores = []
        for token in range(base, min(base+16, length)):
            acc = 0
            for feature in range(len(q)):
                acc = signed(acc + int(q[feature])*int(k[token][feature]), 32)
            combined = signed(sq*sk, 32)
            rounded = signed(signed(acc*combined, 64)+128, 64) >> 8
            scores.append(max(-32768, min(32767, rounded)) >> (2 if len(q) == 16 else 3))
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
        for feature in range(len(q)):
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


def prefill(q, k, v, scales, causal=False, lut=None):
    """N16 initial sequence, logical row arithmetic; no padded RTL tensors."""
    if len(q) != 16 or len(k) != 16 or len(v) != 16:
        raise ValueError('only N16 prefill is enabled')
    result = []
    for row, query in enumerate(q):
        length = row + 1 if causal else 16
        result.append(decode(query, k[:length], v[:length], scales, length-1, causal, lut))
    return result
