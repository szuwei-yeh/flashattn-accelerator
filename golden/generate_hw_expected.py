"""
generate_hw_expected.py — Hardware-accurate expected.hex for any N.

Simulates TRUE cross-tile online FlashAttention softmax (v4):

  Per Q-tile outer loop:
    Per KV-tile inner loop:
      1. QK^T matmul  (INT8×INT8→INT32)
      2. Dequantize:  (QKT * sq_q88 * sk_q88 + 128) >> 8, saturate to INT16
      3. Scale 1/√d:  >> 2 for d=16; >> 3 for d=64
      4. Online softmax per Q-row:
           S_FIND_MAX:      tile_max = max(scores)
           S_RESCALE:       next_max = max(running_max, tile_max)
                            lut_addr = to_addr(running_max_old, next_max)
                            rescale_q88 = lut[lut_addr]   ← exp(m_old−m_new)
                            if first tile: running_sum = 0
           S_RESCALE_WAIT:  running_sum = (running_sum × rescale_q88 + 128) >> 8
                            running_max = next_max
           S_ACCUM:         exp_vals[i] = lut[to_addr(sc[i], running_max)]
                            running_sum += sum(exp_vals)
      5. S_RESCALE_OUTPUT: output_accum *= rescale_q88 / 256  (per row, signed)
      6. P_uint8:          exp_flat capped at 0xFF; unsigned 8-bit PV operand
      7. PV matmul:        P_uint8 @ V_tile  → INT32
      8. Scale PV:         (PV × sv_q88) >> 8
      9. Accumulate:       output_accum += PV_scaled
    End inner loop
    S_NORMALIZE:  output[r,c] = (output_accum[r,c] << 8) // running_sum[r]
                                (signed truncation toward zero, like Verilog /)
  End outer loop

Output: expected.hex  (INT32, 8 hex chars per entry, row-major)

Usage:
    python golden/generate_hw_expected.py --data data
    python golden/generate_hw_expected.py --data data/N64
    python golden/generate_hw_expected.py --data data/N128
    python golden/generate_hw_expected.py --data data/N256
"""

import numpy as np
import argparse
import os
import re


# ── helpers ───────────────────────────────────────────────────────────────────

def load_int8_hex(path, n):
    with open(path) as f:
        tokens = f.read().split()
    if n <= 0 or len(tokens) != n or any(
            re.fullmatch(r'[0-9a-fA-F]{1,2}', token) is None for token in tokens):
        raise ValueError(f'{path}: expected exactly {n} unsigned byte hex tokens')
    vals = [int(token, 16) for token in tokens]
    return np.array([v - 256 if v >= 128 else v for v in vals], dtype=np.int8)


def load_scales(path):
    scales = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                if k.strip() in scales:
                    raise ValueError(f'{path}: duplicate key {k.strip()}')
                scales[k.strip()] = v.strip()
    for key in ('scale_q_q88', 'scale_k_q88', 'scale_v_q88'):
        if re.fullmatch(r'0x[0-9a-fA-F]{1,4}', scales.get(key, '')) is None:
            raise ValueError(f'{path}: missing or invalid {key}')
    sq_q88 = signed16(int(scales['scale_q_q88'], 16))
    sk_q88 = signed16(int(scales['scale_k_q88'], 16))
    sv_q88 = signed16(int(scales['scale_v_q88'], 16))
    N = int(scales['N'])
    d = int(scales['d'])
    return sq_q88, sk_q88, sv_q88, N, d


def load_exp_lut(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f]


def to_addr(score_q88, maxv_q88):
    """Exact replication of hardware to_addr() → LUT index for exp(score−maxv)."""
    diff   = int(score_q88) - int(maxv_q88)
    numer  = diff * 255 + 523264          # 255 * 2048 + 1024 (round half up)
    result = numer >> 11                  # arithmetic (Python int) right-shift
    if result > 255: return 255
    if result < 0:   return 0
    return result


def signed16(x):
    """Interpret a scale port's 16 bits as signed Q8.8."""
    x = int(x) & 0xFFFF
    return x - 0x10000 if x >= 0x8000 else x


def int32(x):
    """Truncate Python int to 32-bit two's-complement signed."""
    x = int(x) & 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def signed_div_trunc(numer, denom):
    """Signed integer division, truncation toward zero (matches Verilog '/')."""
    if denom == 0:
        return int(numer)
    sign = -1 if (numer < 0) != (denom < 0) else 1
    return sign * (abs(int(numer)) // abs(int(denom)))


# ── per-Q-tile FlashAttention ─────────────────────────────────────────────────

def flash_attn_q_tile(Q_tile, K, V, sq_q88, sk_q88, sv_q88, lut, tile_size,
                      causal=False, q_tile_idx=0):
    """
    Simulate hardware FlashAttention for one Q-tile with true cross-tile
    online softmax.  Returns output array (tile_size × d) as Python int list.

    causal    : when True, apply causal masking (upper-triangle → -inf).
    q_tile_idx: Q-tile index (tile_row // tile_size), used for causal skip.
    """
    d            = Q_tile.shape[1]
    sq_q88, sk_q88, sv_q88 = map(signed16, (sq_q88, sk_q88, sv_q88))
    num_kv_tiles = K.shape[0] // tile_size

    # Per Q-row running state (match hardware reset values)
    #   running_max: 16'sh8000 = -32768
    running_max = [-32768] * tile_size
    running_sum = [0]      * tile_size

    # First K/V tile overwrites output state; SRAM itself has no reset.
    output_accum = [[0] * d for _ in range(tile_size)]

    for j in range(num_kv_tiles):
        # ── Causal: skip above-diagonal tiles entirely (matches tile_controller) ──
        if causal and j > q_tile_idx:
            continue

        is_first = (j == 0)
        kvs = j * tile_size
        kve = kvs + tile_size
        K_tile = K[kvs:kve]
        V_tile = V[kvs:kve]

        # ── 1-3. QK^T → dequantize → 1/√d scale ─────────────────────────────
        # scale_shift = log2(sqrt(d)) = clog2(d)/2  (d must be power-of-2 squared)
        scale_shift = d.bit_length() // 2   # d=16→2, d=64→3
        QKT         = Q_tile.astype(np.int32) @ K_tile.astype(np.int32).T  # [TS,TS]
        scores_q88  = ((QKT.astype(np.int64) * sq_q88 * sk_q88) + 128) >> 8
        scores_q88  = np.clip(scores_q88, -32768, 32767)  # dequantizer saturation
        scores_sc   = scores_q88 >> scale_shift                             # ÷√d

        # ── Causal: mask upper triangle of diagonal tile ──────────────────────
        # Masked value = -32768 (0x8000 in Q8.8) → exp LUT index 0 ≈ 0.
        if causal and j == q_tile_idx:
            for r in range(tile_size):
                for c in range(tile_size):
                    if c > r:
                        scores_sc[r, c] = -32768

        # ── 4. Online softmax per Q-row (matches online_softmax FSM) ─────────
        rescale_q88_row = [0] * tile_size  # exp(m_old − m_new), one per Q-row
        exp_flat        = [[0] * tile_size for _ in range(tile_size)]

        for r in range(tile_size):
            row = [int(scores_sc[r, c]) for c in range(tile_size)]

            # S_FIND_MAX
            tile_max_r = max(row)

            # S_RESCALE: compute next_max and LUT address using OLD running_max
            if is_first:
                next_max_r = tile_max_r
            else:
                next_max_r = max(running_max[r], tile_max_r)

            lut_addr = to_addr(running_max[r], next_max_r)
            rq88_r   = lut[lut_addr]   # exp(m_old − m_new) in Q8.8
            rescale_q88_row[r] = rq88_r

            if is_first:
                running_sum[r] = 0          # S_RESCALE resets sum on first tile

            # S_RESCALE_WAIT: rescale running_sum (note: =0 for first tile above)
            rescale_prod    = running_sum[r] * rq88_r
            running_sum[r]  = (rescale_prod + 128) >> 8
            running_max[r]  = next_max_r

            # S_ACCUM: compute exp values and accumulate running_sum
            tile_exp_sum = 0
            for c in range(tile_size):
                ev = lut[to_addr(row[c], running_max[r])]
                exp_flat[r][c] = ev
                tile_exp_sum  += ev

            running_sum[r] += tile_exp_sum

        # ── 5. S_RESCALE_OUTPUT: rescale output_accum per Q-row ──────────────
        #    Hardware: new = $signed(sign_ext48(old) * zero_ext48(rq88)) >> 8
        for r in range(tile_size):
            rq88 = rescale_q88_row[r]
            for c in range(d):
                wide = int(output_accum[r][c]) * int(rq88)   # signed × unsigned
                output_accum[r][c] = int32(wide >> 8)

        # ── 6. P_uint8 from exp_flat: cap at 0xFF, UNSIGNED [0,255] ──────────
        #   P is a softmax weight (always positive). The PE zero-extends it
        #   (a_unsigned=1 in PV phase), so it must NOT be sign-wrapped here.
        P_uint8 = np.zeros((tile_size, tile_size), dtype=np.int32)
        for r in range(tile_size):
            for c in range(tile_size):
                P_uint8[r, c] = min(int(exp_flat[r][c]), 0xFF)   # cap as hardware does

        # ── 7-8. PV matmul + sv_q88 scale ────────────────────────────────────
        PV        = P_uint8.astype(np.int32) @ V_tile.astype(np.int32)     # [TS,d]
        PV_scaled = (PV.astype(np.int64) * sv_q88) >> 8

        # ── 9. Accumulate into output buffer ─────────────────────────────────
        for r in range(tile_size):
            for c in range(d):
                output_accum[r][c] = int32(output_accum[r][c] + int(PV_scaled[r, c]))

    # ── S_NORMALIZE: divide each element by l_global (running_sum per row) ───
    #    Hardware: (old << 8) / running_sum  — signed truncation toward zero
    output_tile = [[0] * d for _ in range(tile_size)]
    for r in range(tile_size):
        l_global = running_sum[r] if running_sum[r] != 0 else 1
        for c in range(d):
            numer = int(output_accum[r][c]) << 8    # 40-bit equivalent
            output_tile[r][c] = int32(signed_div_trunc(numer, l_global))

    return output_tile


# ── top-level generate ────────────────────────────────────────────────────────

def generate(data_dir, exp_lut_path, tile_size=16, causal=False):
    sq_q88, sk_q88, sv_q88, N, d = load_scales(
        os.path.join(data_dir, 'scales.txt'))
    if tile_size != 16 or d not in (16, 64) or N <= 0 or N % tile_size:
        raise ValueError('Expected tile_size=16, d=16/64 and positive tile-aligned N')
    lut = load_exp_lut(exp_lut_path)

    Q = load_int8_hex(os.path.join(data_dir, 'q_input.hex'), N * d).reshape(N, d)
    K = load_int8_hex(os.path.join(data_dir, 'k_input.hex'), N * d).reshape(N, d)
    V = load_int8_hex(os.path.join(data_dir, 'v_input.hex'), N * d).reshape(N, d)

    num_q_tiles = N // tile_size
    print(f"N={N} d={d} tile_size={tile_size} causal={causal}")
    print(f"sq_q88=0x{sq_q88:04X} sk_q88=0x{sk_q88:04X} sv_q88=0x{sv_q88:04X}")
    print(f"Q tiles={num_q_tiles}  KV tiles per Q-tile={N // tile_size}")

    output = [[0] * d for _ in range(N)]

    for i in range(num_q_tiles):
        qs = i * tile_size
        qe = qs + tile_size
        tile_out = flash_attn_q_tile(
            Q[qs:qe], K, V, sq_q88, sk_q88, sv_q88, lut, tile_size,
            causal=causal, q_tile_idx=i)
        for r in range(tile_size):
            output[qs + r] = tile_out[r]

        if (i + 1) % max(1, num_q_tiles // 4) == 0:
            print(f"  Q tile {i+1}/{num_q_tiles} done")

    # Write expected.hex
    out_path = os.path.join(data_dir, 'expected.hex')
    with open(out_path, 'w') as f:
        for row in output:
            for val in row:
                f.write(f'{int32(val) & 0xFFFFFFFF:08X}\n')

    print(f"Written: {out_path}  ({N * d} entries)")
    print(f"output[0,:8] = {[int32(v) for v in output[0][:8]]}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      type=str, required=True,
                        help='Directory with q/k/v_input.hex and scales.txt')
    parser.add_argument('--tile_size', type=int, default=16)
    parser.add_argument('--exp_lut',   type=str, default='data/exp_lut.hex',
                        help='Path to exp_lut.hex (relative to project root)')
    parser.add_argument('--causal',    action='store_true',
                        help='Generate causal (decoder) masked expected output')
    args = parser.parse_args()

    generate(args.data, args.exp_lut, args.tile_size, causal=args.causal)
