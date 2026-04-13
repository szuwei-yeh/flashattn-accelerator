"""
generate_kv_cache_test.py — Generate KV cache test vectors for Week 6.

Produces two sets of HW-accurate test data in data/kv_decode/:

  Prefill (N=32 tokens, d=16):
    q_prefill.hex / k_prefill.hex / v_prefill.hex  — INT8, 1 byte/line
    scales.txt                                      — Q8.8 scale factors
    expected_prefill.hex                            — INT32, 1 word/line (N*d entries)

  Decode (1 new query token attends to the 32 prefill K/V tokens):
    q_decode.hex        — INT8, 16 bytes (1 token)
    expected_decode.hex — INT32, 16 words (row 0 of attention output)

HW-accurate path mirrors generate_hw_expected.py:
  1. QK^T (INT8×INT8→INT32)
  2. Dequantise: (QKT * sq_q88 * sk_q88 + 128) >> 8   → Q8.8
  3. Scale by 1/sqrt(16) = >>2
  4. Per-tile independent softmax via exp LUT
  5. P_int8 @ V_int8 → INT32
  6. Scale by sv_q88: (PV * sv_q88) >> 8
  7. Accumulate across KV tiles

Usage (from project root):
    python golden/generate_kv_cache_test.py --out data/kv_decode
"""

import numpy as np
import argparse
import os
import sys

TILE_SIZE = 16
N_PREFILL = 32
HEAD_DIM  = 16


# ------------------------------------------------------------------ #
#  Exp LUT helpers (identical to generate_hw_expected.py)            #
# ------------------------------------------------------------------ #

def load_exp_lut(path: str):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f]


def to_addr(score_q88: int, maxv_q88: int) -> int:
    diff   = int(score_q88) - int(maxv_q88)
    numer  = diff * 255 + 523264
    result = numer >> 11
    return max(0, min(255, result))


def hw_softmax_tile(scores_q88: list, lut: list) -> list:
    """Per-tile independent softmax (tile_start=tile_last=1 always)."""
    tile_max    = max(scores_q88)
    exp_vals    = [lut[to_addr(s, tile_max)] for s in scores_q88]
    running_sum = sum(exp_vals)
    norm_denom  = max(running_sum, 1)
    result = []
    for ev in exp_vals:
        norm_val = ((ev << 8) + (norm_denom >> 1)) // norm_denom
        norm_val = min(norm_val, 0xFFFF)
        low = norm_val & 0xFF
        if low >= 128:
            low -= 256          # interpret as signed INT8
        result.append(low)
    return result


# ------------------------------------------------------------------ #
#  Quantisation helpers                                               #
# ------------------------------------------------------------------ #

def quantize_symmetric_q88(x: np.ndarray):
    """Return (int8_array, float_scale, q88_int_scale)."""
    max_val = float(np.max(np.abs(x)))
    if max_val == 0.0:
        return np.zeros_like(x, dtype=np.int8), 1.0, 256
    q_max     = 127
    scale     = max_val / q_max
    x_int     = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    scale_q88 = max(1, min(int(round(scale * 256)), 0x7FFF))
    return x_int, scale, scale_q88


# ------------------------------------------------------------------ #
#  HW-accurate attention for one Q tile over all KV tiles            #
# ------------------------------------------------------------------ #

def hw_flash_attn_q_tile(Q_tile: np.ndarray,
                          K_all:  np.ndarray,
                          V_all:  np.ndarray,
                          sq_q88: int,
                          sk_q88: int,
                          sv_q88: int,
                          lut:    list) -> np.ndarray:
    """
    Q_tile : (TILE_SIZE, HEAD_DIM)  INT8
    K_all  : (N_KV,      HEAD_DIM)  INT8   N_KV must be a multiple of TILE_SIZE
    V_all  : (N_KV,      HEAD_DIM)  INT8
    Returns: (TILE_SIZE, HEAD_DIM)  INT64 accumulated output
    """
    ts           = Q_tile.shape[0]
    num_kv_tiles = K_all.shape[0] // TILE_SIZE
    output       = np.zeros((ts, HEAD_DIM), dtype=np.int64)

    for j in range(num_kv_tiles):
        ks     = j * TILE_SIZE
        K_tile = K_all[ks : ks + TILE_SIZE]   # (TS, d)
        V_tile = V_all[ks : ks + TILE_SIZE]   # (TS, d)

        # 1. QK^T
        QKT = Q_tile.astype(np.int32) @ K_tile.astype(np.int32).T   # (ts, TS)

        # 2. Dequantise
        score_q88 = ((QKT.astype(np.int64) * sq_q88 * sk_q88) + 128) >> 8

        # 3. 1/sqrt(16) = >>2
        score_sc = score_q88 >> 2

        # 4. Per-tile softmax
        P_int8 = np.zeros((ts, TILE_SIZE), dtype=np.int8)
        for r in range(ts):
            row = [int(score_sc[r, c]) for c in range(TILE_SIZE)]
            P_int8[r] = np.array(hw_softmax_tile(row, lut), dtype=np.int8)

        # 5. PV
        PV = P_int8.astype(np.int32) @ V_tile.astype(np.int32)   # (ts, d)

        # 6. Scale
        PV_scaled = ((PV.astype(np.int64) * sv_q88) >> 8).astype(np.int32)

        # 7. Accumulate
        output += PV_scaled

    return output


# ------------------------------------------------------------------ #
#  Main generation function                                           #
# ------------------------------------------------------------------ #

def generate(out_dir: str, exp_lut_path: str, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    lut = load_exp_lut(exp_lut_path)
    rng = np.random.default_rng(seed)

    # ---------- Generate random FP32 matrices --------------------
    Q_pf_fp  = rng.standard_normal((N_PREFILL, HEAD_DIM)).astype(np.float32)
    K_pf_fp  = rng.standard_normal((N_PREFILL, HEAD_DIM)).astype(np.float32)
    V_pf_fp  = rng.standard_normal((N_PREFILL, HEAD_DIM)).astype(np.float32)
    Q_dec_fp = rng.standard_normal((1,         HEAD_DIM)).astype(np.float32)

    # ---------- Quantise to INT8 ---------------------------------
    Q_pf_i8,  _, sq_q88 = quantize_symmetric_q88(Q_pf_fp)
    K_pf_i8,  _, sk_q88 = quantize_symmetric_q88(K_pf_fp)
    V_pf_i8,  _, sv_q88 = quantize_symmetric_q88(V_pf_fp)
    Q_dec_i8, _, _      = quantize_symmetric_q88(Q_dec_fp)

    print(f"N_prefill={N_PREFILL}  HEAD_DIM={HEAD_DIM}  seed={seed}")
    print(f"sq=0x{sq_q88:04X}  sk=0x{sk_q88:04X}  sv=0x{sv_q88:04X}")

    # ---------- Prefill expected output (HW-accurate) ------------
    num_q_tiles = N_PREFILL // TILE_SIZE
    output_pf   = np.zeros((N_PREFILL, HEAD_DIM), dtype=np.int64)

    for i in range(num_q_tiles):
        qs     = i * TILE_SIZE
        Q_tile = Q_pf_i8[qs : qs + TILE_SIZE]
        tile_out = hw_flash_attn_q_tile(Q_tile, K_pf_i8, V_pf_i8,
                                        sq_q88, sk_q88, sv_q88, lut)
        output_pf[qs : qs + TILE_SIZE] = tile_out

    # ---------- Decode expected output (HW-accurate) -------------
    # Hardware sees Q tile padded to TILE_SIZE rows:
    #   row 0  = Q_dec (real query token)
    #   rows 1-15 = zero (no other query tokens in decode step)
    Q_dec_tile = np.zeros((TILE_SIZE, HEAD_DIM), dtype=np.int8)
    Q_dec_tile[0] = Q_dec_i8[0]

    decode_out      = hw_flash_attn_q_tile(Q_dec_tile, K_pf_i8, V_pf_i8,
                                           sq_q88, sk_q88, sv_q88, lut)
    expected_dec_r0 = decode_out[0]   # shape (HEAD_DIM,), only row 0 matters

    # ---------- Write hex files ----------------------------------
    def write_int8_hex(path, mat):
        with open(path, 'w') as f:
            for v in mat.flatten():
                f.write(f'{int(v) & 0xFF:02X}\n')
        print(f"  {path}  ({mat.size} bytes)")

    def write_int32_hex(path, arr):
        with open(path, 'w') as f:
            for v in np.array(arr).flatten():
                f.write(f'{int(v) & 0xFFFFFFFF:08X}\n')
        print(f"  {path}  ({np.array(arr).size} words)")

    print("\nWriting files:")
    write_int8_hex(os.path.join(out_dir, 'q_prefill.hex'), Q_pf_i8)
    write_int8_hex(os.path.join(out_dir, 'k_prefill.hex'), K_pf_i8)
    write_int8_hex(os.path.join(out_dir, 'v_prefill.hex'), V_pf_i8)
    write_int8_hex(os.path.join(out_dir, 'q_decode.hex'),  Q_dec_i8)
    write_int32_hex(os.path.join(out_dir, 'expected_prefill.hex'), output_pf)
    write_int32_hex(os.path.join(out_dir, 'expected_decode.hex'),  expected_dec_r0)

    with open(os.path.join(out_dir, 'scales.txt'), 'w') as f:
        f.write(f'scale_q_q88 = 0x{sq_q88:04X}\n')
        f.write(f'scale_k_q88 = 0x{sk_q88:04X}\n')
        f.write(f'scale_v_q88 = 0x{sv_q88:04X}\n')
        f.write(f'N           = {N_PREFILL}\n')
        f.write(f'd           = {HEAD_DIM}\n')
        f.write(f'seed        = {seed}\n')
    print(f"  {os.path.join(out_dir, 'scales.txt')}")

    # ---------- Sanity print -------------------------------------
    print(f"\nPrefill output[0, :8] = {output_pf[0, :8].tolist()}")
    print(f"Decode  output[0, :8] = {expected_dec_r0[:8].tolist()}")
    print("\nDone — files written to:", out_dir)


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',     type=str, default='data/kv_decode',
                        help='Output directory')
    parser.add_argument('--exp_lut', type=str, default='data/exp_lut.hex',
                        help='Path to exp_lut.hex (relative to project root)')
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.exp_lut, args.seed)
