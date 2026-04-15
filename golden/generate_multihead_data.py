"""
generate_multihead_data.py — Generate test data for multi-head AXI FlashAttention.

Supports both MHA (Grouped Query Attention with ratio=1) and GQA (ratio>1).

MHA (default, --num_kv_heads == --num_heads):
  Q/K/V all have shape [N, total_head_dim].
  Each Q-head attends to its own K/V-head.

GQA (--num_kv_heads < --num_heads, e.g. 4 Q-heads, 2 KV-heads):
  Q has shape [N, num_heads * per_head_dim].
  K/V have shape [N, num_kv_heads * per_head_dim] (smaller — the GQA bandwidth saving).
  Q-head h attends to KV-head (h // gqa_ratio).

Outputs in <out_dir>/:
  q_input.hex          N×(num_heads*PHD) INT8, row-major, one byte per line
  k_input.hex          N×(num_kv_heads*PHD) INT8  (= N×total_head_dim for MHA)
  v_input.hex          N×(num_kv_heads*PHD) INT8
  scales.txt           scale_q_q88, scale_k_q88, scale_v_q88, N, d, seed, gqa_ratio
  expected_h0.hex      N×PHD INT32 expected for Q-head 0 (HW-accurate)
  expected_h1.hex      ...
  ...

Usage:
  python golden/generate_multihead_data.py --N 64 --seed 42 --out data/N64_axi
  python golden/generate_multihead_data.py --N 64 --num_kv_heads 2 --out data/N64_gqa
"""

import numpy as np
import argparse
import os
import sys

# Reuse generate_hw_expected logic (true cross-tile FlashAttention)
sys.path.insert(0, os.path.dirname(__file__))
from generate_hw_expected import flash_attn_q_tile, int32, load_exp_lut


def quantize_symmetric_q88(x: np.ndarray, bits: int = 8):
    """Quantize to INT8, return (int8_array, float_scale, q88_scale)."""
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0, 256
    q_max = (1 << (bits - 1)) - 1
    scale = max_val / q_max
    x_int = np.clip(np.round(x / scale), -q_max - 1, q_max).astype(np.int8)
    scale_q88 = int(round(scale * 256))
    scale_q88 = max(1, min(scale_q88, 0x7FFF))
    return x_int, scale, scale_q88


def hw_expected_for_head(Q_h, K_h, V_h, sq_q88, sk_q88, sv_q88, lut, tile_size=16):
    """
    Compute HW-accurate expected output for one head using true cross-tile
    online FlashAttention (running_max/running_sum carry across KV tiles).
    Q_h, K_h, V_h: int8 numpy arrays of shape [N, per_head_dim]
    tile_size: sequence-dimension tile (= TILE_SIZE = 16, independent of per_head_dim)
    Returns: int64 numpy array of shape [N, per_head_dim]
    """
    N, d = Q_h.shape
    num_q_tiles = N // tile_size
    all_rows = []
    for i in range(num_q_tiles):
        qs = i * tile_size
        tile_out = flash_attn_q_tile(Q_h[qs:qs + tile_size], K_h, V_h,
                                     sq_q88, sk_q88, sv_q88, lut, tile_size)
        all_rows.extend(tile_out)
    return np.array([[int32(v) for v in row] for row in all_rows], dtype=np.int64)


def generate(N: int, total_head_dim: int, num_heads: int, seed: int, out_dir: str,
             exp_lut_path: str, num_kv_heads: int = None):
    """
    num_kv_heads: number of KV heads (GQA). None or equal to num_heads → MHA.
    For GQA: K/V are [N, num_kv_heads * per_head_dim]; Q is [N, total_head_dim].
    Q-head h attends to KV-head (h // gqa_ratio).
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    gqa_ratio = num_heads // num_kv_heads
    per_head_dim = total_head_dim // num_heads
    kv_head_dim  = num_kv_heads * per_head_dim

    rng = np.random.default_rng(seed)

    # Generate FP32 matrices
    Q_fp = rng.standard_normal((N, total_head_dim)).astype(np.float32)
    K_fp = rng.standard_normal((N, kv_head_dim)).astype(np.float32)
    V_fp = rng.standard_normal((N, kv_head_dim)).astype(np.float32)

    # Quantize with a single scale per matrix
    Q_i8, sq, sq_q88 = quantize_symmetric_q88(Q_fp)
    K_i8, sk, sk_q88 = quantize_symmetric_q88(K_fp)
    V_i8, sv, sv_q88 = quantize_symmetric_q88(V_fp)

    print(f"N={N}  total_head_dim={total_head_dim}  num_heads={num_heads}"
          f"  per_head_dim={per_head_dim}")
    print(f"num_kv_heads={num_kv_heads}  gqa_ratio={gqa_ratio}"
          f"  kv_head_dim={kv_head_dim}")
    print(f"Q  scale: {sq:.6f}  (Q8.8 = 0x{sq_q88:04X})")
    print(f"K  scale: {sk:.6f}  (Q8.8 = 0x{sk_q88:04X})")
    print(f"V  scale: {sv:.6f}  (Q8.8 = 0x{sv_q88:04X})")

    os.makedirs(out_dir, exist_ok=True)

    def write_int8_hex(path, mat):
        with open(path, 'w') as f:
            for val in mat.flatten():
                f.write(f"{int(val) & 0xFF:02X}\n")
        print(f"Written: {path}  ({mat.size} entries)")

    # Q is always full NUM_HEADS wide; K/V may be narrower for GQA
    write_int8_hex(os.path.join(out_dir, 'q_input.hex'), Q_i8)
    write_int8_hex(os.path.join(out_dir, 'k_input.hex'), K_i8)
    write_int8_hex(os.path.join(out_dir, 'v_input.hex'), V_i8)

    scales_path = os.path.join(out_dir, 'scales.txt')
    with open(scales_path, 'w') as f:
        f.write(f"scale_q_q88 = 0x{sq_q88:04X}\n")
        f.write(f"scale_k_q88 = 0x{sk_q88:04X}\n")
        f.write(f"scale_v_q88 = 0x{sv_q88:04X}\n")
        f.write(f"N           = {N}\n")
        f.write(f"d           = {per_head_dim}\n")
        f.write(f"seed        = {seed}\n")
        f.write(f"gqa_ratio   = {gqa_ratio}\n")
        f.write(f"num_kv_heads = {num_kv_heads}\n")
    print(f"Written: {scales_path}")

    # Load exp LUT
    lut = load_exp_lut(exp_lut_path)
    print(f"Loaded exp LUT: {len(lut)} entries from {exp_lut_path}")

    # Compute HW-accurate expected for each Q-head
    # Q-head h attends to KV-head (h // gqa_ratio)
    for h in range(num_heads):
        kv_h = h // gqa_ratio  # which KV-head this Q-head uses
        Q_h = Q_i8[:, h*per_head_dim : (h+1)*per_head_dim]
        K_h = K_i8[:, kv_h*per_head_dim : (kv_h+1)*per_head_dim]
        V_h = V_i8[:, kv_h*per_head_dim : (kv_h+1)*per_head_dim]

        print(f"\nComputing head {h} HW expected (N={N}, d={per_head_dim}"
              f", kv_head={kv_h})...")
        output = hw_expected_for_head(Q_h, K_h, V_h,
                                      sq_q88, sk_q88, sv_q88,
                                      lut, tile_size=16)  # TILE_SIZE always 16

        exp_path = os.path.join(out_dir, f'expected_h{h}.hex')
        with open(exp_path, 'w') as f:
            for val in output.flatten():
                f.write(f'{int(val) & 0xFFFFFFFF:08X}\n')
        print(f"Written: {exp_path}  ({N * per_head_dim} entries)")
        print(f"  output[0,:4] = {output[0, :4].tolist()}")

    print(f"\nDone. All files written to: {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',           type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--total_dim',   type=int, default=64,
                        help='Total Q head dimension (NUM_HEADS * PER_HEAD_DIM)')
    parser.add_argument('--num_heads',   type=int, default=4)
    parser.add_argument('--num_kv_heads',type=int, default=None,
                        help='Number of KV heads for GQA (default=num_heads = MHA)')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--out',         type=str, default='data/N64_axi',
                        help='Output directory')
    parser.add_argument('--exp_lut',     type=str, default='data/exp_lut.hex',
                        help='Path to exp_lut.hex')
    args = parser.parse_args()

    num_kv = args.num_kv_heads if args.num_kv_heads is not None else args.num_heads
    print(f"Generating multihead test data: N={args.N}, "
          f"total_dim={args.total_dim}, q_heads={args.num_heads}, "
          f"kv_heads={num_kv}, seed={args.seed}")
    print("=" * 60)
    generate(args.N, args.total_dim, args.num_heads, args.seed,
             args.out, args.exp_lut, num_kv_heads=num_kv)
