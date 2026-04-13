"""
generate_multihead_data.py — Generate test data for 4-head AXI FlashAttention.

Generates Q/K/V matrices of shape [N, HEAD_DIM] (= [N, 64]) with a single
quantization scale, then computes hardware-accurate expected outputs for each
of the 4 heads independently (each head processes [N, 16] sub-matrices).

Outputs in <out_dir>/:
  q_input.hex          N×64 INT8, row-major, one byte per line
  k_input.hex          N×64 INT8
  v_input.hex          N×64 INT8
  scales.txt           scale_q_q88, scale_k_q88, scale_v_q88, N, d=16, seed
  expected_h0.hex      N×16 INT32 expected for head 0 (HW-accurate)
  expected_h1.hex      N×16 INT32 expected for head 1
  expected_h2.hex      N×16 INT32 expected for head 2
  expected_h3.hex      N×16 INT32 expected for head 3

Usage:
  python golden/generate_multihead_data.py --N 64 --seed 42 --out data/N64_axi
"""

import numpy as np
import argparse
import os
import sys

# Reuse generate_hw_expected logic
sys.path.insert(0, os.path.dirname(__file__))


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


def load_exp_lut(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def to_addr(score_q88, maxv_q88):
    """Replicate hardware address computation for exp LUT lookup."""
    diff = int(score_q88) - int(maxv_q88)
    numer = diff * 255 + 523264
    result = numer >> 11
    if result > 255: return 255
    if result < 0:   return 0
    return result


def hw_softmax_tile(scores_q88, lut):
    """HW-accurate per-tile softmax (tile_start=tile_last=True always)."""
    # S_FIND_MAX
    tile_max = scores_q88[0]
    for s in scores_q88[1:]:
        if s > tile_max:
            tile_max = s

    running_max = tile_max
    exp_vals = [lut[to_addr(s, running_max)] for s in scores_q88]
    running_sum = sum(exp_vals)
    norm_denom = max(running_sum, 1)

    result = []
    for ev in exp_vals:
        norm_val = ((ev << 8) + (norm_denom >> 1)) // norm_denom
        if norm_val > 0xFFFF:
            norm_val = 0xFFFF
        low_byte = norm_val & 0xFF
        if low_byte >= 128:
            low_byte -= 256
        result.append(low_byte)
    return result


def hw_expected_for_head(Q_h, K_h, V_h, sq_q88, sk_q88, sv_q88, lut, tile_size=16):
    """
    Compute HW-accurate expected output for one head.
    Q_h, K_h, V_h: int8 numpy arrays of shape [N, per_head_dim]
    Returns: int64 numpy array of shape [N, per_head_dim]
    """
    N, d = Q_h.shape
    assert d == tile_size, f"per_head_dim {d} must equal tile_size {tile_size}"
    num_tiles = N // tile_size
    output = np.zeros((N, d), dtype=np.int64)

    for i in range(num_tiles):
        qs, qe = i * tile_size, (i + 1) * tile_size
        Q_tile = Q_h[qs:qe]

        for j in range(num_tiles):
            kvs, kve = j * tile_size, (j + 1) * tile_size
            K_tile = K_h[kvs:kve]
            V_tile = V_h[kvs:kve]

            # 1. QK^T matmul
            QKT = Q_tile.astype(np.int32) @ K_tile.astype(np.int32).T  # [TS, TS]

            # 2. Dequantize (SHIFT=8): (QKT * sq * sk + 128) >> 8
            score_q88 = ((QKT.astype(np.int64) * sq_q88 * sk_q88) + 128) >> 8

            # 3. Scale by 1/sqrt(d) = >>2 (d=16 → 0.25)
            score_scaled = score_q88 >> 2

            # 4. Per-tile independent softmax
            P_int8 = np.zeros((tile_size, tile_size), dtype=np.int8)
            for r in range(tile_size):
                row = [int(score_scaled[r, c]) for c in range(tile_size)]
                p_row = hw_softmax_tile(row, lut)
                P_int8[r] = np.array(p_row, dtype=np.int8)

            # 5. PV matmul
            PV = P_int8.astype(np.int32) @ V_tile.astype(np.int32)

            # 6. Scale by sv_q88: (PV * sv_q88) >> 8
            PV_scaled = ((PV.astype(np.int64) * sv_q88) >> 8).astype(np.int32)

            # 7. Accumulate
            output[qs:qe] += PV_scaled

    return output


def generate(N: int, total_head_dim: int, num_heads: int, seed: int, out_dir: str,
             exp_lut_path: str):
    per_head_dim = total_head_dim // num_heads
    assert per_head_dim == 16, f"per_head_dim must be 16, got {per_head_dim}"

    rng = np.random.default_rng(seed)

    # Generate FP32 matrices (N × total_head_dim)
    Q_fp = rng.standard_normal((N, total_head_dim)).astype(np.float32)
    K_fp = rng.standard_normal((N, total_head_dim)).astype(np.float32)
    V_fp = rng.standard_normal((N, total_head_dim)).astype(np.float32)

    # Quantize with a single scale per matrix
    Q_i8, sq, sq_q88 = quantize_symmetric_q88(Q_fp)
    K_i8, sk, sk_q88 = quantize_symmetric_q88(K_fp)
    V_i8, sv, sv_q88 = quantize_symmetric_q88(V_fp)

    print(f"N={N}  total_head_dim={total_head_dim}  num_heads={num_heads}"
          f"  per_head_dim={per_head_dim}")
    print(f"Q  scale: {sq:.6f}  (Q8.8 = 0x{sq_q88:04X})")
    print(f"K  scale: {sk:.6f}  (Q8.8 = 0x{sk_q88:04X})")
    print(f"V  scale: {sv:.6f}  (Q8.8 = 0x{sv_q88:04X})")

    os.makedirs(out_dir, exist_ok=True)

    # Write full N×64 matrices
    def write_int8_hex(path, mat):
        with open(path, 'w') as f:
            for val in mat.flatten():
                f.write(f"{int(val) & 0xFF:02X}\n")
        print(f"Written: {path}  ({mat.size} entries)")

    write_int8_hex(os.path.join(out_dir, 'q_input.hex'), Q_i8)
    write_int8_hex(os.path.join(out_dir, 'k_input.hex'), K_i8)
    write_int8_hex(os.path.join(out_dir, 'v_input.hex'), V_i8)

    # Write scales.txt (d=per_head_dim for each head's hw_expected computation)
    scales_path = os.path.join(out_dir, 'scales.txt')
    with open(scales_path, 'w') as f:
        f.write(f"scale_q_q88 = 0x{sq_q88:04X}\n")
        f.write(f"scale_k_q88 = 0x{sk_q88:04X}\n")
        f.write(f"scale_v_q88 = 0x{sv_q88:04X}\n")
        f.write(f"N           = {N}\n")
        f.write(f"d           = {per_head_dim}\n")
        f.write(f"seed        = {seed}\n")
    print(f"Written: {scales_path}")

    # Load exp LUT
    lut = load_exp_lut(exp_lut_path)
    print(f"Loaded exp LUT: {len(lut)} entries from {exp_lut_path}")

    # Compute HW-accurate expected for each head
    for h in range(num_heads):
        hs = h * per_head_dim
        he = hs + per_head_dim
        Q_h = Q_i8[:, hs:he]
        K_h = K_i8[:, hs:he]
        V_h = V_i8[:, hs:he]

        print(f"\nComputing head {h} HW expected (N={N}, d={per_head_dim})...")
        output = hw_expected_for_head(Q_h, K_h, V_h,
                                      sq_q88, sk_q88, sv_q88,
                                      lut, tile_size=per_head_dim)

        exp_path = os.path.join(out_dir, f'expected_h{h}.hex')
        with open(exp_path, 'w') as f:
            for val in output.flatten():
                f.write(f'{int(val) & 0xFFFFFFFF:08X}\n')
        print(f"Written: {exp_path}  ({N * per_head_dim} entries)")
        print(f"  output[0,:4] = {output[0, :4].tolist()}")

    print(f"\nDone. All files written to: {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',         type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--total_dim', type=int, default=64,
                        help='Total head dimension (NUM_HEADS * PER_HEAD_DIM)')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--out',       type=str, default='data/N64_axi',
                        help='Output directory')
    parser.add_argument('--exp_lut',   type=str, default='data/exp_lut.hex',
                        help='Path to exp_lut.hex')
    args = parser.parse_args()

    print(f"Generating multihead test data: N={args.N}, "
          f"total_dim={args.total_dim}, heads={args.num_heads}, seed={args.seed}")
    print("=" * 60)
    generate(args.N, args.total_dim, args.num_heads, args.seed,
             args.out, args.exp_lut)
