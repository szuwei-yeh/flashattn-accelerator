"""
generate_test_vectors.py — Generate Q/K/V INT8 test vectors + expected output

Usage:
    python golden/generate_test_vectors.py --N 16 --d 16 --seed 42

Outputs (all in data/):
    q_input.hex       — Q matrix, INT8, row-major, one byte per line
    k_input.hex       — K matrix, INT8, row-major, one byte per line
    v_input.hex       — V matrix, INT8, row-major, one byte per line
    expected.hex      — Expected output O, INT32, row-major, one word per line (8 hex digits)
    scales.txt        — scale_q and scale_k in Q8.8 format (for DUT configuration)
"""

import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from int8_attention import quantize_symmetric, dequantize, int8_matmul, online_softmax


def quantize_symmetric_q88(x: np.ndarray, bits: int = 8):
    """Same as int8_attention.py but also returns scale in Q8.8 integer format."""
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0, 256  # 1.0 in Q8.8
    q_max = (1 << (bits - 1)) - 1
    scale = max_val / q_max
    x_int = np.clip(np.round(x / scale), -q_max - 1, q_max).astype(np.int8)
    scale_q88 = int(round(scale * 256))   # convert to Q8.8
    scale_q88 = max(1, min(scale_q88, 0x7FFF))  # clamp to signed 16-bit positive
    return x_int, scale, scale_q88


def generate(N: int, d: int, seed: int, out_dir: str):
    rng = np.random.default_rng(seed)

    # ── Generate random FP32 matrices ────────────────────────
    Q_fp = rng.standard_normal((N, d)).astype(np.float32)
    K_fp = rng.standard_normal((N, d)).astype(np.float32)
    V_fp = rng.standard_normal((N, d)).astype(np.float32)

    # ── Quantize to INT8 ─────────────────────────────────────
    Q_i8, sq, sq_q88 = quantize_symmetric_q88(Q_fp)
    K_i8, sk, sk_q88 = quantize_symmetric_q88(K_fp)
    V_i8, sv, sv_q88 = quantize_symmetric_q88(V_fp)

    print(f"Q  scale: {sq:.6f}  (Q8.8 = 0x{sq_q88:04X})")
    print(f"K  scale: {sk:.6f}  (Q8.8 = 0x{sk_q88:04X})")
    print(f"V  scale: {sv:.6f}  (Q8.8 = 0x{sv_q88:04X})")

    # ── Compute expected output (INT8 attention) ──────────────
    scale_factor = 1.0 / np.sqrt(d)

    # QK^T in INT8
    QKt_i32 = int8_matmul(Q_i8, K_i8.T)           # [N, N] INT32
    QKt_fp  = dequantize(QKt_i32, sq * sk)         # recover float
    scores  = QKt_fp * scale_factor                 # scale by 1/sqrt(d)

    # Softmax
    W = online_softmax(scores)                      # [N, N] float weights

    # PV: use float weights × INT8 V
    PV = W @ V_i8.astype(np.float32) * sv          # [N, d] float

    # Convert to INT32 for storage (scale up by 256 to keep precision)
    PV_i32 = np.round(PV * 256).astype(np.int32)

    # ── Write hex files ───────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    def write_int8_hex(path, mat):
        """Write INT8 matrix as hex, one byte per line (2 hex digits, unsigned)."""
        with open(path, "w") as f:
            for val in mat.flatten():
                f.write(f"{int(val) & 0xFF:02X}\n")
        print(f"Written: {path}  ({mat.size} entries)")

    def write_int32_hex(path, mat):
        """Write INT32 matrix as hex, one word per line (8 hex digits)."""
        with open(path, "w") as f:
            for val in mat.flatten():
                f.write(f"{int(val) & 0xFFFFFFFF:08X}\n")
        print(f"Written: {path}  ({mat.size} entries)")

    write_int8_hex(os.path.join(out_dir, "q_input.hex"), Q_i8)
    write_int8_hex(os.path.join(out_dir, "k_input.hex"), K_i8)
    write_int8_hex(os.path.join(out_dir, "v_input.hex"), V_i8)
    write_int32_hex(os.path.join(out_dir, "expected.hex"), PV_i32)

    # ── Write scales ──────────────────────────────────────────
    scales_path = os.path.join(out_dir, "scales.txt")
    with open(scales_path, "w") as f:
        f.write(f"scale_q_q88 = 0x{sq_q88:04X}\n")
        f.write(f"scale_k_q88 = 0x{sk_q88:04X}\n")
        f.write(f"scale_v_q88 = 0x{sv_q88:04X}\n")
        f.write(f"N           = {N}\n")
        f.write(f"d           = {d}\n")
        f.write(f"seed        = {seed}\n")
    print(f"Written: {scales_path}")

    # ── Sanity check ──────────────────────────────────────────
    from int8_attention import fp32_attention
    out_fp32    = fp32_attention(Q_fp, K_fp, V_fp)
    out_int8    = PV
    abs_err     = np.abs(out_fp32 - out_int8)
    row_max     = np.abs(out_fp32).max(axis=-1, keepdims=True)
    rel_err     = abs_err / (row_max + 1e-8)
    print(f"\nSanity check — INT8 vs FP32:")
    print(f"  Mean relative error: {rel_err.mean()*100:.3f}%")
    print(f"  Max  relative error: {rel_err.max()*100:.3f}%")
    if rel_err.mean() < 0.01:
        print("  PASS ✅")
    else:
        print("  WARN ⚠️  > 1% — check quantization")

    print(f"\nDone. Files written to: {out_dir}/")
    return sq_q88, sk_q88


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",    type=int, default=16,   help="Sequence length")
    parser.add_argument("--d",    type=int, default=16,   help="Head dimension")
    parser.add_argument("--seed", type=int, default=42,   help="Random seed")
    parser.add_argument("--out",  type=str, default="data", help="Output directory")
    args = parser.parse_args()

    print(f"Generating test vectors: N={args.N}, d={args.d}, seed={args.seed}")
    print("=" * 55)
    generate(args.N, args.d, args.seed, args.out)