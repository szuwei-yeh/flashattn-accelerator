"""
generate_test_vectors.py
Generates INT8 Q/K/V matrices and the expected FP output.
Called before running RTL simulation.

Usage:
    python golden/generate_test_vectors.py --N 64 --d 16
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8)


def generate(N: int, d: int, seed: int = 42):
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    # Random float matrices
    Q_fp = rng.standard_normal((N, d)).astype(np.float32)
    K_fp = rng.standard_normal((N, d)).astype(np.float32)
    V_fp = rng.standard_normal((N, d)).astype(np.float32)

    # Per-tensor INT8 quantisation
    q_scale = float(np.abs(Q_fp).max()) / 127.0
    k_scale = float(np.abs(K_fp).max()) / 127.0
    v_scale = float(np.abs(V_fp).max()) / 127.0

    Q_i8 = quantize_int8(Q_fp, q_scale)
    K_i8 = quantize_int8(K_fp, k_scale)
    V_i8 = quantize_int8(V_fp, v_scale)

    # Golden output (float attention on quantised data, to match RTL)
    scale_factor = 1.0 / (d ** 0.5)
    Qt = torch.from_numpy(Q_i8.astype(np.int32))
    Kt = torch.from_numpy(K_i8.astype(np.int32))
    Vt = torch.from_numpy(V_i8.astype(np.int32))

    scores  = (Qt.float() @ Kt.float().T) * q_scale * k_scale * scale_factor
    weights = F.softmax(scores, dim=-1)
    out     = (weights @ Vt.float()) * v_scale     # [N, d]

    # ── Write files ──────────────────────────────────────────────
    # INT8 matrices: one value per line (signed decimal, 2's complement)
    def write_int8(path, mat):
        with open(path, "w") as f:
            for row in mat:
                for val in row:
                    # Write as 8-bit hex (2's complement)
                    f.write(f"{int(val) & 0xFF:02X}\n")

    write_int8(DATA_DIR / "q_input.txt",  Q_i8)
    write_int8(DATA_DIR / "k_input.txt",  K_i8)
    write_int8(DATA_DIR / "v_input.txt",  V_i8)

    # Expected output: one float per line (for Python comparison)
    np.savetxt(DATA_DIR / "expected_output.txt", out.numpy(), fmt="%.6f")

    # Also save scales so dequantiser can use them
    with open(DATA_DIR / "scales.txt", "w") as f:
        f.write(f"q_scale={q_scale:.8f}\n")
        f.write(f"k_scale={k_scale:.8f}\n")
        f.write(f"v_scale={v_scale:.8f}\n")
        f.write(f"N={N}\n")
        f.write(f"d={d}\n")

    print(f"✅  Generated test vectors  N={N}  d={d}")
    print(f"    q_scale={q_scale:.4f}  k_scale={k_scale:.4f}  v_scale={v_scale:.4f}")
    print(f"    Files saved to: {DATA_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",    type=int, default=64)
    parser.add_argument("--d",    type=int, default=16,
                        help="Use 16 for Week-1 systolic test; 64 for full attention")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.N, args.d, args.seed)
