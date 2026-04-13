"""
FlashAttention Golden Model — Week 1
Standard attention vs. tiled online-softmax attention.
Both should produce identical results (within atol=1e-4).
"""

import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1. Standard O(N²) attention  (reference baseline)
# ─────────────────────────────────────────────
def standard_attention(Q, K, V, scale):
    """
    Naive scaled dot-product attention.
    Materialises the full N×N score matrix — O(N²) memory.
    """
    scores  = (Q @ K.T) * scale          # [N, N]
    weights = F.softmax(scores, dim=-1)  # [N, N]
    return weights @ V                   # [N, d]


# ─────────────────────────────────────────────
# 2. FlashAttention tiled  (matches what RTL will compute)
# ─────────────────────────────────────────────
def flash_attention_tiled(Q, K, V, scale, tile_size=64):
    """
    Tiled online-softmax attention.
    Never materialises the full N×N score matrix → O(N) memory.

    Per-Q-tile state:
        mj  — running max   (shape: [tile])
        lj  — running sum   (shape: [tile])
        Oj  — output accum  (shape: [tile, d])

    On each K/V tile:
        m_new = max(mj, row_max(Sij))
        P     = exp(Sij − m_new)          # re-centred scores
        l_new = exp(mj − m_new)*lj + rowsum(P)
        O_new = exp(mj − m_new)*Oj + P @ Vi
    """
    N, d = Q.shape
    O = torch.zeros(N, d, dtype=torch.float32)

    for j in range(0, N, tile_size):          # outer: Q tiles
        Qj = Q[j : j + tile_size]             # [tile, d]
        t  = Qj.shape[0]

        Oj = torch.zeros(t, d)
        mj = torch.full((t,), float('-inf'))  # running max
        lj = torch.zeros(t)                   # running sum

        for i in range(0, N, tile_size):      # inner: K/V tiles
            Ki = K[i : i + tile_size]         # [tile, d]
            Vi = V[i : i + tile_size]         # [tile, d]

            Sij   = (Qj @ Ki.T) * scale                        # [t, tile]
            m_new = torch.maximum(mj, Sij.max(dim=-1).values)  # [t]

            P     = torch.exp(Sij - m_new.unsqueeze(-1))       # [t, tile]
            alpha = torch.exp(mj - m_new)                      # [t]  rescale factor

            l_new = alpha * lj + P.sum(dim=-1)                 # [t]
            Oj    = alpha.unsqueeze(-1) * Oj + P @ Vi          # [t, d]

            mj, lj = m_new, l_new

        O[j : j + tile_size] = Oj / lj.unsqueeze(-1)          # normalise

    return O


# ─────────────────────────────────────────────
# 3. INT8 quantised version  (matches RTL exactly — Week 2+)
# ─────────────────────────────────────────────
def quantize_to_int8(x, scale):
    """Per-tensor symmetric INT8 quantisation."""
    return torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)

def int8_attention(Q_fp, K_fp, V_fp, q_scale, k_scale, v_scale, acc_scale):
    """
    INT8 quantised attention — mirrors RTL data path:
      1. Quantise Q, K, V to INT8
      2. Compute Q×Kᵀ with INT32 accumulation
      3. Apply softmax on INT32 scores (float conversion)
      4. Compute P×V with INT32 accumulation
    Approximation errors here should be < 1% vs float.
    """
    N, d = Q_fp.shape
    scale = 1.0 / (d ** 0.5)

    Q_i8 = quantize_to_int8(Q_fp, q_scale)
    K_i8 = quantize_to_int8(K_fp, k_scale)
    V_i8 = quantize_to_int8(V_fp, v_scale)

    # INT32 matrix multiply (simulate accumulation)
    Q_i32 = Q_i8.to(torch.int32)
    K_i32 = K_i8.to(torch.int32)
    V_i32 = V_i8.to(torch.int32)

    scores_i32 = Q_i32 @ K_i32.T                      # [N, N] INT32
    scores_fp  = scores_i32.float() * (q_scale * k_scale) * scale

    weights = F.softmax(scores_fp, dim=-1)             # float softmax

    out_fp = weights @ V_i32.float() * v_scale         # dequantise V
    return out_fp


# ─────────────────────────────────────────────
# 4. Verification
# ─────────────────────────────────────────────
def verify_golden_models():
    torch.manual_seed(42)

    N, d    = 64, 64
    scale   = 1.0 / (d ** 0.5)
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    out_std   = standard_attention(Q, K, V, scale)
    out_flash = flash_attention_tiled(Q, K, V, scale, tile_size=64)

    max_err = (out_std - out_flash).abs().max().item()
    assert torch.allclose(out_std, out_flash, atol=1e-4), \
        f"MISMATCH! max error = {max_err:.6f}"
    print(f"✅  standard vs flash_tiled   — max error: {max_err:.2e}  (PASS)")

    # Also test when N is NOT a multiple of tile_size (partial last tile)
    N2 = 70
    Q2 = torch.randn(N2, d)
    K2 = torch.randn(N2, d)
    V2 = torch.randn(N2, d)
    out_s2 = standard_attention(Q2, K2, V2, scale)
    out_f2 = flash_attention_tiled(Q2, K2, V2, scale, tile_size=64)
    max_err2 = (out_s2 - out_f2).abs().max().item()
    assert torch.allclose(out_s2, out_f2, atol=1e-4), \
        f"MISMATCH (partial tile)! max error = {max_err2:.6f}"
    print(f"✅  partial-tile test (N=70)  — max error: {max_err2:.2e}  (PASS)")

    # INT8 test (looser tolerance — quantisation noise expected)
    q_scale = Q.abs().max().item() / 127.0
    k_scale = K.abs().max().item() / 127.0
    v_scale = V.abs().max().item() / 127.0
    out_i8  = int8_attention(Q, K, V, q_scale, k_scale, v_scale, acc_scale=1.0)
    rel_err = ((out_std - out_i8).abs() / (out_std.abs() + 1e-6)).mean().item()
    print(f"✅  INT8 quantised            — mean rel error: {rel_err:.2%}  "
          f"({'PASS' if rel_err < 0.01 else 'WARN — > 1%'})")


if __name__ == "__main__":
    verify_golden_models()
