"""
roofline.py — Roofline analysis for FlashAttention hardware accelerator.

Shows:
  1. Hardware roofline (peak compute + DRAM bandwidth bound)
  2. FlashAttention vs naive attention operational intensity (per N)
  3. Measured achieved performance from Verilator simulation
  4. Summary table printed to stdout

References
----------
  Synthesis:  sky130A, Yosys/ABC, OpenSTA  →  f_max ~99 MHz
  Simulation: Verilator, TILE_SIZE=16, HEAD_DIM=16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Hardware parameters (from synthesis + simulation) ───────────────────────

F_MAX_HZ     = 99e6     # post-synthesis f_max estimate (Hz)
N_PES        = 256      # 16×16 systolic array
FLOPS_PER_PE = 2        # 1 MAC = 2 FLOPs
HEAD_DIM     = 16       # attention head dimension (d)

PEAK_GFLOPS  = N_PES * FLOPS_PER_PE * F_MAX_HZ / 1e9   # 50.69 GFLOPS

# External DRAM bandwidth assumption (GB/s).
# sky130A ASIC with simple off-chip memory interface; adjust as needed.
DRAM_BW_GBS  = 2.0

# ── Simulation results  (N, cycles, n_heads) ─────────────────────────────────

SIM_DATA = [
    ( 16,    917, 1),   # flash_attn_top, single head
    ( 64,  11549, 1),
    (128,  44121, 1),
    (256, 172337, 1),
    ( 64,  36134, 4),   # flash_attn_top_axi, 4-head AXI
]

# ── FLOPs and DRAM byte models ───────────────────────────────────────────────

def total_flops(N, d=HEAD_DIM, n_heads=1):
    """QK^T + PV, each N²×d MACs.  1 MAC = 2 FLOPs."""
    return n_heads * 2 * (2 * N * N * d)


def dram_bytes_flashattn(N, d=HEAD_DIM, n_heads=1):
    """
    FlashAttention DRAM traffic.

    The N×N score matrix lives entirely on-chip (tile SRAM + output_buffer)
    and is NEVER written to external DRAM.

    Reads  : Q  (N×d INT8) + K  (N×d INT8) + V  (N×d INT8)
    Writes : Output (N×d INT32 = N×d×4 bytes)
    """
    return n_heads * (3 * N * d + 4 * N * d)   # 7·N·d bytes


def dram_bytes_naive(N, d=HEAD_DIM, n_heads=1):
    """
    Naive attention DRAM traffic.

    The score matrix is materialized to DRAM between QK^T and softmax/PV.

    Reads  : Q, K, V (3·N·d INT8)
    Writes : QK^T scores (N²×4 INT32) → DRAM
    Reads  : QK^T scores back (N²×4)         for softmax
    Writes : softmax(S) (N²×2 Q8.8) → DRAM
    Reads  : softmax(S) back (N²×2)           for PV
    Writes : Output (N×d×4 INT32)
    """
    qkv      = 3 * N * d                    # INT8
    score_rw = 2 * N * N * 4               # INT32 write + read
    softmax_rw = 2 * N * N * 2             # Q8.8 write + read
    output   = N * d * 4                   # INT32
    return n_heads * (qkv + score_rw + softmax_rw + output)


def oi(N, flash=True, d=HEAD_DIM, n_heads=1):
    """Operational intensity (FLOPs / DRAM byte)."""
    flops = total_flops(N, d, n_heads)
    bw    = dram_bytes_flashattn(N, d, n_heads) if flash else dram_bytes_naive(N, d, n_heads)
    return flops / bw


def achieved_gops(N, cycles, n_heads=1):
    flops  = total_flops(N, HEAD_DIM, n_heads)
    time_s = cycles / F_MAX_HZ
    return flops / time_s / 1e9

# ── Print summary table ──────────────────────────────────────────────────────

def print_summary():
    ridge = PEAK_GFLOPS / DRAM_BW_GBS

    print("=" * 72)
    print("  FlashAttention Accelerator — Roofline Summary")
    print("=" * 72)
    print(f"  Peak compute     : {PEAK_GFLOPS:.2f} GFLOPS"
          f"  ({N_PES} PEs × {FLOPS_PER_PE} FLOPs × {F_MAX_HZ/1e6:.0f} MHz)")
    print(f"  DRAM bandwidth   : {DRAM_BW_GBS} GB/s  (assumed, off-chip)")
    print(f"  Ridge point      : {ridge:.1f} FLOPs/byte")
    print()

    hdr = (f"{'N':>5}  {'OI Flash':>10}  {'OI Naive':>10}  "
           f"{'BW saved':>10}  {'Achieved':>10}  {'HW Eff':>8}")
    print(hdr)
    print("-" * len(hdr))

    sim_map = {(N, nh): cyc for N, cyc, nh in SIM_DATA}

    for N in [16, 64, 128, 256]:
        oi_f = oi(N, flash=True)
        oi_n = oi(N, flash=False)
        bw_saved = dram_bytes_naive(N) / dram_bytes_flashattn(N)

        cyc = sim_map.get((N, 1))
        if cyc is not None:
            perf     = achieved_gops(N, cyc)
            hw_eff   = perf / PEAK_GFLOPS * 100
            perf_str = f"{perf:.2f} GOPS"
            eff_str  = f"{hw_eff:.1f}%"
        else:
            perf_str = "N/A"
            eff_str  = "N/A"

        print(f"{N:>5}  {oi_f:>10.1f}  {oi_n:>10.1f}  "
              f"{bw_saved:>9.1f}×  {perf_str:>10}  {eff_str:>8}")

    # 4-head AXI special row
    N, cyc, nh = 64, 36134, 4
    oi_f    = oi(N, flash=True, n_heads=nh)
    bw_saved = dram_bytes_naive(N, n_heads=nh) / dram_bytes_flashattn(N, n_heads=nh)
    perf    = achieved_gops(N, cyc, nh)
    hw_eff  = perf / PEAK_GFLOPS * 100
    print(f"{'64×4':>5}  {oi_f:>10.1f}  {'—':>10}  "
          f"{bw_saved:>9.1f}×  {perf:.2f} GOPS  {hw_eff:.1f}%  ← 4-head AXI")

    print()
    print("  OI Flash / OI Naive = FlashAttention advantage over naive attention")
    print("  HW Eff = Achieved GOPS / Peak GOPS  (gap explained by sequential")
    print("           QK^T→softmax→PV phases and FSM overhead)")
    print("=" * 72)

# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_roofline(save_path="roofline.png"):
    fig, ax = plt.subplots(figsize=(11, 7))

    ridge = PEAK_GFLOPS / DRAM_BW_GBS
    Ns    = [16, 64, 128, 256]
    sim_map = {(N, nh): cyc for N, cyc, nh in SIM_DATA}

    # ── Roofline backbone ─────────────────────────────────────────────────────
    oi_x = np.logspace(-0.2, 2.6, 2000)
    roof = np.minimum(PEAK_GFLOPS, DRAM_BW_GBS * oi_x)
    ax.plot(oi_x, roof, "k-", linewidth=2.5, zorder=4)

    ax.axhline(PEAK_GFLOPS, color="black", linestyle=":", linewidth=1, alpha=0.35)
    ax.text(220, PEAK_GFLOPS * 1.12,
            f"Peak: {PEAK_GFLOPS:.0f} GFLOPS", fontsize=8.5, ha="right", color="#444")

    ax.axvline(ridge, color="#888", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.text(ridge * 1.07, 1.2,
            f"Ridge\n{ridge:.0f} FLOPs/B", fontsize=8, color="#888", va="bottom")

    for pct, label in [(0.05, "5%"), (0.10, "10%")]:
        ax.axhline(PEAK_GFLOPS * pct, color="#ddd", linestyle=":", linewidth=0.8)
        ax.text(210, PEAK_GFLOPS * pct * 1.1, label,
                fontsize=7, color="#bbb", ha="right")

    # ── Naive OI markers (red triangles) ──────────────────────────────────────
    # N=64,128,256 are extremely close (OI 4.7–5.1) → draw individually,
    # but annotate as one cluster to avoid overlapping labels.
    for N in Ns:
        oi_n = oi(N, flash=False)
        pf_n = min(PEAK_GFLOPS, DRAM_BW_GBS * oi_n)
        ax.plot(oi_n, pf_n, "^", color="#e07070", markersize=9,
                markeredgecolor="#b03030", linewidth=1.2, zorder=6, alpha=0.9)

    # N=16: standalone label to the upper-left
    oi_n16 = oi(16, flash=False)
    pf_n16 = min(PEAK_GFLOPS, DRAM_BW_GBS * oi_n16)
    ax.annotate("Naive  N=16", (oi_n16, pf_n16),
                xytext=(-40, 10), textcoords="offset points",
                fontsize=8, color="#b03030", ha="center",
                arrowprops=dict(arrowstyle="-", color="#b03030", lw=0.8))

    # N=64–256 cluster: single label with curved arrow to the cluster center
    oi_c = oi(128, flash=False)
    pf_c = min(PEAK_GFLOPS, DRAM_BW_GBS * oi_c)
    ax.annotate("Naive  N=64–256\n(clustered)", (oi_c, pf_c),
                xytext=(48, 14), textcoords="offset points",
                fontsize=8, color="#b03030",
                arrowprops=dict(arrowstyle="->", color="#b03030", lw=0.9,
                                connectionstyle="arc3,rad=-0.3"))

    # ── FlashAttention OI markers (blue squares) ──────────────────────────────
    for N in Ns:
        oi_f = oi(N, flash=True)
        pf_f = min(PEAK_GFLOPS, DRAM_BW_GBS * oi_f)
        ax.plot(oi_f, pf_f, "s", color="steelblue", markersize=10,
                markeredgecolor="navy", linewidth=1.2, zorder=7)

    # N=16: on the slope → label below-right
    oi_f16 = oi(16, flash=True)
    pf_f16 = min(PEAK_GFLOPS, DRAM_BW_GBS * oi_f16)
    ax.annotate("Flash  N=16", (oi_f16, pf_f16),
                xytext=(8, -16), textcoords="offset points",
                fontsize=8, color="navy",
                arrowprops=dict(arrowstyle="-", color="navy", lw=0.7, alpha=0.5))

    # N=64,128,256: all at the compute ceiling → labels below each marker
    # OI values are 36.6, 73.1, 146.3 — well separated on log scale
    for N, dy in [(64, -20), (128, -20), (256, -20)]:
        oi_f = oi(N, flash=True)
        pf_f = min(PEAK_GFLOPS, DRAM_BW_GBS * oi_f)
        ax.annotate(f"Flash  N={N}", (oi_f, pf_f),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=8, color="navy", ha="center",
                    arrowprops=dict(arrowstyle="-", color="navy", lw=0.6, alpha=0.45))

    # ── Achieved performance diamonds ─────────────────────────────────────────
    # All at OI = flash OI (same design).  N=64 single-head and 4-head share
    # the same OI=36.6, so label them left / right of the marker pair.
    for N in Ns:
        cyc  = sim_map[(N, 1)]
        oi_f = oi(N, flash=True)
        perf = achieved_gops(N, cyc)
        ax.plot(oi_f, perf, "D", color="#44aa44", markersize=9,
                markeredgecolor="#226622", linewidth=1.2, zorder=8)

    # Inline N= tags (short — GOPS detail is in the table below)
    label_offsets_1h = {16: (-12, 8), 64: (-14, 8), 128: (-16, 8), 256: (-18, 8)}
    for N, (dx, dy) in label_offsets_1h.items():
        cyc  = sim_map[(N, 1)]
        oi_f = oi(N, flash=True)
        perf = achieved_gops(N, cyc)
        ax.annotate(f"N={N}", (oi_f, perf),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color="#226622", ha="center")

    # 4-head AXI diamond (gold) — label to the right to avoid N=64 single-head label
    cyc_4h  = sim_map[(64, 4)]
    oi_4h   = oi(64, flash=True, n_heads=4)   # same OI as single-head N=64
    perf_4h = achieved_gops(64, cyc_4h, 4)
    ax.plot(oi_4h, perf_4h, "D", color="gold", markersize=11,
            markeredgecolor="goldenrod", linewidth=1.2, zorder=8)
    ax.annotate("4-head", (oi_4h, perf_4h),
                xytext=(10, 6), textcoords="offset points",
                fontsize=7.5, color="goldenrod", ha="left")

    # ── Achieved performance table (lower-left text box) ──────────────────────
    rows = []
    for N in Ns:
        cyc  = sim_map[(N, 1)]
        perf = achieved_gops(N, cyc)
        eff  = perf / PEAK_GFLOPS * 100
        rows.append(f"  N={N:<4}  {perf:.2f} GOPS  ({eff:.1f}%)")
    perf_4 = achieved_gops(64, cyc_4h, 4)
    eff_4  = perf_4 / PEAK_GFLOPS * 100
    rows.append(f"  4-head  {perf_4:.2f} GOPS  ({eff_4:.1f}%)")

    table_txt = "Achieved Performance  (Verilator, 99 MHz)\n" \
                + "─" * 42 + "\n" \
                + "\n".join(rows)
    ax.text(0.015, 0.025, table_txt,
            transform=ax.transAxes, va="bottom",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#aaa", alpha=0.88))

    # ── Legend (upper-left — naturally empty on a roofline) ───────────────────
    legend_elements = [
        mlines.Line2D([0], [0], color="black", lw=2.5,
                      label="Roofline"),
        mlines.Line2D([0], [0], marker="s", color="steelblue", markersize=9,
                      markeredgecolor="navy", linestyle="None",
                      label="FlashAttention  (theoretical OI)"),
        mlines.Line2D([0], [0], marker="^", color="#e07070", markersize=9,
                      markeredgecolor="#b03030", linestyle="None",
                      label="Naive attention  (theoretical OI)"),
        mlines.Line2D([0], [0], marker="D", color="#44aa44", markersize=9,
                      markeredgecolor="#226622", linestyle="None",
                      label="Achieved — single head  (Verilator)"),
        mlines.Line2D([0], [0], marker="D", color="gold", markersize=10,
                      markeredgecolor="goldenrod", linestyle="None",
                      label="Achieved — 4-head AXI  (Verilator)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
              framealpha=0.92, edgecolor="#ccc")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity  (FLOPs / DRAM byte)", fontsize=12)
    ax.set_ylabel("Performance  (GFLOPS)", fontsize=12)
    ax.set_title(
        "Roofline Analysis — FlashAttention Hardware Accelerator\n"
        f"sky130A ASIC · 99 MHz · 16×16 INT8 Systolic Array  "
        f"(DRAM BW assumed {DRAM_BW_GBS} GB/s)",
        fontsize=12)
    ax.set_xlim(1.5, 250)
    ax.set_ylim(1.0, PEAK_GFLOPS * 3)
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {save_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_summary()
    plot_roofline()
