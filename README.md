# FlashAttention Hardware Accelerator

A cycle-accurate RTL implementation of the FlashAttention algorithm in
SystemVerilog, verified with Verilator across 18 test configurations, 0 mismatches.

---

## Problem Statement

Standard scaled dot-product attention materializes an N×N score matrix,
requiring O(N²) memory bandwidth. For N=1024 this is ~4 MB of intermediate
data written and read back per layer — bandwidth, not compute, is the bottleneck.

FlashAttention tiles the computation into TILE_SIZE×TILE_SIZE blocks and uses
online softmax (running max + running sum carried across KV tiles) so the
intermediate score matrix never leaves on-chip SRAM. This design implements
that algorithm in hardware using INT8 quantization and a systolic array.

---

## Architecture

```
AXI4-Stream IN  (Q: N×d, K/V: N×d' — d'=d for MHA, d'=d/GQA_RATIO for GQA)
         ↓
┌──────────────────────────────────────────┐
│           flash_attn_top_axi             │
│                                          │
│  ┌───────────────────────────────────┐   │
│  │        axi4_stream_slave          │   │
│  │  routes Q/K/V bytes → per-head   │   │
│  │  SRAMs; GQA: K/V broadcast to    │   │
│  │  grouped Q-head pairs             │   │
│  └──────────────┬────────────────────┘   │
│                 ↓  (4 cores, parallel)   │
│  ┌────────────────────────────────────┐  │
│  │       flash_attn_core × 4          │  │
│  │                                    │  │
│  │  Q/K/V SRAM  →  tile_controller   │  │
│  │                      ↓             │  │
│  │         ┌────────────────────┐     │  │
│  │         │  16×16 systolic    │     │  │
│  │         │  array (INT8 MAC)  │     │  │
│  │         │  QK^T and PV share │     │  │
│  │         └────────┬───────────┘     │  │
│  │                  ↓                 │  │
│  │         online_softmax × 16        │  │
│  │         (exp LUT, running max/sum) │  │
│  │                  ↓                 │  │
│  │         output_buffer              │  │
│  │         (INT32 accum + rescale     │  │
│  │          + normalize across tiles) │  │
│  │                                    │  │
│  │         kv_cache (prefill/decode)  │  │
│  └────────────────────────────────────┘  │
│                 ↓                        │
│  ┌───────────────────────────────────┐   │
│  │       axi4_stream_master          │   │
│  │  streams INT32 output row-major   │   │
│  └───────────────────────────────────┘   │
└──────────────────────────────────────────┘
         ↓
AXI4-Stream OUT (attention output, INT32)
```

---

## Key Design Decisions

**16×16 INT8 Systolic Array — reused for QK^T and PV**
The same array handles both matrix multiplications via an input mux
(`is_pv_phase`), halving area vs two separate arrays. INT8×INT8→INT32
accumulation prevents overflow across 16 MAC operations. For HEAD_DIM=64,
the inner dimension is tiled into 4 chunks (NUM_CHUNKS=4); the PE
accumulators are not cleared between chunks (`array_no_clear`), so partial
sums accumulate correctly across passes.

**Online Softmax with exp LUT**
A 256-entry ROM covers exp(x) for x ∈ [−8, 0] in Q8.8 format — sufficient
because scores are always shifted to (score − running_max) ≤ 0 before
lookup, and exp(x < −8) ≈ 0. 16 softmax instances run in parallel (one
per Q-row). `running_max` and `running_sum` carry across KV tiles; the
output buffer applies a rescale correction (× exp(m_old − m_new)) whenever
a new tile lowers the running maximum, and a final normalize (÷ running_sum)
on the last tile.

**K Stored Row-Major, Transposed at Read Time**
QK^T requires K^T. K is stored row-major in a flat SRAM. The systolic array
data-slicing mux reads K with swapped row/col indices:
`K_reg[col * HEAD_DIM + chunk * TILE_SIZE + row]` — so K^T is obtained
without a transpose unit or extra cycles.

**KV Prefetch Double-Buffer**
While the systolic array processes the current KV tile, the next tile's K/V
data is prefetched from SRAM into shadow registers (`K_reg_nxt/V_reg_nxt`).
On tile completion, a single swap copies shadow → active in one cycle.
This overlaps SRAM reads with compute, reducing idle cycles by ~20%.

**Dynamic SRAM Sizing**
`SRAM_DEPTH = SEQ_LEN × HEAD_DIM` is computed from parameters, so the
same RTL supports N=16 d=16 (256 entries) through N=256 d=64 (16384 entries)
without any structural changes.

**GQA (Grouped Query Attention)**
`GQA_RATIO` controls how many Q-heads share each KV-head. For `GQA_RATIO=2`
(LLaMA 2/3 / Mistral style): 4 Q-heads, 2 KV-heads — Q-heads 0,1 share
KV-head 0; Q-heads 2,3 share KV-head 1. The K/V AXI stream is
`NUM_KV_HEADS`-wide (half the bandwidth of MHA), and the AXI slave uses
separate address decompositions for Q vs K/V phases.

**KV Cache for Decode Mode**
Two SRAMs store K/V vectors indexed by token position (up to 256 tokens).
In decode mode (`mode=1`) a single query attends to all cached K/V; `kv_len`
sets the inner loop bound dynamically, enabling autoregressive generation
without recomputing past keys and values.

---

## Performance

Simulation cycles (Verilator, TILE_SIZE=16, with KV prefetch pipeline):

### Non-causal (prefill)

| N \ d | d=16    | d=64    |
|-------|---------|---------|
| 16    | 1,429   | —       |
| 64    | 13,585  | 52,429  |
| 128   | 48,161  | 185,081 |
| 256   | 180,289 | 691,057 |

### Causal (decoder self-attention)

| N \ d | d=16    | d=64    |
|-------|---------|---------|
| 64    | 9,655   | 37,393  |
| 256   | 101,689 | 390,337 |

Causal mode skips above-diagonal tiles (~50% fewer KV tiles at large N).

4-head AXI top (N=64, d=16): done at cycle 21,785 (MHA), cycle 21,785 (GQA).

---

## Features

- **Tiled online FlashAttention** — running max + running sum carried across
  KV tiles; never materializes the full N×N score matrix
- **16×16 INT8 systolic array** — skewed input feeding, INT8×INT8→INT32 MAC,
  reused for QK^T and PV via input mux
- **HEAD_DIM=64** — inner-dimension tiling (4 chunks of 16); same RTL,
  backward-compatible with HEAD_DIM=16
- **Causal masking** — decoder self-attention; above-diagonal tiles skipped
  entirely (no wasted cycles)
- **KV prefetch pipeline** — double-buffer hides SRAM load latency behind
  compute; measured cycle reduction vs serial load grows with sequence length:
  18.5% (N=64), 23.0% (N=128), 25.5% (N=256)
- **Dynamic SRAM depth** — `SEQ_LEN × HEAD_DIM`; tested N=16–256, d=16/64
- **4-head parallel attention** — four `flash_attn_core` instances share one
  AXI4-Stream interface
- **GQA (Grouped Query Attention)** — `GQA_RATIO` parameter; 4Q+2KV tested
  (LLaMA 2/3 / Mistral style); K/V stream bandwidth halved vs MHA
- **KV cache + decode mode** — append-only token storage; single-token
  decode attending to full prefill context; tested d=16 and d=64

---

## RTL Modules (19 files)

| Module | Path | Description |
|--------|------|-------------|
| `pe` | `rtl/systolic/pe.sv` | Single INT8×INT8→INT32 MAC PE |
| `systolic_array` | `rtl/systolic/systolic_array.sv` | 16×16 PE array, flat acc port |
| `array_controller` | `rtl/systolic/array_controller.sv` | Skewed-input FSM, no_clear mode |
| `quantizer` | `rtl/quantization/quantizer.sv` | Fixed-point → INT8 |
| `dequantizer` | `rtl/quantization/dequantizer.sv` | INT32×scale → Q8.8 (÷256) |
| `exp_lut` | `rtl/softmax/exp_lut.sv` | 256-entry ROM, Q8.8, 1-cycle |
| `online_softmax` | `rtl/softmax/online_softmax.sv` | Running max+sum, cross-tile rescale |
| `sram_1r1w` | `rtl/memory/sram_1r1w.sv` | 1R1W SRAM behavioral model |
| `kv_cache` | `rtl/memory/kv_cache.sv` | Append-only KV token store (max 256) |
| `q_tile_buffer` | `rtl/memory/q_tile_buffer.sv` | Q tile SRAM wrapper |
| `kv_tile_buffer` | `rtl/memory/kv_tile_buffer.sv` | K/V ping-pong buffer |
| `output_buffer` | `rtl/memory/output_buffer.sv` | INT32 accum + rescale + normalize |
| `addr_gen` | `rtl/ctrl/addr_gen.sv` | SRAM address generator, global offset |
| `tile_controller` | `rtl/ctrl/tile_controller.sv` | Two-level tile FSM (11 states) |
| `axi4_stream_slave` | `rtl/interface/axi4_stream_slave.sv` | Q/K/V byte stream → per-head SRAM, GQA-aware |
| `axi4_stream_master` | `rtl/interface/axi4_stream_master.sv` | INT32 output → AXI4-Stream |
| `flash_attn_top` | `rtl/top/flash_attn_top.sv` | Single-head top (no AXI) |
| `flash_attn_core` | `rtl/top/flash_attn_core.sv` | Single-head core + KV cache |
| `flash_attn_top_axi` | `rtl/top/flash_attn_top_axi.sv` | 4-head top with AXI4-Stream + GQA |

---

## Verification

| Test | Config | Mismatches |
|------|--------|------------|
| systolic_array | identity, all-ones, ramp, negative, back-to-back, random INT8 | 0 |
| quantizer | random Q8.8, scale sweep (100 cases) | 0 |
| exp_lut | full 256-entry sweep | 0 |
| online_softmax | single tile, dual tile, running max reset | 0 |
| sram_1r1w | write/read, sequential | 0 |
| addr_gen | overflow, counter stop | 0 |
| kv_tile_buffer | ping-pong full cycle | 0 |
| flash_attn_top | N=16/64/128/256 d=16 | 0 |
| flash_attn_top | N=64/128/256 d=64 | 0 |
| flash_attn_top | causal N=64/256 d=16 | 0 |
| flash_attn_top | causal N=64/256 d=64 | 0 |
| axi4_stream_slave | byte routing, mat_sel transition, second row | 0 |
| flash_attn_top_axi | MHA 4-head N=64 d=16 | 0 |
| flash_attn_top_axi | MHA 4-head N=64 d=64 | 0 |
| flash_attn_top_axi | GQA 4Q+2KV N=64 d=16 | 0 |
| kv_cache | write/read, fill, overflow guard, reset | 0 |
| flash_attn_core | prefill+decode N=32 d=16 | 0 |
| flash_attn_core | prefill+decode N=32 d=64 | 0 |
| **Total** | | **0 / 0** |

All tests run automatically via `make regression`.

---

## How to Run

### Prerequisites

```bash
brew install verilator   # macOS
pip install numpy
```

### Full regression

```bash
cd sim/verilator
make regression
```

### Individual targets

```bash
# Single-head (no AXI)
make tb_top_N16                # N=16, d=16
make tb_top_N64                # N=64, d=16
make tb_top_N256               # N=256, d=16
make tb_top_N64_d64            # N=64, d=64
make tb_top_N256_d64           # N=256, d=64
make tb_top_causal_N64         # N=64, d=16, causal
make tb_top_causal_N256_d64    # N=256, d=64, causal

# 4-head AXI
make axi_top_N64               # MHA, N=64, d=16
make axi_top_N64_d64           # MHA, N=64, d=64
make axi_top_N64_gqa           # GQA 4Q+2KV, N=64, d=16

# KV cache / decode
make tb_kv_cache               # KV cache unit test
make tb_kv_decode              # prefill + decode, d=16
make tb_kv_decode_d64          # prefill + decode, d=64
```

### Regenerate test vectors

```bash
python golden/generate_test_vectors.py
python golden/generate_hw_expected.py
python golden/generate_multihead_data.py
python golden/generate_multihead_data.py --num_kv_heads 2 --out data/N64_gqa
python golden/generate_kv_cache_test.py
```

---

## File Structure

```
flashattn-accelerator/
├── rtl/
│   ├── systolic/       pe, systolic_array, array_controller
│   ├── quantization/   quantizer, dequantizer
│   ├── softmax/        exp_lut, online_softmax
│   ├── memory/         sram_1r1w, kv_cache, tile buffers, output_buffer
│   ├── ctrl/           addr_gen, tile_controller
│   ├── interface/      axi4_stream_slave (GQA-aware), axi4_stream_master
│   └── top/            flash_attn_top, flash_attn_core, flash_attn_top_axi
├── sim/verilator/
│   ├── Makefile        (regression + all individual targets)
│   └── tb_*.cpp
├── golden/
│   └── *.py            (HW-accurate Python reference models)
├── data/               (pre-generated test vectors, all configs)
└── roofline.png        (arithmetic intensity analysis)
```
