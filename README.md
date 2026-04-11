# FlashAttention Hardware Accelerator

A cycle-accurate RTL implementation of the FlashAttention algorithm in
SystemVerilog, verified with Verilator (391 test cases, 0 mismatches).

---

## Problem Statement

Standard scaled dot-product attention materializes an N×N score matrix,
requiring O(N²) memory bandwidth to HBM. For N=1024 this is ~4 MB of
intermediate data written and read back per attention layer — bandwidth,
not compute, is the bottleneck.

FlashAttention tiles the computation into TILE_SIZE×TILE_SIZE blocks and
uses online softmax (running max + running sum) so intermediate scores
never leave on-chip SRAM. This design implements that algorithm in
hardware using INT8 quantization and a systolic array.

---

## Architecture

```
AXI4-Stream IN (Q, K, V)
         ↓
┌──────────────────────────────────────┐
│         flash_attn_top_axi           │
│                                      │
│  ┌─────────────────────────────────┐ │
│  │     axi4_stream_slave           │ │
│  │  (byte routing → 4 heads)       │ │
│  └────────────┬────────────────────┘ │
│               ↓                      │
│  ┌────────────────────────────────┐  │
│  │  flash_attn_core × 4 (heads)   │  │
│  │                                │  │
│  │  Q/K/V SRAM → tile_controller  │  │
│  │       ↓              ↓         │  │
│  │  systolic_array   kv_cache     │  │
│  │  (16×16 INT8 MAC)  (prefill/   │  │
│  │       ↓             decode)    │  │
│  │  online_softmax                │  │
│  │  (exp LUT, running max/sum)    │  │
│  │       ↓                        │  │
│  │  output_buffer (INT32 accum)   │  │
│  └────────────────────────────────┘  │
│               ↓                      │
│  ┌─────────────────────────────────┐ │
│  │     axi4_stream_master          │ │
│  │  (row-major INT32 streaming)    │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
         ↓
AXI4-Stream OUT (attention output)
```

---

## Key Design Decisions

**16×16 INT8 Systolic Array**
The array size matches HEAD_DIM=16, so one tile fits exactly one
attention head dimension. INT8×INT8→INT32 accumulation prevents
overflow across 16 multiply-accumulate operations. The same array
is reused for both QK^T and PV phases via an input mux, reducing
area at the cost of serializing the two matrix multiplications.

**Online Softmax with exp LUT**
Hardware exp is expensive. A 256-entry ROM covers exp(x) for
x ∈ [-8, 0] in Q8.8 format — sufficient because scores are always
normalized to (score − running_max) ≤ 0, and exp(x < -8) ≈ 0.
16 softmax instances run in parallel, one per Q row.

**K Transpose at Load Time**
QK^T requires K transposed. Rather than a separate transpose unit,
K is stored into tile registers with swapped address nibbles during
LOAD_KV, so K_reg is naturally laid out as K^T. No extra logic or
cycles required.

**Flat SRAM + Global Offset Addressing**
All Q/K/V data is preloaded into flat SRAMs before start. The address
generator computes `global_offset = tile_row × HEAD_DIM` and adds a
local counter, supporting multi-tile sequences (N=16/64/128/256)
without architectural changes.

**KV Cache for Decode Mode**
Two SRAMs store K/V vectors indexed by token position (max 256 tokens).
In decode mode (mode=1) the tile_controller outer loop runs once, and
kv_len sets the inner loop bound dynamically, enabling autoregressive
generation without recomputing past keys and values.

---

## Performance

Simulation cycles (Verilator, TILE_SIZE=16, HEAD_DIM=16):

| Sequence Length | Cycles  | Tile iterations |
|----------------|---------|-----------------|
| N=16           | 917     | 1×1 = 1         |
| N=64           | 11,549  | 4×4 = 16        |
| N=128          | 44,121  | 8×8 = 64        |
| N=256          | 172,337 | 16×16 = 256     |

4-head AXI top (N=64): done at cycle 36,134, all heads max_err=0.

---

## Features

- **Tiled online softmax** — never materializes the full N×N score matrix;
  uses running max + running sum with a 256-entry exp LUT (Q8.8)
- **16×16 INT8 systolic array** — skewed input feeding, INT8×INT8→INT32
  MAC, reused for both QK^T and PV phases
- **4 parallel attention heads** — each head is an independent
  `flash_attn_core` instance fed through a single AXI4-Stream interface
- **KV Cache** — append-only token storage supporting prefill and decode modes
- **AXI4-Stream interface** — slave routes Q/K/V byte streams per head;
  master streams INT32 output row-major
- **Scalable sequence length** — tested N=16/64/128/256

---

## RTL Modules (19 files)

| Module | Path | Description |
|--------|------|-------------|
| `pe` | `rtl/systolic/pe.sv` | Single INT8×INT8→INT32 MAC PE |
| `systolic_array` | `rtl/systolic/systolic_array.sv` | 16×16 PE array, 1D flat acc port |
| `array_controller` | `rtl/systolic/array_controller.sv` | Skewed input FSM |
| `quantizer` | `rtl/quantization/quantizer.sv` | Fixed-point → INT8 |
| `dequantizer` | `rtl/quantization/dequantizer.sv` | INT32 → Q8.8 score (÷256, ×1/√d) |
| `exp_lut` | `rtl/softmax/exp_lut.sv` | 256-entry ROM, Q8.8, 1-cycle latency |
| `online_softmax` | `rtl/softmax/online_softmax.sv` | Running max + sum + normalize FSM |
| `sram_1r1w` | `rtl/memory/sram_1r1w.sv` | 1R1W SRAM behavioral model |
| `kv_cache` | `rtl/memory/kv_cache.sv` | KV cache (2×SRAM, token append) |
| `q_tile_buffer` | `rtl/memory/q_tile_buffer.sv` | Q tile SRAM wrapper |
| `kv_tile_buffer` | `rtl/memory/kv_tile_buffer.sv` | K/V ping-pong buffer |
| `output_buffer` | `rtl/memory/output_buffer.sv` | INT32 accumulation buffer (DEPTH=4096) |
| `addr_gen` | `rtl/ctrl/addr_gen.sv` | SRAM address generator w/ global offset |
| `tile_controller` | `rtl/ctrl/tile_controller.sv` | Two-level tile loop FSM (11 states) |
| `axi4_stream_slave` | `rtl/interface/axi4_stream_slave.sv` | Q/K/V byte-stream → per-head SRAM |
| `axi4_stream_master` | `rtl/interface/axi4_stream_master.sv` | INT32 output → AXI4-Stream |
| `flash_attn_top` | `rtl/top/flash_attn_top.sv` | Single-head top (no AXI) |
| `flash_attn_core` | `rtl/top/flash_attn_core.sv` | Single-head core + KV cache |
| `flash_attn_top_axi` | `rtl/top/flash_attn_top_axi.sv` | 4-head top with AXI4-Stream |

---

## Verification

| Module | Scenarios | Cases | Mismatches |
|--------|-----------|-------|------------|
| systolic_array | identity, all-ones, ramp, negative, back-to-back, random INT8 | 10 | 0 |
| quantizer | random Q8.8, scale sweep | 100 | 0 |
| exp_lut | full 256-entry sweep | 256 | 0 |
| online_softmax | single tile, dual tile, running max reset | 3 | 0 |
| sram_1r1w | write/read, sequential | 2 | 0 |
| addr_gen | overflow fix, counter stop | 2 | 0 |
| kv_tile_buffer | ping-pong full cycle | 3 | 0 |
| flash_attn_top | end-to-end N=16/64/128/256 (max abs err=0) | 4 | 0 |
| axi4_stream_slave | byte routing, mat_sel transition, second row | 3 | 0 |
| flash_attn_top_axi | 4-head N=64, max_err=0 | 1 | 0 |
| kv_cache | write/read, fill, overflow guard, reset | 5 | 0 |
| flash_attn_core | prefill N=32, decode mode, KV readback | 3 | 0 |
| **Total** | | **391** | **0** |

---

## How to Run

### Prerequisites

```bash
brew install verilator   # macOS
pip install numpy
```

### Run all tests

```bash
cd sim/verilator
make regression
```

### Individual targets

```bash
make tb_top_N16        # end-to-end N=16
make tb_top_N64        # end-to-end N=64
make tb_top_N128       # end-to-end N=128
make tb_top_N256       # end-to-end N=256
make axi_top_N64       # 4-head AXI top N=64
make kv_cache          # KV cache unit test
make kv_decode         # prefill + decode end-to-end
```

### Regenerate test vectors

```bash
python golden/generate_test_vectors.py
python golden/generate_hw_expected.py
python golden/generate_multihead_data.py
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
│   ├── interface/      axi4_stream_slave, axi4_stream_master
│   └── top/            flash_attn_top, flash_attn_core, flash_attn_top_axi
├── sim/verilator/
│   ├── Makefile
│   └── tb_*.cpp
├── golden/
│   └── *.py
└── data/
```
