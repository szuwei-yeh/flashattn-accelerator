# FlashAttention Hardware Accelerator

A cycle-accurate FlashAttention accelerator implemented in SystemVerilog, verified with Verilator (391 test cases, 0 mismatches).

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

## Features

- **Tiled online softmax** — never materializes the full N×N score matrix; uses running max + running sum with a 256-entry exp LUT (Q8.8)
- **16×16 INT8 systolic array** — skewed input feeding, INT8×INT8→INT32 MAC, configurable systolic array controller
- **4 parallel attention heads** — each head is an independent `flash_attn_core` instance, all fed and drained through a single AXI4-Stream interface
- **KV Cache** — `kv_cache.sv` stores K/V vectors token-by-token; `tile_controller` supports `mode=0` (prefill) and `mode=1` (decode) with runtime `kv_len`
- **AXI4-Stream interface** — slave handles Q→K→V byte-stream routing per head; master streams INT32 output row-major with 3-stage pipeline to compensate SRAM read latency
- **Sequence length scalable** — flat SRAM + global address offset; tested N=16/64/128/256

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

| Module | Test scenarios | Cases | Mismatches |
|--------|---------------|-------|------------|
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
# Verilator (macOS)
brew install verilator

# Python golden models
pip install numpy
```

### Run all tests

```bash
cd sim/verilator
make regression        # all testbenches (Week 1–7)
make coverage          # regression + coverage summary
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
python golden/generate_test_vectors.py      # N=16 Q/K/V + float expected
python golden/generate_hw_expected.py       # HW-accurate expected (exp LUT sim)
python golden/generate_multihead_data.py    # 4-head AXI test data
python golden/generate_kv_cache_test.py     # KV cache prefill/decode data
```

---

## File Structure

```
flashattn-accelerator/
├── rtl/
│   ├── systolic/         pe, systolic_array, array_controller
│   ├── quantization/     quantizer, dequantizer
│   ├── softmax/          exp_lut, online_softmax
│   ├── memory/           sram_1r1w, kv_cache, q_tile_buffer, kv_tile_buffer, output_buffer
│   ├── ctrl/             addr_gen, tile_controller
│   ├── interface/        axi4_stream_slave, axi4_stream_master
│   └── top/              flash_attn_top, flash_attn_core, flash_attn_top_axi
├── sim/verilator/
│   ├── Makefile           regression / coverage / per-module targets
│   ├── sim_main.cpp       systolic array testbench
│   ├── tb_quantizer.cpp
│   ├── tb_exp_lut.cpp
│   ├── tb_online_softmax.cpp
│   ├── tb_sram.cpp
│   ├── tb_addr_gen.cpp
│   ├── tb_kv_buf.cpp
│   ├── tb_flash_attn_top.cpp
│   ├── tb_axi_slave.cpp
│   ├── tb_flash_attn_top_axi.cpp
│   ├── tb_kv_cache.cpp
│   └── tb_kv_decode.cpp
├── golden/
│   ├── flash_attention.py
│   ├── int8_attention.py
│   ├── generate_exp_lut.py
│   ├── generate_test_vectors.py
│   ├── generate_hw_expected.py
│   ├── generate_multihead_data.py
│   └── generate_kv_cache_test.py
└── data/                 N16 / N64 / N128 / N256 / N64_axi / kv_decode
```
