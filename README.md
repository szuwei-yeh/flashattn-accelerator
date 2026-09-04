# FlashAttention Hardware Accelerator

## Overview

Cycle-accurate SystemVerilog implementation of tiled FlashAttention with INT8
matrix multiplication, online softmax, on-chip scratchpads, DMA-fed memory paths,
and optional output write-back. The design is verified with Verilator through
unit, subsystem, and end-to-end regression tests with zero mismatches against the
hardware-accurate reference model.

Standard attention materializes an `N x N` score matrix, requiring `O(N^2)`
intermediate storage and memory traffic. This accelerator instead processes
`16 x 16` tiles and carries the softmax running maximum and running sum across KV
tiles, so the complete score matrix never needs to be written to external memory.

## Key Results

- **1.26x end-to-end speedup** at `N=64, d=16`: byte-DMA/flat-core 15,332 cycles
  to vector-DMA/banked-core 12,140 cycles.
- **1.30x end-to-end speedup** at `N=64, d=64`: 60,421 to 46,412 cycles.
- **5.6x faster 256-byte DMA fill:** 312 to 56 cycles.
- **15.1x faster 256-byte scratchpad drain:** 256 to 17 cycles,
  demonstrating the benefit of 16-bank stripe access for tile staging.
- KV active/shadow double buffering provides a further **1.7-1.9% end-to-end
  improvement** over the non-prefetch banked path for the measured `N=64` cases.
- The current optimized banked-prefetch core meets the first-pass 100 MHz Design
  Compiler target with **+3.93 ns setup slack**, a **5.99 ns critical path**,
  **0 TNS**, and **0 violating paths**.
- Current core area is approximately **5.06M FreePDK45 library area units**.
  SRAM and ROM storage are logical blackboxes with zero reported macro area, so
  this is a non-signoff standard-cell estimate, not whole-accelerator physical
  area.

## Architecture

The compute path is shared by the original and DMA-banked integrations:

```text
Q tile ─┐
        ├─> 16x16 INT8 systolic array ─> dequantize ─> online softmax ─┐
K tile ─┘                                                               │
                                                                        v
P tile, V tile ─> same systolic array ─> output buffer
                                         (rescale,
                                          accumulate,
                                          normalize)
```

The same systolic array performs both `QK^T` and `PV`, avoiding a second matrix
engine. Each of its 256 processing elements performs an INT8 multiply with an
INT32 accumulator. For `HEAD_DIM=64`, the inner dimension is processed as four
16-element chunks and the PE accumulators carry partial sums across chunks.

Sixteen online-softmax engines operate in parallel, one per query row. Their
running maximum and running sum persist across KV tiles. When a new tile
**increases** the running maximum, previously accumulated output state is
rescaled by `exp(m_old - m_new)` before the new contribution is added. The final
output is normalized by the accumulated running sum.

### Two integration paths

The repository contains two related but distinct integration paths:

1. **Original multi-head AXI4-Stream path**
   - `flash_attn_top_axi`
   - Four parallel `flash_attn_core` instances.
   - Supports MHA and parameterized GQA input routing.
   - Includes the original prefill/decode and KV-cache path.

2. **Newer single-head DMA-banked path**
   - `flash_attn_top_dma_banked`, `flash_attn_top_dma_banked_prefetch`, and
     `flash_attn_top_dma_banked_wb`.
   - Adds an AXI read DMA, banked Q/K/V scratchpads, vector tile loading,
     residency tracking, optional KV prefetch, and a separately verified
     write-back path.
   - This is the path used for current memory-system optimization and
     synthesis studies.

The DMA-banked and write-back designs are currently single-head. They are not
integrated into the four-head GQA wrapper.

## Memory-System Optimization

The memory path was developed in independently verified stages:

```text
Byte-DMA baseline
external memory -> scalar DMA -> flat scratchpad -> byte-serial tile load

Vector banked path
external memory -> 64-bit AXI read DMA -> 128-bit stripe
                -> 16-bank Q/K/V scratchpads -> shared tile loader
                -> active tile registers -> systolic array

Prefetch variant
current K/V active registers + next K/V shadow registers
                -> overlap next-tile staging with current-tile compute
```

`banked_scratchpad` interleaves contiguous bytes across 16 byte-wide banks, so a
16-byte tile stripe can be read or written without a bank conflict.
`dma_engine_vec` combines two 64-bit AXI beats into one 128-bit scratchpad write,
and `banked_tile_loader` transfers one complete stripe per cycle. A single loader
serves Q loads, foreground K/V loads, and K/V prefetch.

The read-only prefetch integration is:

```text
AXI AR/R -> dma_engine_vec -> banked scratchpads
         -> flash_attn_core_banked_prefetch -> output read port
```

The implemented round-trip path is:

```text
AXI AR/R
   -> dma_engine_vec
   -> 16-bank Q/K/V scratchpads
   -> flash_attn_core_banked
   -> output_writeback_packer
   -> dma_write_engine
   -> AXI AW/W/B
```

The write-back top currently uses `flash_attn_core_banked`, not the newer
banked-prefetch core. Read/write DRAM models are simulation-only and are not part
of the synthesizable accelerator hierarchy.

## Performance

All cycle counts below are measured with cycle-accurate Verilator simulation and
the same hardware reference outputs for each compared configuration.

### DMA and banked-memory path

| Configuration | `N=64, d=16` | `N=64, d=64` |
|---|---:|---:|
| Byte DMA + flat core | 15,332 | 60,421 |
| Vector DMA + banked core | 12,140 | 46,412 |
| Speedup | **1.26x** | **1.30x** |

| Microbenchmark, 256 bytes | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| DMA fill | 312 cycles | 56 cycles | **5.6x** |
| Scratchpad drain | 256 cycles | 17 cycles | **15.1x** |

### KV double-buffer prefetch

Incremental comparison at zero modeled DRAM latency:

| Configuration | No prefetch | Prefetch | Reduction |
|---|---:|---:|---:|
| DMA-banked, `N=64, d=16` | 12,140 | 11,912 | 228 cycles / **1.88%** |
| DMA-banked, `N=64, d=64` | 46,412 | 45,608 | 804 cycles / **1.73%** |

Prefetch remains bit-exact for the tested `rd_latency = {0, 20, 100}` sweep. Its
gain is intentionally modest because tile loading is only one part of total
runtime; systolic compute and output-buffer update passes are unchanged.

### Original preloaded integration path

These results describe the original `flash_attn_top` path and should not be
combined with the DMA-banked numbers above.

| Non-causal prefill | `d=16` | `d=64` |
|---|---:|---:|
| `N=16` | 1,429 | - |
| `N=64` | 13,585 | 52,429 |
| `N=128` | 48,161 | 185,081 |
| `N=256` | 180,289 | 691,057 |

| Causal attention | `d=16` | `d=64` |
|---|---:|---:|
| `N=64` | 9,655 | 37,393 |
| `N=256` | 101,689 | 390,337 |

The original four-head AXI4-Stream top was measured at `N=64` with core
completion at 25,881 cycles for MHA with per-head `d=16`, 101,589 cycles for MHA
with per-head `d=64`, and 21,785 cycles for GQA with per-query-head `d=16`.

## Synthesis / PPA

The checked-in flow uses Synopsys Design Compiler R-2020.09-SP4, FreePDK45
`gscl45nm.db`, a 10 ns clock, and 1 ns input/output delay. See
[`syn/README.md`](syn/README.md) for profiles, filelists, and server commands.

### Pure-logic blocks

| Design | Setup slack | Total cell area (library units) |
|---|---:|---:|
| `systolic_array` | +5.15 ns | 442,795.819439 |
| `dma_engine_vec` | +6.62 ns | 4,181.462976 |
| `banked_tile_loader` | +8.00 ns | 970.043098 |

### Banked-prefetch core evolution

All rows use `SEQ_LEN=64`, `HEAD_DIM=16`, the same timing constraints, and
logical SRAM/ROM blackboxes.

| Version | Architectural change | Setup slack | Critical path | TNS | Violating paths | Cell area |
|---|---|---:|---:|---:|---:|---:|
| Initial fused core | Baseline | -35.31 ns | 45.28 ns | -75,405.72 ns | 4,096 | 7,722,500.0800 |
| No softmax normalization | Remove unused per-tile normalization hardware | +2.29 ns | 6.62 ns | 0 | 0 | 5,463,487.8893 |
| Combined scale v1 | One shared 16x16 scale multiply; 256 32x32 lane multiplies | +0.98 ns | 7.94 ns | 0 | 0 | 5,061,648.8099 |
| **Registered combined scale v2** | Capture combined scale when a transaction is accepted | **+3.93 ns** | **5.99 ns** | **0** | **0** | **5,060,883.8510** |

The current synthesis top is `flash_attn_core_banked_prefetch`, not the
DMA-connected top. It includes the 16x16 array, dequantizers, softmax logic,
output logic, Q/K/V bank wrappers, tile loader, explicit active/shadow tile
registers, and prefetch control. It excludes the vector DMA, AXI protocol logic,
write-back engine, and external memory.

The Q/K/V memories, output memory, and exponential LUT are present as logical
blackboxes, but no physical memory `.db` is linked. Their macro area is therefore
reported as zero and their real access timing and power are absent. The 5.06M
result is useful for comparing the synthesized standard-cell core logic across
these revisions, but it is not a complete core-with-memory or whole-accelerator
physical estimate.

No placement, clock-tree synthesis, routing, extracted interconnect, or
activity-annotated power analysis has been performed.

## Verification

The regression compares RTL output bit-for-bit against fixed-point reference
models and covers unit blocks, complete compute paths, DMA/memory subsystems,
prefetch behavior, and output write-back.

| Coverage area | Representative targets | Result |
|---|---|---|
| Arithmetic and compute | `systolic_array`, `tb_dequantizer_combined`, `tb_softmax` | PASS, 0 mismatches |
| Original single-head core | `tb_top_N16/N64/N128/N256`, `d=16/64`, causal variants | PASS, 0 mismatches |
| Multi-head AXI4-Stream | `axi_top_N64`, `axi_top_N64_d64`, `axi_top_N64_gqa` | PASS, 0 mismatches |
| KV cache/decode | `tb_kv_cache`, `tb_kv_decode`, `tb_kv_decode_d64` | PASS, 0 mismatches |
| DMA and scratchpads | `tb_dma_unit`, `tb_banked_scratchpad`, `dma_bench`, `dma_vec_bench` | PASS, 0 mismatches |
| Banked core and prefetch | core-only and DMA-fed `d=16/64` targets | PASS, 0 mismatches |
| Write-back | `tb_dma_write_unit`, `wb_drain_bench`, `dma_banked_wb_top_*` | PASS, 0 mismatches |
| Full `make regression` | all targets above | PASS |

Write-back was verified bottom-up and end-to-end:

| End-to-end target | Read bytes | Write-back bytes | Write beats | Golden comparison |
|---|---:|---:|---:|---|
| `dma_banked_wb_top_N64`, `d=16` | 3,072 | 4,096 | 512 | exact |
| `dma_banked_wb_top_N64_d64`, `d=64` | 12,288 | 16,384 | 2,048 | exact |

## Limitations

- The DMA-banked path is currently proven only for `N=64` with `d=16` and
  `d=64`; other DMA-banked shapes are not claimed.
- The DMA-banked, prefetch, and write-back paths are single-head. Four-head GQA
  support belongs to the separate original AXI4-Stream integration.
- `axi_mem_model.sv` and `axi_mem_model_rw.sv` are behavioral simulation models,
  not real DDR/HBM controllers.
- Current DC results omit physical SRAM/ROM macro area, timing, and power.
- The write-back path has not been characterized with real memory macros.
- The current write-back top uses the non-prefetch banked core; current
  banked-prefetch synthesis optimizations have not yet been integrated into that
  round-trip wrapper.
- Runtime `cfg_seq_len` configures DMA transfer scheduling; the compute sequence
  extent remains a compile-time parameter, so the validated use is currently
  `cfg_seq_len == SEQ_LEN`.
- Current PPA is pre-layout: no placement, CTS, routing, or extracted
  interconnect is included.

## How to Run

### Prerequisites

```bash
brew install verilator  # macOS
python3 -m pip install numpy
```

### Full regression

```bash
cd sim/verilator
make regression
```

### Selected targets

```bash
# Original single-head and causal paths
make tb_top_N64
make tb_top_N64_d64
make tb_top_causal_N64

# Original four-head AXI4-Stream / GQA path
make axi_top_N64
make axi_top_N64_d64
make axi_top_N64_gqa

# Current banked-prefetch core and DMA integration
make core_banked_prefetch_N64
make core_banked_prefetch_N64_d64
make dma_banked_prefetch_top_N64
make dma_banked_prefetch_top_N64_d64

# Write-back path
make tb_dma_write_unit
make wb_drain_bench
make dma_banked_wb_top_N64
make dma_banked_wb_top_N64_d64
```

### Regenerate test vectors

```bash
python3 golden/generate_test_vectors.py
python3 golden/generate_hw_expected.py
python3 golden/generate_multihead_data.py
python3 golden/generate_kv_cache_test.py
```

## RTL Organization

| Directory | Contents |
|---|---|
| `rtl/systolic/` | Processing element, 16x16 systolic array, input-skew controller |
| `rtl/quantization/` | Quantizer and combined-scale dequantizer |
| `rtl/softmax/` | Exponential LUT and cross-tile online softmax |
| `rtl/memory/` | Behavioral SRAM, flat/banked tile storage, KV cache, output buffer |
| `rtl/ctrl/` | Address generation and original/banked/prefetch tile controllers |
| `rtl/interface/` | AXI4-Stream adapters, read/write DMA engines, simulation DRAM models |
| `rtl/core/` | Original, banked, and banked-prefetch compute cores |
| `rtl/top/` | Single-head, multi-head AXI4-Stream, DMA, banked, prefetch, and write-back tops |
| `rtl/bench/` | Synthesizable benchmark wrappers used by subsystem tests |

## Repository Structure

```text
flashattn-accelerator/
├── rtl/                 SystemVerilog RTL and simulation memory models
├── sim/verilator/       C++/SystemVerilog testbenches and regression Makefile
├── golden/              Fixed-point Python reference and vector generators
├── data/                Pre-generated test vectors and expected outputs
└── syn/                 Design Compiler scripts, filelists, and blackbox stubs
```
