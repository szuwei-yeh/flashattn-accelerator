# FlashAttention Hardware Accelerator

## Overview

This repository contains a cycle-accurate SystemVerilog implementation of tiled
INT8 FlashAttention. One shared `16 x 16` systolic array is reused for `QK^T`
and `PV`; sixteen online-softmax lanes carry the running state across tiles.
The optimized data path combines an AXI read-side vector DMA, 16-bank Q/K/V
scratchpads, K/V prefetch, and a fused output update.

The project follows an RTL/ASIC optimization workflow: profile an end-to-end
design, isolate the memory and compute bottlenecks, change one architectural
stage at a time, measure cycle and logical-synthesis impact, and preserve exact
fixed-point behavior with regression and bounded formal checks.

Standard attention materializes an `N x N` score matrix, which requires
quadratic intermediate storage and memory traffic. This accelerator processes
`16 x 16` tiles and carries each query row's running softmax maximum and sum
across K/V tiles. The complete attention matrix is therefore never written to
external memory.

## Key Results

Unless noted otherwise, cycle results use `N=64`; synthesis uses the d16 core
with a 10 ns target and logical SRAM/ROM blackboxes.

| Optimization / result | Before | After | Measured impact |
|---|---:|---:|---:|
| 256-byte DMA fill | 312 cycles | 56 cycles | **5.57x** |
| 256-byte scratchpad drain | 256 cycles | 17 cycles | **15.06x** |
| Memory-stage E2E, d16 | 15,332 cycles | 12,140 cycles | **1.263x** |
| Memory-stage E2E, d64 | 60,421 cycles | 46,412 cycles | **1.302x** |
| Fused output update, d16 | 11,912 cycles | 7,800 cycles | **-34.52%** |
| Fused output update, d64 | 45,608 cycles | 29,160 cycles | **-36.06%** |
| Remove unused divider path | 45.28 ns | 6.62 ns | **-29.25% area; 4,096 -> 0 setup violations** |
| Final core | — | 5.96 ns, +3.96 ns WNS | **Meets 100 MHz** |
| Final DMA-integrated top | — | 6.02 ns, +3.90 ns WNS | **Meets 100 MHz; 0.23743% standard-cell area overhead** |
| Exact fixed-point regression | — | d16: 0 / 1,024; d64: 0 / 4,096 | **0 exact mismatches** |

## Accelerator Architecture

The optimized read-side hierarchy is:

```text
External memory
      │  AXI4 AR/R, 64-bit data
      ▼
Vector DMA (two read beats -> one 128-bit local write)
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
 Q scratchpad    K scratchpad    V scratchpad
   16 banks        16 banks        16 banks
      └──────────────┴──────────────┘
                     │ 16 B/cycle
                     ▼
              Shared tile loader
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Q tile register     K/V active + shadow registers
          └──────────┬──────────┘
                     ▼
             Shared 16x16 INT8
               systolic array
                     │
          ┌──────────┴──────────┐
          │ Phase 1: Q and K    │
          │ QK^T -> dequantize  │
          │ -> online softmax   │
          └──────────┬──────────┘
                     │ per-tile P operand
          ┌──────────▼──────────┐
          │ Phase 2: P and V    │
          │ reuse the same      │
          │ systolic array      │
          └──────────┬──────────┘
                     ▼
          Fused rescale + accumulate
                     │
                     ▼
               Output buffer
          (final running-sum normalization)
```

There is one physical `16 x 16` systolic array, reused sequentially for both
`QK^T` and `PV`. Each of its 256 processing elements performs an INT8 multiply
with an INT32 accumulator. For `HEAD_DIM=64`, the inner dimension is evaluated
as four 16-element chunks while the array retains its partial sums.

Sixteen online-softmax engines operate in parallel, one per query row. They
produce per-tile exponent weights and maintain the cross-tile running maximum
and running sum. When a new maximum is observed, the existing output state is
rescaled before the new `PV` contribution is accumulated.

The repository also retains an earlier four-head AXI4-Stream integration with
MHA/GQA routing and a prefill/decode path. The banked DMA, prefetch, synthesis,
and memory-system results below refer to the newer single-head hierarchy.

## Optimization Journey

### Memory-System Bottleneck

The first DMA-fed implementation had a wide AXI input but serialized movement
inside the accelerator: byte-wide local writes fed a flat scratchpad, followed
by one-element-per-cycle tile staging. Profiling separated DMA fill from local
scratchpad drain and showed that external width was not reaching the compute
array.

```text
Baseline:  AXI read -> scalar DMA -> flat scratchpad -> byte-serial staging
```

### Vector DMA and Banked Scratchpads

The optimized path widens local movement end to end:

```text
Optimized: AXI read -> vector DMA -> 128-bit stripe write
                    -> 16-bank Q/K/V scratchpads
                    -> shared 16 B/cycle tile loader -> compute
```

`dma_engine_vec` combines two 64-bit AXI beats into one 128-bit local write.
Contiguous bytes are interleaved across 16 banks, so the shared tile loader can
move a complete stripe without a bank conflict. This changed a 256-byte DMA
fill from 312 to 56 cycles and scratchpad drain from 256 to 17 cycles without
changing compute scheduling or numerical behavior.

### K/V Prefetch

Active and shadow K/V registers let the shared loader stage tile `n+1` while
tile `n` computes. A resident-tile count gates prefetch; if data has not arrived,
the controller waits and falls back to foreground loading rather than consuming
an incomplete tile. The tested memory-latency sweep remained exact.

### Fused Output Update

The original output phase traversed the output buffer once to rescale old state
and again to accumulate the new `PV`. A fused read-modify-write performs:

```text
O_new = rescale_factor * O_old + PV_new
```

This eliminates intermediate output materialization and reduces output-buffer
traffic and controller cycles. It is a later compute/output-path optimization,
not part of the DMA-only speedup.

### Synthesis Bottleneck

The initial d16/N64 logical synthesis failed the 100 MHz target with a 45.28 ns
critical path and 4,096 setup violations. Timing traces led into online-softmax
normalization, where the active hierarchy still contained an output path that
the optimized core did not consume.

### Removing the Redundant Softmax Divider Path

Tracing that output revealed 256 unused combinational dividers. Removing the
redundant normalized-softmax output reduced critical-path delay to 6.62 ns,
eliminated all setup violations, reduced total standard-cell area by 29.25%,
and reduced the softmax hierarchy by 84.19%. Estimated power moved from 172.08
to 153.67 mW.

### Shared Q/K Scale

Dequantization then became the main mapped area hotspot. Sharing
`scale_q * scale_k` across the 256 dequantizer lanes reduced total area by a
further 7.36% at that stage and dequantizer area by 9.94%. The combined scale is
captured at accepted start, preserving the fixed-point policy and workload
cycles. The final 5.96 ns core result is reported after the complete interface
and configuration cleanup; it is not attributed to any single cleanup change.

## Performance Results

All cycle counts come from cycle-accurate Verilator simulation against the
hardware-aware fixed-point reference. End-to-end comparisons use `N=64`,
`TILE_SIZE=16`, and zero modeled DRAM latency unless stated otherwise.

### Memory-path evolution

| Result | Baseline | Optimized | Improvement |
|---|---:|---:|---:|
| 256-byte DMA fill | 312 | 56 | **5.57x** |
| 256-byte scratchpad drain / tile load | 256 | 17 | **15.06x** |
| E2E, d16 | 15,332 | 12,140 | **1.263x** |
| E2E, d64 | 60,421 | 46,412 | **1.302x** |

The E2E rows compare the byte-DMA/flat core with the vector-DMA/banked core at
the same architecture stage. They exclude the later prefetch and fused-output
changes.

### Incremental K/V prefetch

| Configuration | No prefetch | Prefetch | Reduction |
|---|---:|---:|---:|
| `N=64, d=16` | 12,140 | 11,912 | 228 cycles / **1.878%** |
| `N=64, d=64` | 46,412 | 45,608 | 804 cycles / **1.732%** |

The incremental gain is modest because tile staging is only one part of total
runtime. The `rd_latency={0,20,100}` sweep remained exact.

### Fused output update

| Banked-prefetch full top | Before fusion | After fusion | Reduction |
|---|---:|---:|---:|
| `N=64, d=16` | 11,912 | **7,800** | 34.52% |
| `N=64, d=64` | 45,608 | **29,160** | 36.06% |

These results belong to the later fused-output RTL revision and are not
reported as DMA-only improvement.

## Final ASIC Synthesis / PPA

The current core and `flash_attn_top_dma_banked_prefetch` use the same Design
Compiler R-2020.09-SP4 flow, FreePDK45 `gscl45nm.db`, 10 ns clock, 1 ns I/O
delay, and logical memory blackboxes.

### Core optimization evolution

| Core synthesis stage | Area | Critical path | WNS @ 100 MHz | Setup violations |
|---|---:|---:|---:|---:|
| Baseline | 7,722,500 | 45.28 ns | -35.31 ns | 4,096 |
| Remove unused normalized-softmax divider path | 5,463,488 | 6.62 ns | +2.29 ns | 0 |
| Shared Q/K combined scale | 5,061,648.8 | 7.94 ns | +0.98 ns | 0 |
| Final cleanup RTL | **5,061,246.151** | **5.96 ns** | **+3.96 ns** | **0** |

### Final core versus DMA-integrated top

| Metric | Final core only | DMA + final core |
|---|---:|---:|
| Standard-cell area | 5,061,246.151 | **5,073,263.046** |
| Integration overhead | — | **12,016.896 (+0.23743%)** |
| Critical path | 5.96 ns | **6.02 ns** |
| WNS at 100 MHz | +3.96 ns | **+3.90 ns** |
| TNS | 0 ns | **0 ns** |
| Setup violations | 0 | **0** |
| Critical-path block | Core dequantizer | **Core dequantizer** |

The full-top critical path remains inside the core dequantizer. Integrating the
vector DMA, AXI read side, scheduler, and configuration logic therefore adds
about 0.24% standard-cell area without becoming the timing bottleneck. Detailed
hierarchy accounting is kept in the local synthesis reports.

## Verification

### Functional regression

The final optimized-core regression uses exact `hw[i] == expected[i]` equality,
including entries whose expected value is zero. "Bit-exact" here means exact
against the hardware-aware fixed-point reference, which models Q8.8 scales, LUT
exponential, saturation, truncation/rounding, running softmax, and `scale_v`; it
does not mean bit-exact against FP32 attention.

| Final-cleanup check | Result | Cycles |
|---|---:|---:|
| Optimized core, `N=64, d=16` | PASS, 0 / 1,024 exact mismatches | 7,589 |
| Optimized core, `N=64, d=64` | PASS, 0 / 4,096 exact mismatches | 28,337 |
| Optimized core causal, `N=64, d=16` | PASS, 0 / 1,024 exact mismatches | 5,195 |
| DMA + optimized core causal, latency 0 / 20 / 100 | PASS, 0 exact mismatches | 5,406 / 5,646 / 6,606 |
| Full existing regression | **PASS** | — |
| Verilator lint | **PASS** | — |
| Formal BMC depth 48 / cover | **PASS / PASS** | cover steps 19 / 27 / 41 |

Directed contract tests also pass for invalid, shorter, and misaligned
`cfg_seq_len`; unaligned AXI bases; accepted-start locking of Q/K/V scales,
causal mode, DMA bases, and related configuration; busy-time extra starts; and
the documented single-shot-until-reset behavior. Unsupported shorter lengths
are rejected without hanging.

### Lightweight formal checks

The scheduler/prefetch boundary has three focused formal properties. The current
environment did not provide an `sby` executable, so the final-cleanup run used
the equivalent current-source Yosys + `yosys-smtbmc` + Z3 flow rather than a
native SymbiYosys command. Bounded model checking to depth 48 passed for:

- compute starting only with a ready, matching active K/V tile;
- shadow-to-active promotion requiring a complete, matching shadow tile; and
- compute and prefetch tile indices remaining within the configured range.

The cover run also passed, reaching the three coverage points at steps 19, 27,
and 41. This is a bounded control-safety result, not an unbounded proof of the
full datapath.

### AXI read-channel characterization

Directed tests confirm normal completion when `RLAST` arrives on the expected
final beat. They also show that the current prototype ignores early or missing
`RLAST`, completes from its internal beat count before a late `RLAST`, and
ignores `SLVERR` and `DECERR`. No read-error status is exposed.

These tests characterize the current implementation; they do not claim that
the read DMA is fully AXI-robust.

## Supported Configuration

The optimized banked-prefetch prototype intentionally supports a bounded
parameter space:

- `TILE_SIZE=16` and `HEAD_DIM` in `{16, 64}`, with
  `HEAD_DIM % TILE_SIZE == 0`;
- `NUM_BANKS=16` for the current Q/K/V scratchpad organization;
- a nonzero, tile-aligned compile-time `SEQ_LEN`; runtime `cfg_seq_len` must
  equal that compile-time value;
- `SEQ_LEN * HEAD_DIM <= SRAM_DEPTH`, `SRAM_DEPTH % 16 == 0`, and
  `SRAM_DEPTH <= 4096` under the current 12-bit SRAM address interface;
- `AXI_ADDR_W=32`, `AXI_DATA_W=64`, and 16-byte-aligned Q/K/V base addresses.

At accepted start, the design locks Q/K/V base and DMA configuration,
`scale_q`, `scale_k`, `scale_v`, causal mode, and relevant core mode/length
state. External configuration changes and extra `start` pulses while busy do
not affect the active transaction. Invalid configurations assert `cfg_error`
and are not accepted.

## Limitations

- SRAM and exponential-LUT ROM instances are logical blackboxes. Reported area,
  timing, and power exclude physical memory macro area and access behavior.
- Results are logical post-synthesis with an ideal clock and no placement, CTS,
  routing, or extracted parasitics. They are not physical-signoff results;
  max-capacitance and high-fanout issues remain for implementation.
- Architectural cycle measurements use behavioral memory models with a
  configurable read latency, not a DDR/HBM controller or a characterized memory
  subsystem.
- AXI read-side `RLAST`/`RRESP` error detection and propagation are not
  production-grade.
- The optimized single-head path has a bounded configuration space and is
  primarily validated at `N=64`, `d=16/64`; the earlier MHA/GQA integration is
  separate.
- The optimized DMA/prefetch top does not implement runtime-variable sequence
  length. A transaction is accepted only when `cfg_seq_len == SEQ_LEN`; zero,
  shorter, longer, or misaligned values assert `cfg_error` and are rejected.
- The optimized core and DMA/prefetch top are single-shot until reset. The
  output SRAM is accumulation state and has no transaction-clear traversal;
  accepting a second transaction without reset would otherwise add to stale
  output data.
- The write-back wrapper currently uses the non-prefetch banked core rather
  than the latest optimized banked-prefetch core.

Detailed synthesis warnings, tool-specific parameter-guard behavior, and mapped
timing audits are kept in the local reports; the reproducible flow and its
assumptions are documented in `syn/README.md`.

## Future Work

- Integrate characterized SRAM and ROM macros, then run placement, CTS,
  routing, parasitic extraction, and post-route timing analysis.
- Resolve max-capacitance and high-fanout physical-design issues.
- Add AXI `RLAST`/`RRESP` checking, error propagation, and broader protocol
  stress testing.
- Integrate output write-back with the optimized prefetch core.
- Support back-to-back transactions by adding an output-SRAM clear traversal,
  an epoch/tag scheme, or explicit output-buffer ownership/buffering. A simple
  `DONE -> IDLE` transition is insufficient because the second result otherwise
  accumulates onto prior output SRAM contents.
- Extend the banked DMA path beyond the currently validated single-head N64
  configurations.

## Engineering Highlights

- Identified local-memory movement as the initial performance bottleneck and
  redesigned the path with a vector DMA, 16-bank scratchpads, and K/V prefetch.
- Fused output rescale and accumulation into one buffer traversal to eliminate
  intermediate SRAM materialization.
- Used synthesis timing reports to identify an unused normalized-softmax path
  containing 256 combinational dividers and eliminated the failing 100 MHz path.
- Shared the Q/K scale product while preserving the hardware-aware fixed-point
  behavior.
- Closed the final core and DMA-integrated hierarchy at 100 MHz without moving
  the critical path into DMA or scheduler logic.
- Verified supported d16/d64 configurations against a hardware-aware fixed-point
  reference with exact, causal, transaction-contract, and bounded formal checks.

## Reproducing the Tests

### Prerequisites

```bash
brew install verilator  # macOS
python3 -m pip install numpy
```

Formal jobs additionally require SymbiYosys, Yosys, and Z3.

### Full Verilator regression

```bash
make -C sim/verilator regression
```

### Selected optimized-path targets

```bash
make -C sim/verilator dma_vec_bench
make -C sim/verilator dma_bench
make -C sim/verilator core_banked_prefetch_N64
make -C sim/verilator core_banked_prefetch_N64_d64
make -C sim/verilator core_banked_prefetch_causal_N64
make -C sim/verilator dma_banked_prefetch_top_N64
make -C sim/verilator dma_banked_prefetch_top_N64_d64
make -C sim/verilator dma_banked_prefetch_causal_top_N64
make -C sim/verilator tb_dma_banked_prefetch_contract
make -C sim/verilator tb_dma_vec_axi_protocol
```

### Formal scheduler checks

```bash
make -C formal all
```

## RTL Organization

| Directory | Contents |
|---|---|
| `rtl/systolic/` | Processing elements, shared 16x16 systolic array, and array controller |
| `rtl/quantization/` | Quantization and combined-scale dequantization |
| `rtl/softmax/` | Exponential LUT and cross-tile online-softmax state |
| `rtl/memory/` | Behavioral SRAM, flat/banked scratchpads, tile loaders, and output buffer |
| `rtl/ctrl/` | Address generation and original/banked/prefetch controllers |
| `rtl/interface/` | AXI4-Stream adapters, read/write DMA engines, and simulation memory models |
| `rtl/core/` | Original, banked, and banked-prefetch compute cores |
| `rtl/top/` | Single-head, multi-head, DMA, prefetch, and write-back integrations |
| `formal/` | Focused scheduler/prefetch formal harness and SymbiYosys jobs |
| `sim/verilator/` | C++/SystemVerilog testbenches and regression Makefile |
| `golden/` | Fixed-point reference model and test-vector generators |
| `syn/` | Design Compiler profiles, filelists, constraints, and reporting flow |

See [`syn/README.md`](syn/README.md) for the reproducible logical-synthesis
profiles and constraints.
