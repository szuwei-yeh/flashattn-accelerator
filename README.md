# FlashAttention Hardware Accelerator

## Overview

This repository contains a cycle-accurate SystemVerilog implementation of tiled
FlashAttention. The current optimized path combines INT8 matrix multiplication,
online softmax, banked on-chip storage, an AXI read-side vector DMA, K/V
double-buffered prefetch, and an optional output write-back path.

The project follows an RTL/ASIC optimization workflow: profile an end-to-end
design, isolate the memory and compute bottlenecks, change one architectural
stage at a time, measure cycle and synthesis impact, and preserve bit-exact
behavior with regression tests.

## Why FlashAttention

Standard attention materializes an `N x N` score matrix, which requires
quadratic intermediate storage and memory traffic. This accelerator processes
`16 x 16` tiles and carries each query row's running softmax maximum and sum
across K/V tiles. The complete attention matrix is therefore never written to
external memory.

For the optimized single-head path, the main measured results at `N=64` are:

- Vector DMA and banked storage reduce the memory-path milestone from 15,332 to
  12,140 cycles at `d=16`, and from 60,421 to 46,412 cycles at `d=64`.
- K/V prefetch adds a further 1.73–1.88% reduction at that design stage.
- A later fused output update reduces the current full-top result to 7,800
  cycles at `d=16` and 29,160 cycles at `d=64`. This later gain is not
  attributed to the DMA redesign alone.
- The optimized compute core uses 5.061M standard-cell area units and has a
  5.99 ns critical path. Adding the DMA and top-level system logic raises the
  full-top result to 5.073M, a net increase of approximately 0.23%.

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

## Baseline Bottleneck

The first DMA-fed implementation had an AXI read input, but movement inside the
accelerator remained serialized. Incoming data was unpacked into byte-wide
local writes, stored in a flat scratchpad, and staged into tile registers one
element per cycle. The external interface was wider than the local data path,
so AXI bandwidth could not translate directly into lower tile-loading latency.

```text
Baseline
AXI read -> scalar DMA unpack -> flat scratchpad
         -> byte-serial tile staging -> compute
```

Profiling separated DMA fill time from local scratchpad drain time. That made it
possible to optimize the storage and staging path without changing the compute
schedule or numerical behavior.

## Memory-System Optimization

The optimized path widens local movement end to end:

```text
Optimized
AXI read -> Vector DMA -> 128-bit stripe write
         -> independent 16-bank Q/K/V scratchpads
         -> shared 16 B/cycle tile loader -> compute
```

`dma_engine_vec` combines two 64-bit AXI read beats into one 128-bit local
write. Each scratchpad interleaves contiguous bytes across 16 banks, allowing a
complete 16-byte tile stripe to be accessed without a bank conflict. A single
`banked_tile_loader` serves Q loading, foreground K/V loading, and K/V prefetch.

For a 256-byte transfer, the measured DMA fill falls from 312 to 56 cycles and
the local scratchpad drain falls from 256 to 17 cycles. The 17-cycle drain is 16
data stripes plus the synchronous-read boundary cycle.

## K/V Prefetch and Double Buffering

The compute core has active K/V registers and a shadow K/V register set. Once
the next K/V tile is resident in its scratchpad, the shared loader may fill the
shadow set while the current tile proceeds through QK, online softmax, PV, and
output update.

```text
DMA completes tile n+1
        │
        ▼
tile n+1 becomes resident
        │
        ▼
shared loader fills shadow K/V while tile n computes
        │
        ▼
next compute-tile boundary
        │
        ▼
one-cycle shadow-to-active promotion
```

Prefetch is gated by the resident-tile count. If the next tile has not arrived,
the controller waits for residency and uses the foreground load path. This
preserves correctness across the tested memory-latency sweep instead of
allowing compute to consume incomplete K/V data.

## Performance Results

All cycle counts are from cycle-accurate Verilator simulation with bit-exact
comparison against the fixed-point reference outputs. The main end-to-end
comparisons use `N=64`, `TILE_SIZE=16`, and zero modeled DRAM latency.

### Memory microbenchmarks

| 256-byte benchmark | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| DMA fill | 312 cycles | 56 cycles | **5.57x** |
| Scratchpad drain / tile load | 256 cycles | 17 cycles | **15.06x** |

### Memory-path milestone

| Configuration | `d=16` | `d=64` |
|---|---:|---:|
| Byte DMA + flat core | 15,332 cycles | 60,421 cycles |
| Vector DMA + banked core | 12,140 cycles | 46,412 cycles |
| Speedup | **1.263x** | **1.302x** |

These rows isolate the vector-DMA and banked-storage stage. They do not include
the later K/V prefetch or fused output-update improvements.

### Incremental K/V prefetch

| Configuration | No prefetch | Prefetch | Reduction |
|---|---:|---:|---:|
| `N=64, d=16` | 12,140 | 11,912 | 228 cycles / **1.878%** |
| `N=64, d=64` | 46,412 | 45,608 | 804 cycles / **1.732%** |

The modest incremental gain is consistent with tile staging being only one
part of the total runtime. All tested `rd_latency={0,20,100}` cases remained
bit-exact.

### Later fused output update

After the memory-system milestones above, rescaling the previous output and
accumulating the new `PV` contribution were fused into one output-buffer
traversal.

| Banked-prefetch full top | Before fusion | After fusion | Reduction |
|---|---:|---:|---:|
| `N=64, d=16` | 11,912 | **7,800** | 34.52% |
| `N=64, d=64` | 45,608 | **29,160** | 36.06% |

This is a later compute/output-path optimization. Its complete improvement must
not be attributed to the DMA or banked scratchpads.

## Core PPA Optimization

The synthesis optimization targeted the current
`flash_attn_core_banked_prefetch` at `SEQ_LEN=64`, `HEAD_DIM=16`. Profiling the
mapped design showed that dequantization arithmetic, rather than DMA or
scheduler control, dominated the standard-cell cost and critical path.

The dequantization scale arithmetic was restructured around a shared combined
scale product, then registered when the transaction was accepted. This removed
a long live-input/fanout path without changing workload cycles.

The "Before" row below already excludes unused per-tile softmax normalization;
the measured area reduction is primarily from the combined-scale arithmetic
restructuring.

| Metric | Before | Optimized | Change |
|---|---:|---:|---:|
| Standard-cell area | 5,463,487.889 | 5,060,883.851 | **-7.369%** |
| Critical path | 6.62 ns | 5.99 ns | **-0.63 ns** |
| WNS at 100 MHz | +2.29 ns | **+3.93 ns** | +1.64 ns |
| TNS | 0 ns | 0 ns | unchanged |
| Setup violations | 0 | 0 | unchanged |
| Core cycles, `d=16 / d=64` | 7,589 / 28,337 | 7,589 / 28,337 | unchanged |

## Full DMA + Optimized-Core Synthesis

The full synthesis top is `flash_attn_top_dma_banked_prefetch`. It uses the same
Synopsys Design Compiler R-2020.09-SP4 flow, FreePDK45 `gscl45nm.db`, 10 ns
clock, 1 ns input/output delay, and logical memory blackboxes as the standalone
core comparison.

| Metric | Optimized core only | DMA + optimized core |
|---|---:|---:|
| Standard-cell area | 5,060,883.851 | **5,072,707.864** |
| Net area increase vs. standalone core | — | **11,824.013 (+0.2336%)** |
| Critical path | 5.99 ns | **5.99 ns** |
| WNS at 100 MHz | +3.93 ns | **+3.93 ns** |
| TNS | 0 ns | **0 ns** |
| Setup violations | 0 | **0** |
| Critical-path block | Core dequantizer | **Core dequantizer** |

The preserved full-top hierarchy reports 3,682.128 area units in `u_dma` and
8,176.614 in top-local configuration, scheduler, counters, glue, and arithmetic.
Because the core maps 34.728 area units smaller in the full-top context than
alone, the hierarchical system-side sum is 11,858.742 while the directly
comparable net top-versus-standalone-core increase is 11,824.013.

The DMA and system-side control therefore add approximately 0.23% standard-cell
area relative to the optimized compute core while preserving the 100 MHz
synthesis target. The critical path remains in a core dequantizer; DMA, AXI
read-side logic, and scheduler control do not become the frequency bottleneck.

These values are logical-synthesis standard-cell estimates, not final physical
accelerator area.

## Verification

### Functional regression

The Verilator regression checks fixed-point RTL output bit-for-bit against
generated reference data across unit, subsystem, and end-to-end tests.

| Coverage area | Representative configurations | Result |
|---|---|---|
| Arithmetic and shared compute array | INT8 array, dequantization, online softmax | PASS, 0 mismatches |
| Original prefill/decode integrations | single-head, causal, MHA/GQA, `d=16/64` | PASS, 0 mismatches |
| DMA and banked scratchpads | scalar/vector DMA, banked storage, shared loader | PASS, 0 mismatches |
| Banked core and K/V prefetch | core-only and DMA-fed `N=64, d=16/64` | PASS, 0 mismatches |
| Output write-back | unit, drain benchmark, full DRAM round trip | PASS, 0 mismatches |
| Full `make regression` | all targets above | **PASS** |

### Lightweight formal checks

The scheduler/prefetch boundary has three focused SymbiYosys properties. The
harness models the `N=64, d=16` prefill controller with legal abstract completion
handshakes. Bounded model checking to depth 48 found no counterexample for:

- compute starting only with a ready, matching active K/V tile;
- shadow-to-active promotion requiring a complete, matching shadow tile; and
- compute and prefetch tile indices remaining within the configured range.

Non-vacuity covers reached `array_start`, `pf_start`, and `kv_swap_banks`. This
is a bounded control-safety result, not an unbounded proof of the full datapath.

### AXI read-channel characterization

Directed tests confirm normal completion when `RLAST` arrives on the expected
final beat. They also show that the current prototype ignores early or missing
`RLAST`, completes from its internal beat count before a late `RLAST`, and
ignores `SLVERR` and `DECERR`. No read-error status is exposed.

These tests characterize the current implementation; they do not claim that
the read DMA is fully AXI-robust.

## Limitations

- The Q/K/V and output SRAMs are logical synthesis blackboxes. Physical SRAM
  storage area, access timing, and power are not included.
- The exponential LUT ROM is also a logical blackbox; its physical area and
  access timing are not included.
- Synthesis is pre-layout with an ideal clock. There is no placement, clock-tree
  synthesis, routing, extracted parasitics, input driving-cell model, output
  load, or clock uncertainty.
- The full top still reports max-capacitance violations, so the result is not
  design-rule or signoff clean. No final mm², post-route Fmax, or signoff PPA is
  claimed.
- Architectural cycle measurements use behavioral memory models with a
  configurable read latency, not a DDR/HBM controller or a characterized memory
  subsystem.
- The current AXI read DMA does not implement complete `RLAST` or `RRESP` error
  detection and propagation.
- The optimized DMA/prefetch path is single-head and is primarily validated for
  `N=64` with `d=16` and `d=64`. The four-head MHA/GQA path is a separate,
  earlier integration.
- Runtime `cfg_seq_len` controls DMA scheduling, while the compute extent
  remains a compile-time parameter; validated runs use
  `cfg_seq_len == SEQ_LEN`.
- The write-back wrapper currently uses the non-prefetch banked core rather
  than the latest optimized banked-prefetch core.

## Future Work

- Integrate characterized SRAM and ROM macros, then run placement, CTS,
  routing, parasitic extraction, and post-route timing analysis.
- Resolve max-capacitance and high-fanout physical-design issues.
- Add AXI `RLAST`/`RRESP` checking, error propagation, and broader protocol
  stress testing.
- Integrate output write-back with the optimized prefetch core.
- Extend the banked DMA path beyond the currently validated single-head N64
  configurations.

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
make -C sim/verilator dma_banked_prefetch_top_N64
make -C sim/verilator dma_banked_prefetch_top_N64_d64
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
