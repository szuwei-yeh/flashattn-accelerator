# Design and optimization

[Home](../README.md) · [Verification](VERIFICATION.md) · [Results and evidence](RESULTS_EVIDENCE.md)

This guide describes the optimized single-head DMA/banked/prefetch path.
The architecture figure follows the measured RTL at `dfbd28e`.

![FlashAttention architecture: simulation environment outside the DUT, data movement through active/shadow staging, shared QK/PV compute and fused output update](figures/architecture.svg)

The DMA loads Q first, then streams K/V tile pairs. Compute starts once the first
pair is resident; later DMA transfers overlap compute. `kv_tiles_ready` increases
only after both K and V are complete. Prefetch moves an already resident next
K/V tile from scratchpad to shadow registers during processing of the current
tile. Promotion copies shadow to active. A shared-loader interlock prevents
foreground loading from colliding with prefetch.

## Detailed architecture and simulation boundary

The external memory box is the **simulation-only behavioral AXI read slave**
`axi_mem_model`, instantiated beside the DUT in the simulation harness. The
testbench preloads its backing store through `init_*`; those setup writes do not
travel through the accelerator DMA. The harness defaults to a 64 KiB backing
store. `rd_latency` delays the first beat of each burst; subsequent beats can
transfer once per cycle while `RREADY` is high. This is a latency model, without
DDR/HBM command timing, bank conflicts, refresh or response reordering. It is
excluded from synthesis and PPA.

There are two distinct opportunities to overlap memory work:

1. **External transfer:** the top scheduler loads all Q, then K/V tile pairs via
   vector DMA. It starts the core after the first complete pair and continues
   filling later pairs while compute runs. `kv_tiles_ready` is the residency
   boundary; seeing K alone never makes a pair available.
2. **Local staging:** one shared loader reads the scratchpads into tile
   registers. Foreground loading fills Q or active K/V. Prefetch fills shadow
   K/V only when the next pair is resident and the loader is available. K and V
   use separate scratchpads and can each supply a 16-byte stripe in the same
   cycle. At the tile handoff, the controller promotes a complete next tile
   through a one-cycle parallel register copy into active K/V.

The array consumes active registers through the Q/K-versus-P/V operand mux.
K remains row-major in its register bank; the mux supplies the transposed QK
operand. Softmax sends **unnormalized exponential weights** back for PV, with
unsigned 8-bit P and signed INT8 V. Row rescale factors and running sums feed the
output buffer, which fuses rescale and accumulation and normalizes after the
last K/V tile. The final read port is available after `done`; this optimized top
has no AXI output-writeback engine. Causal masking follows dequantization and
the 1/√d shift.

Solid teal arrows show data movement; dashed gray arrows summarize setup and
control. AXI ready/valid signals, individual controller wires, address generation
and arithmetic pipeline stages are not drawn separately. The core controller
sequences foreground loads, prefetch, array phases, d64 chunks and output passes.
The external driver also observes completion, errors and performance counters.

| Diagram block | Implementation |
|---|---|
| Simulation boundary and external memory | [Simulation harness](../sim/verilator/tb_dma_banked_prefetch_harness.sv), [AXI memory model](../rtl/interface/axi_mem_model.sv) |
| Configuration, DMA scheduling and residency | [DMA-prefetch top](../rtl/top/flash_attn_top_dma_banked_prefetch.sv), [vector DMA](../rtl/interface/dma_engine_vec.sv) |
| Scratchpads and shared stripe loader | [Banked scratchpad](../rtl/memory/banked_scratchpad.sv), [tile loader](../rtl/memory/banked_tile_loader.sv) |
| Active/shadow registers, operand mux and shared scale | [Banked-prefetch core](../rtl/core/flash_attn_core_banked_prefetch.sv) |
| Tile/chunk sequencing and loader interlocks | [Prefetch controller](../rtl/ctrl/tile_controller_banked_prefetch.sv) |
| Shared array, dequantization and online softmax | [Array controller](../rtl/systolic/array_controller.sv), [dequantizer](../rtl/quantization/dequantizer.sv), [online softmax](../rtl/softmax/online_softmax.sv) |
| Fused update, normalization and external readout | [Output buffer](../rtl/memory/output_buffer.sv) |

## Arithmetic and precision

Each PE has a signed INT32 accumulator. QK uses signed INT8 operands; PV treats
P as unsigned 8-bit and V as signed INT8. For d64, QK retains partial sums across
four 16-element chunks, and PV processes four output-column chunks.

Scales are signed Q8.8. Dequantization preserves the full scale product, rounds,
saturates to INT16, then applies the supported 1/√d shift. Online softmax uses a
running maximum, LUT exponential and running sum. Fused output update preserves
the original 32-bit truncation/wrap semantics; final division truncates toward
zero. Exact fixed-point equality does not imply FP32 attention equality.

The numerical contract includes two deliberate precision limits. The PV operand
caps exp(0) from 256 to 255, while the softmax denominator retains 256: uniform
scores with V=1 and scale_v=256 therefore produce 255/256 (0.390625% low).
Dequantizer saturation occurs before the sqrt(d) shift, so large distinct logits
can become equal; this is not a full-range FP32 softmax. Dequantization rounds
ties toward positive infinity, PV/rescale shifts round down, and final signed
division truncates toward zero.

## Optimization evidence

- Vector writes and conflict-free banked stripe reads eliminated byte-serial
  movement between AXI and compute.
- Prefetch overlaps local tile staging with softmax/PV/output work. Its isolated
  E2E gain is modest because loading is a small part of runtime.
- Fusing rescale and accumulation removes one output-buffer traversal per
  tile/chunk: `16 tile pairs × {1,4} chunks × 257 cycles` matches the measured
  d16/d64 savings.
- Synthesis exposed 256 unused normalized-softmax compatibility dividers.
  Disabling that output path removed the failing divider path while preserving
  the running state and exp weights actually consumed by the core.
- Sharing the Q/K scale product reduced area further. The dequantizers remain
  the largest area contributor, 72.74% of the final measured integrated top.

## Supported configuration and limitations

- Optimized single-head prefill: `TILE_SIZE=16`, `HEAD_DIM` in `{16,64}`.
- Compile-time nonzero tile-aligned `SEQ_LEN`; runtime `cfg_seq_len` must equal it.
- `SRAM_DEPTH=4096` only, with `SEQ_LEN * HEAD_DIM <= 4096`. The optimized
  path uses fixed 12-bit internal interfaces; smaller depths are unsupported
  and rejected by the RTL guards and synthesis parameter validator.
- AXI 32-bit addresses, 64-bit data; Q/K/V bases 16-byte aligned, with each
  complete `SEQ_LEN * HEAD_DIM`-byte matrix fitting below the 2^32-byte limit.
- One shared clock and one outstanding AXI read burst.
- Behavioral external memory models fixed first-beat latency, not a DDR/HBM
  controller, bank scheduling, refresh, response reordering, or physical timing.
- Optimized output is an external read port. The older AXI writeback wrapper
  uses the non-prefetch banked core.
- Broader N/d support, multihead integration, physical memory macros and physical
  implementation are outside the currently validated optimized path.

## Figure sources and regeneration

`figures/architecture.svg` shows the simulation environment outside the
integrated DUT, then groups the accelerator into data movement and tiled
compute. These are functional groupings, not RTL module boundaries or a physical
floorplan. The array reads active registers; shadow registers must first be
copied into active. The array is one physical RTL instance reused for QK and PV.
Individual operand muxes, AXI channel signals and controller interlocks are
summarized rather than drawn separately. SRAM/ROM in the accepted synthesis flow
are logical blackboxes; the external memory model is simulation-only. The figure
is a logical dataflow, not a complete port or cycle-accurate timing diagram.

`figures/cycle-milestones.csv` records the historical cycle-count milestones from
the [results table](../README.md#measured-optimization-milestones). Each bar uses
its actual cycle count; the two dimension panels use different axis ranges.
The final values also match the current guarded RTL regression. These are
successive implementation milestones, not four configurations of the current RTL.

Regenerate both SVGs from the repository root with Python 3 and Matplotlib:

```bash
python3 docs/figures/render.py
```
