# Shared dequantizer experiment

[Home](../README.md) · [Design](DESIGN.md) · [Numerical accuracy](NUMERICAL_ACCURACY.md)

`DEQUANT_LANES` selects **256 (default), 32 or 16** on the optimized core and DMA
top. The 256-lane path retains the original parallel arithmetic and launch timing.
The 32/16-lane paths process one 256-score tile in 8/16 batches using unchanged
signed 32×32→64-bit multiplication, rounding and INT16 saturation.

The [batch adapter](../rtl/quantization/dequantizer_tile.sv) selects accumulator
groups, pipelines their indices with the registered dequantizer outputs, and
assembles all 256 scores. It publishes `tile_ready` only after the final batch
has been stored. QK accumulators and the accepted Q/K scale remain stable until
completion; PV does not launch until softmax consumes the complete score tile.
The controller FSM and active/shadow loader arbitration are unchanged.

The [recorded behavioral sweep](analysis/2026-10-03/verification.json) completed
**312 invocations / 624 transactions**, with exact outputs for all six lane/dimension
combinations. The original full regression, six full-core corner cases, Python
checks, and bounded controller formal also passed. Complete canonical latency
and causal counts are in [cycles.csv](analysis/2026-10-03/cycles.csv).

| Lanes | d16 core | d16 DMA top | d64 core | d64 DMA top |
|---|---:|---:|---:|---:|
| 256 | 7,589 | 7,800 | 28,337 | 29,160 |
| 32 | 7,717 | 7,928 | 28,465 | 29,288 |
| 16 | 7,845 | 8,056 | 28,593 | 29,416 |

These are noncausal N64 counts at zero modeled read latency. Top counts are
`start → done`; they exclude host initialization and output readout. The d16 top
adds **1.64% / 3.28%** latency for 32/16 lanes. Causal operation processes fewer
score tiles; its counts are reported separately in the CSV.

These are optional architecture experiments. The preserved N64/d16 PPA evidence
at `dfbd28e` describes the earlier 256-lane source; it does not measure the new
shared variants. A matched 256/32/16-lane integrated-top synthesis comparison is
required before claiming area or timing improvement.

Run a sweep from a checkout path **without spaces** (a Verilator build requirement):

```bash
make -C sim/verilator tb_dequantizer_tile tb_dequantizer_tile_init tb_dequantizer_contract DEQUANT_LANES=32
make -C sim/verilator tb_dequantizer_tile tb_dequantizer_tile_init tb_dequantizer_contract DEQUANT_LANES=16
python3 sim/verilator/run_dequant_sweep.py --lanes 32 --dim 16
python3 sim/verilator/run_dequant_sweep.py --lanes 16 --dim 64
```

Repeat the Python command for all six lane/dimension combinations. Each full
sweep checks 26 fixtures per core/top: two stored canonical noncausal/causal
fixtures plus all 24 generated accuracy cases for that dimension. Each top run
includes read latencies 0/20/100. `--smoke` limits the sweep to canonical fixtures;
CI runs all six smoke combinations. Unique ignored `obj_dequant_*` directories
prevent binaries from different configurations overwriting each other.

The unit test checks 200 tiles per shared lane count, including signed extremes,
rounding, saturation, repeated score tiles and reset during partial assembly.
The Icarus test starts with unknown score storage and checks complete publication,
ready retention and reset during partial assembly. The shared top contract test
checks accepted configuration locking, hostile busy-time inputs, rejected
configurations and DMA error/reset recovery.
Repeated unit-level score tiles do not change the top's one-transaction-per-reset
contract. The full sweep preserves canonical data and uses exact comparisons.

For matched synthesis, use the unchanged library, compile flow, 10 ns clock and
1 ns I/O delays, with logical memory blackboxes:

```bash
ELAB_PARAMETERS=DEQUANT_LANES=32 syn/scripts/run_server.sh top dequant_l32 compile
ELAB_PARAMETERS=DEQUANT_LANES=16 syn/scripts/run_server.sh top dequant_l16 compile
ELAB_PARAMETERS=DEQUANT_LANES=256 syn/scripts/run_server.sh top dequant_l256 compile
```

Set `DC_TARGET_LIBRARY` first as described in the [synthesis guide](../syn/README.md).
Run full-size profiles sequentially on memory-constrained machines. Compare
mapped area, setup slack and measured cycle counts; muxes and result storage
mean area cannot be inferred from lane count alone. This remains pre-layout
logical synthesis, without physical-memory area/timing or electrical signoff.
