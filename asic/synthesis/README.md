# Synthesis — flash_attn_core (OpenLane / sky130A)

Post-synthesis + gate-level STA results for the single-head core
`flash_attn_core` (includes GQA, causal masking, KV-decode, and d=64
inner-dimension tiling).

## Flow

| | |
|---|---|
| Tool | OpenLane **v1.1.1** (Yosys 0.38 + ABC + OpenSTA) |
| PDK | **sky130A**, `sky130_fd_sc_hd` standard cells |
| Clock target | 20 ns (**50 MHz**) |
| Constraints | [`flash_attn_core.sdc`](flash_attn_core.sdc) — `rst_n` declared as a false path |

## Results (run_006)

| Metric | Value |
|--------|-------|
| Standard cells | ~343,900 |
| Logic core area | ~3.65 mm² |
| Critical path | 12.73 ns (softmax index path) |
| Setup slack (worst) | **+7.16 ns — MET** |
| Hold slack (worst) | **+0.35 ns — MET** |
| TNS / WNS | 0.00 / 0.00 |
| f_max (estimated) | ~78 MHz |

Reports are committed under [`reports/`](reports/):
- [`synthesis_area.rpt`](reports/synthesis_area.rpt) — cell count + area
- [`sta_summary.rpt`](reports/sta_summary.rpt) — TNS / WNS / worst slack
- [`sta_worst_paths.rpt`](reports/sta_worst_paths.rpt) — worst setup + hold path

## Memory / IP modeled as macros (blackboxed)

`sram_1r1w`, `dequantizer` (×16, time-multiplexed), and `output_buffer` are
synthesized as **blackbox memory macros** — their area is **not** included in
the 3.65 mm² above. In a real flow these are hard macros from a memory
compiler (e.g. OpenRAM). Port-only stubs used for synthesis:
[`sram_1r1w_bb.v`](sram_1r1w_bb.v), [`dequantizer_bb.v`](dequantizer_bb.v),
[`output_buffer_bb.v`](output_buffer_bb.v).

> Modeling the SRAMs as macros (rather than letting them synthesize into
> flip-flop register files) is what makes this a *logic-core* area number.
> A behavioral-SRAM synthesis of the same RTL inflates to ~1.05 M cells /
> ~11.5 mm² because the read-port mux trees and storage flops dominate.

## Scope / limitations

- These are **post-synthesis STA numbers with an ideal (zero-skew) clock —
  pre-P&R.** Floorplan / P&R were not run because the blackboxed macros
  lack LEF. Post-route timing would be worse (clock tree + wire delay);
  `~78 MHz` is an estimate from the critical path, not a routed/signed-off
  frequency.
- The full OpenLane run directory (~1.3 GB) is **not** committed; only the
  curated reports above are kept.
