# FlashAttention final results evidence — September 30 run series

Navigation: [documentation index](README.md). The evidence directory below is the
current baseline; September 24 reports are retained as historical evidence.

The final N64/d16 core and DMA-integrated top were mapped from the same clean
commit **`dfbd28e6262f9c40b5ff840557a4c9d1957b7606`**, including the source/destination address-range guards.
The commit ID was translated after repository history cleanup; the recorded
source hashes identify the same tested bytes. These are pre-layout logical-synthesis results with uncharacterized SRAM/ROM
blackboxes, not physical signoff.

| Final mapped result | Standalone core | DMA-integrated top |
|---|---:|---:|
| Standard-cell area (library units) | 5,061,521.160 | 5,074,696.289 |
| Critical path length | 6.01 ns | 5.98 ns |
| Worst setup slack at 100 MHz | +3.90 ns | +3.94 ns |
| Setup TNS / violating paths | 0.00 ns / 0 | 0.00 ns / 0 |

Matched integration area difference: **13,175.128162 library units,
0.26030%** relative to the standalone core. It includes mapping-context
effects; it is not a direct measurement of DMA area alone.

Public report extracts: [core](evidence/2026-09-30/core.txt),
[top](evidence/2026-09-30/top.txt). Machine-readable source/report/netlist/SDC
identity and verification hashes: [provenance](evidence/2026-09-30/provenance.json).

## Source identity and constraints

Core uses `final_20260930_n64d16`; top uses `final_20260930_n64d16_retry`.
Both use `compile`, `TILE_SIZE=16`, `HEAD_DIM=16`,
`SEQ_LEN=64`, `SRAM_DEPTH=4096`; the top uses AXI widths 32/64. Both manifests
record the same commit and `git_dirty=no`; their common source hashes match.
The initial parallel top process was OOM-killed (exit 137). Its failed output
is excluded; the accepted retry ran after core completion with identical settings.
The directory date names the September 30 run series; retained raw reports
keep their original tool timestamps.
The source-union SHA-256 is:

```text
8aba9424da8ff6df8f147e47031f0d1a5ef1f7ee451ede06035e5d496855d027
```

Library: Design Compiler R-2020.09-SP4, typical `gscl45nm.db` from FreePDK45.
The library SHA-256 is `4968d1dba7ff9911cc51dfac7d8ea8b94fbab59d2f862f0531d7c774fb0a5791`. The generated SDC confirms one
10 ns clock, 1 ns input/output delays, and the reset-only false-path exception.
There is no specified driving cell, output load, clock uncertainty or physical
clock tree. Mapped hierarchy confirms the intended N64/d16 geometry; each core
contains 256 dequantizers and 16 softmax lanes with unused normalized output
disabled. Arithmetic and checked-in fixture files did not change during closure.

## Area and timing interpretation

| Hierarchy area (library units) | Core | DMA top |
|---|---:|---:|
| 256 dequantizers | 3,691,327.0698 | 3,691,327.0698 |
| 16 softmax lanes | 424,565.8596 | 424,565.8596 |
| Array controller + array | 590,842.1344 | 590,838.3800 |
| Output buffer | 20,298.6333 | 20,298.6333 |
| DMA instance | — | 4,709.4255 |

These are nested hierarchy rows and must not be added as disjoint totals.
Dequantizers account for 72.74% of final top cell area. Logical memory
blackboxes contribute no characterized storage area or access delay.

The top worst reported setup path is `u_core/combined_scale_reg_reg[31]` → `u_core/gen_dequant[251].u_deq/data_out_reg[14]`.
Setup slack agrees between QoR and timing reports. The mapped min-delay report
shows +0.12 ns for core and
+0.12 ns for top under these same logical
constraints; this is not physical hold closure. Mapped `check_timing` reports
no warning, but missing memory timing remains a limitation of the blackbox model.
Do not calculate a signoff Fmax by taking 1/critical-path length.

The mapped lint review retains 320 undriven multiplier `PRODUCT[1]` output
warnings per profile. Every corresponding pin has zero flattened fanout
endpoints; these are unused low product bits. Although elaboration emits
`ELAB-978` for the softmax sequential block, `all_registers -level_sensitive`
finds **zero mapped latches** in both designs. This structural check is not
mapped-netlist equivalence. Other mapped lint counts are retained in provenance.
DC reports three high-fanout nets and uses a 1000-load cap for their delay
estimation; the reports do not model a physical clock/reset distribution network.

Core/top retain 105,926/107,039
max-capacitance violations. Every required limit is 0.000000000 in the
high-precision recheck, consistent with zero `max_capacitance` attributes on
the sampled library output pins. The raw violations remain in the evidence;
the run does not claim max-cap cleanliness or silicon electrical signoff.

## Power-report scope

| DC vectorless estimate | Core | DMA top |
|---|---:|---:|
| Total dynamic power | 94.1163 mW | 85.8273 mW |
| Cell leakage power | 24.9417 mW | 25.0368 mW |

These are the report's headline components, using low-effort activity propagation.
DC reports unannotated primary inputs, sequential outputs and blackbox outputs.
No workload SAIF/VCD activity, physical memory power or extracted parasitics were
used. Core/top have different activity-propagation contexts and do not form
a controlled power comparison. These figures are tool estimates only: no measured-power, energy-efficiency
or power-reduction claim is supported. Full reports stay in the retained bundle.

## Change from the prior mapped baseline

The September 24 baseline is preserved in
[its provenance](evidence/2026-09-24/provenance.json). The final guarded top area
changes from 5,073,910.211095 to 5,074,696.288599
(+786.077504 library units); setup slack changes
from +3.91 to +3.94 ns.
The new mapping closes the previously documented gap between guarded RTL and
whole-top PPA. Older source manifests and results have not been overwritten.

## Verification of the final revision

Verification ran from a fresh checkout of `dfbd28e`. Source hashes identify
the RTL, tests, data, golden models and synthesis inputs; tracked files stayed
unchanged through the run. See [verification summary](evidence/2026-09-30/verification.txt).

| Check | Result and boundary |
|---|---|
| Full Verilator regression, scenario checks, audit checks, lint | PASS from clean checkout |
| Optimized N64 d16/d64 core/top | Exact noncausal/causal comparisons; top latency 0/20/100 |
| Numerical invariants without hardware-golden import | 40 fixtures, 160 core/top transactions PASS |
| Oracle negative controls | 147 rejected scenarios PASS, including malformed data and missing completion |
| Python reference/data | Five methods PASS, 11 single-head fixtures and LUT/address checks |
| Extreme numerical cases | Six full-core signed-scale/saturation cases PASS |
| Four-state verification | Icarus/VCS output-buffer checks; eight VCS full-top transactions including changed-V reset/restart PASS |
| DMA/configuration | Stalls, boundaries, RRESP/RLAST faults, address-span rejection, configuration locking and common-reset recovery PASS |
| Controller formal | Yosys/SMT/Z3 BMC depth 48 and reachability covers PASS |
| Synthesis parameter validator | Five test methods PASS |
| Mapped artifacts | Source/library hashes, N64 hierarchy, netlist/SDC, timing/QoR/area consistency and constraints checked |

Formal uses abstract datapath/loader completions, not an unbounded datapath
proof. Scenario coverage is not a measured code/functional coverage percentage.
No gate-level simulation or mapped-netlist functional-equivalence proof is claimed.
The optimized interface remains one transaction per common reset, and errors
invalidate partial output. The old writeback/multihead/decode paths retain their
own regression criteria; they are not the measured optimized top.

Fixed-point equality is distinct from FP32 or model accuracy. P clips exp(0)
from 256 to 255 while the denominator retains 256, so equal scores and V=1 at
scale_v=256 give 255/256 (0.390625% low). Saturation before the sqrt(d) shift can
collapse large distinct logits. These behaviors are covered and documented.

## Optimization claims

| Claim | Supported interpretation |
|---|---|
| DMA fill 312→56, 5.57× | 256 B, vector versus scalar DMA, modeled read latency 10 |
| Scratchpad drain 256→17, 15.06× | 256 B, banked stripe versus byte-serial drain |
| Memory-path E2E 1.26–1.30× | Historical d16/d64 memory-stage pair |
| Fused update −34.5–36.1% E2E cycles | Historical unfused-to-fused prefetch pair; final latency-zero cycles remain 7800/29160 |
| Shared Q/K scale −7.35% core area | Matched historical N64/d16 pair; not this release's integration delta |
| 256 unused softmax dividers removed | Structural result; do not attribute the shared-scale area delta to this change |
| 100 MHz, +3.94 ns top setup slack | Final guarded N64/d16 revision, pre-layout logical synthesis with memory blackboxes |

The shared-scale pair `558f6a2` → `f4bccfa` measured
5,463,487.889260 → 5,061,648.809908, a 7.354991674% reduction, rounded directly
to **7.35%**. Both endpoints predate later correctness fixes. Their retained
manifests, geometry and reports support that historical comparison; they do not
replace current correctness evidence. See
[historical shared-scale excerpts](evidence/2026-09-24/historical_shared_scale.txt).
Earlier 7.36% used intermediate rounding; the September 4 ~0.24% integration
and divider-stage 29.25% area comparisons mixed N16/N64 and remain withdrawn.

## Reproduction and retained artifacts

Use `syn/scripts/run_server.sh core <fresh-tag> compile` and the corresponding
`top` command, with `HEAD_DIM=16,SEQ_LEN=64`, 10 ns clock, 1 ns I/O delays,
the recorded library and exact source hashes. Never overwrite an older run tag.
[Synthesis instructions](../syn/README.md) describe the environment and constraints.
README verification commands cover the public checks. Supplemental four-state
results have pass markers and log hashes in the public provenance record.

Public evidence contains portable report excerpts and hashes. Full logs, reports,
DDC, mapped Verilog, SDC and the frozen source checkout are retained outside the
published repository.
