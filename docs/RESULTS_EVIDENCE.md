# FlashAttention results evidence

Navigation: [documentation index](README.md).

## October 4 additional verification

The unchanged design also passed a VCS/URG four-state integrated sweep: **102
invocations / 986 exact transactions** over all six lane/dimension profiles,
with full output/accounting checks, DMA/configuration contracts and common-reset
recovery. This includes **400 stage-reset recoveries**: prime old output, interrupt
the named stage, then replace Q/K/V/scales and compare complete nonzero golden
outputs without another reset. All completed jobs match the earlier cycle count
for their geometry and read latency. DUT line coverage is
96.97–97.64%; branch coverage is 96.17–97.79%. The defined functional bins all
hit, and FSM transition coverage improves from 64.09–69.55% to **98.64–99.55%**
with no waivers. All reported reset-to-idle transitions are covered; remaining
transitions are the d16-unused chunk paths and the integrated DMA descriptor guard.
[Scope, gaps and reproduction](COVERAGE_POWER.md) explain those distinctions.
This adds evidence, without changing the original measured RTL or netlists.

The companion PrimeTime/PrimePower study completes 12 activity analyses on the
three preserved N64/d16 netlists. Canonical noncausal standard-cell estimates
are **127.234 / 105.331 / 103.033 mW** and **9.926 / 8.352 / 8.301 µJ** per job
for 256/32/16 lanes. The 16-lane energy estimate is 16.36% below the baseline.
Primary inputs are directly annotated; sequential file-plus-implied coverage is
95.20–95.24%, with propagated/default activity disclosed separately. These are
pre-layout standard-cell estimates, excluding physical memory and parasitics.
[Four workloads, provenance and limits](COVERAGE_POWER.md#workload-based-standard-cell-power-estimates)
retain the old vectorless results as a different experiment.

## Latest matched shared-dequantizer sweep

The October 3 series maps three **N64/d16 DMA-integrated tops** from clean
`d5f9123fbbf73154073b28d683ea4bb88b9e3571`, with identical source/library hashes,
DC R-2020.09-SP4, ordinary `compile`, a 10 ns clock and 1 ns I/O delays.
After history consolidation, the identical measured snapshot is available at
`dd68964`; see the [revision map](README.md#october-3-revision-identities).
Original manifest identities and measurement hashes are preserved.

| Lanes | Cell area (library units) | Area reduction vs 256 | Critical path | Setup slack | Noncausal top cycles |
|---|---:|---:|---:|---:|---:|
| 256 | 5,074,665.315 | — | 6.00 ns | +3.92 ns | 7,800 |
| 32 | 1,940,361.688 | 61.76% | 9.26 ns | +0.66 ns | 7,928 |
| 16 | 1,715,678.209 | 66.19% | 7.80 ns | +2.12 ns | 8,056 |

All three have zero setup TNS/violating paths. Cycle counts are N64/d16,
noncausal, zero modeled read latency, `start → done`, excluding host setup and
output readout. The exact reductions are 61.76375055% / 66.19130321%; d16 cycles
increase by 1.64102564% / 3.28205128%. The 256-lane default is preserved.

[Experiment details](DEQUANT_EXPERIMENT.md) and [new provenance](evidence/2026-10-03/provenance.json)
record actual mapped lane counts, source identity, report/netlist/SDC hashes and
mapped postchecks. Shared-path behavioral verification completes 312 invocations /
624 exact transactions across all lane/dimension combinations; see
[verification](analysis/2026-10-03/verification.json) and
[full latency/causal counts](analysis/2026-10-03/cycles.csv).
The original full regression, six core corners, shared units and contract tests,
Python checks and bounded controller formal also passed. CI remains a subset of
that recorded verification.

This newer PPA is **top-only and d16-only**. Do not subtract the older standalone
core area from these top areas. Memories are logical blackboxes; there is no
physical/electrical signoff, mapped equivalence or new d64 PPA claim.

### New mapped warning and power scope

All three saved DDCs were reloaded and linked to the same library. There are
**zero mapped latches** despite the softmax elaboration warning. The audited
unused `PRODUCT[1]` pins total 320 / 96 / 80 for 256 / 32 / 16 lanes; each has
zero flattened fanout endpoints, matching the actual undriven-output/net lint
counts. Other mapped lint categories remain recorded in provenance.
`check_timing` emits no warning; min-delay slack is +0.12 / +0.09 / +0.09 ns under
the logical constraints, not physical hold closure.

Max-cap violations are **107,039 / 103,877 / 100,206**. Every required cap limit
is zero in the high-precision recheck, and sampled library output pins also have
`max_capacitance=0`. The violations are retained; electrical closure is not claimed.

| Vectorless DC estimate | 256 lanes | 32 lanes | 16 lanes |
|---|---:|---:|---:|
| Total dynamic power | 85.8273 mW | 82.6901 mW | 82.0574 mW |
| Cell leakage power | 25.0366 mW | 10.6205 mW | 9.5850 mW |

These use unannotated activity and omit physical memory power and parasitics.
They are tool estimates, not workload power or an energy-efficiency result.
The measured area reduction must not be presented as the same percentage
reduction in system power or die area.

## Preserved September 30 standalone-core/top pair

The following sections retain that earlier matched pair, including its warning
review and verification boundaries. September 24 reports remain historical.

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
