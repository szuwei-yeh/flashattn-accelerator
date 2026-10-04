# Four-state coverage and workload activity

The October 4 experiment adds verification and power-analysis infrastructure
without changing RTL. The mapped inputs remain the October 3 N64/d16 matched
netlists from `d5f9123` (equivalent source snapshot `dd68964`). Original evidence
and the earlier Verilator sweep remain intact.

## Main-path VCS/URG coverage

VCS/URG **V-2023.12-SP2** completed **102 invocations / 986 exact transactions**
across 256/32/16 lanes and d16/d64. All outputs, DMA byte/tile accounting and
cycle-counter checks passed, including **400 stage-reset/recovery cases** and
2,524,160 checked output words. Every completed job retains the October 3 cycle
count for its lane/dimension/causal/read-latency geometry. These are additional
checks of the same design, not an expansion to a multiple-jobs-per-reset interface.

| Lanes | Dimension | Line | Condition | Toggle | FSM transitions | Branch | Defined functional groups |
|---|---|---:|---:|---:|---:|---:|---:|
| 256 | d16 | 96.97% | 88.04% | 88.64% | 98.64% | 96.17% | 100% |
| 256 | d64 | 97.11% | 91.77% | 89.88% | 99.55% | 96.24% | 100% |
| 32 | d16 | 97.36% | 90.26% | 82.40% | 98.64% | 97.58% | 100% |
| 32 | d64 | 97.58% | 90.30% | 83.06% | 99.55% | 97.66% | 100% |
| 16 | d16 | 97.41% | 90.26% | 81.41% | 98.64% | 97.71% | 100% |
| 16 | d64 | 97.64% | 90.30% | 81.97% | 99.55% | 97.79% | 100% |

The [initial 90-invocation / 186-transaction sweep](analysis/2026-10-04/coverage.json)
is preserved separately. Its FSM transitions were 69.55% for d16 and 64.09% for
d64. The follow-up reruns those scenarios and adds 12 reset-sweep invocations;
line/condition/toggle/branch coverage is unchanged. Reset already executed at
startup, so covering reset from additional states improves transition coverage
without necessarily adding executed lines. No cross-elaboration VDB merge is used.

Each profile runs canonical causal/noncausal fixtures at read latencies
0/20/100 and four additional fixture families in both causal modes: random
seed 7, uniform bias, distinct saturated logits and signed scales. All regular
runs complete one original-data job and one changed-V-zero job after a common
reset. Contract runs add invalid lengths/alignment/range, hostile input changes
and starts while busy, ignored post-done starts, RRESP/RLAST faults, a failed
later resident tile, reset recovery and reset during a partial output update.

Fresh SRAM is checked to contain unknown values. Missing/unknown fixture words
are rejected. Every successful job reads and compares all 1,024 or 4,096 output
words with four-state comparisons. Two intentional negative controls—corrupting
the first expected word and truncating the expected file—were rejected by real
VCS execution. Python failure-handling checks also reject zero-exit failures and
missing completion markers.

### Stage reset/recovery follow-up

The optional `--reset-sweep` adds 32 named checkpoints for the parallel design
and 34 for shared dequantization, in both causal modes at read latency 100:

| Block | Targeted stages |
|---|---|
| Tile controller | All 11 non-idle states, including Q/K/V loading, QKᵀ, softmax, PV, fused output update, normalization and prefetch wait |
| DMA scheduler | All seven non-idle states, including Q/K/V issue and wait |
| Read DMA | Address request, receive with a retained half stripe, and sticky error |
| Array controller | Clear, compute after at least eight cycles, and finish |
| Tile loader | Issue after at least four stripes, and drain |
| All 16 softmax lanes | Find max, rescale, rescale wait, accumulate and norm |
| Dequantization | Published-score validity; shared variants additionally interrupt mid-issue and the final pending retirement |

Each checkpoint primes a fully checked canonical output, starts another
canonical job and interrupts it at the named stage. Reset is asserted between
clock edges. The bench checks idle control, cleared counters/errors/validity,
and no DMA/output/loader writes or AXI activity during reset and for five cycles
after release. It then loads a different prepared Q/K/V fixture and accepted
scales and compares every output word with the new golden **without another
reset**. The new golden must differ from the old output and cannot be all zero.
Replacement fixture words are cleared to unknown before reading, so incomplete
files cannot borrow old data. Neither scratchpad nor output SRAM is cleared by
the bench. Regular cases separately retain changed-V-zero recovery.

Recovery bins are sampled only after full-output and accounting checks. The
runner rejects missing, duplicate, reordered or misidentified checkpoint
markers, incorrect completion counts and incomplete reset-bin coverage.
The reset remains common to DUT and behavioral AXI slave. This is selected-stage
simulation, not exhaustive interruption timing, independent-master AXI recovery
or physical asynchronous-reset recovery/removal signoff.

### Remaining coverage gaps

Coverage includes only the integrated DUT hierarchy; there are no waivers. The
100% functional number means all explicitly declared bins were hit: successful
latency/causal combinations, nine contract events, 12 controller states,
shadow promotion and 32/34 checked reset recoveries. It does not mean every legal input or fault interleaving was
exercised. Procedural error checks contribute code coverage; no new complete
SVA assertion-coverage claim is made.

URG's FSM score counts transitions; state coverage is reported separately and
is not included in that score. All enumerated controller, scheduler, DMA,
array-controller, loader and softmax states were reached, and all reported
reset-to-idle transitions are now covered. Every softmax instance reaches
10/10 transitions; the tile controller reaches 26/26 for d64 and 25/26 for d16.
The only remaining reported FSM transitions are:

| Transition | Profiles | Reason / separate evidence |
|---|---|---|
| Controller `S_FUSED_UPDATE → S_MATMUL_PV` | d16 only | Next output chunk; d16 has one chunk. Covered by d64. |
| Array `IDLE → COMPUTE` without clear | d16 only | Retaining partial sums across QK chunks; d16 has one chunk. Covered by d64. |
| DMA `S_IDLE → S_ERROR` | All six | Illegal descriptor guard; validated integrated scheduler emits legal descriptors. The separate DMA protocol unit test checks eight invalid descriptor cases. |

Other gaps include the output buffer's standalone rescale branch (the main path
uses fused rescale/accumulate), forbidden-mode error statements and signal bits
that stay constant under these dimensions/fixtures. These are retained in the
reported percentages. They do not establish unreachable-code proofs or replace
unit tests, protocol checks and the separately recorded bounded formal analysis.

[Commands and bench contracts](../sim/vcs/README.md) ·
[Latest metrics and report hashes](analysis/2026-10-04/reset_sweep/coverage.json) ·
[Transactions and source hashes](analysis/2026-10-04/reset_sweep/coverage_verification.json) ·
[All 400 reset recoveries](analysis/2026-10-04/reset_sweep/reset_recoveries.csv) ·
[Unwaived FSM detail](analysis/2026-10-04/reset_sweep/fsm_review.txt)

A temporary bench mutation omitting the targeted reset was rejected by real VCS
at the first checkpoint; [mutation, diagnostic and hashes](analysis/2026-10-04/reset_sweep/negative_control.json)
are retained. The original corrupt/truncated-output
[negative controls](analysis/2026-10-04/negative_controls.json) remain separate.

## Workload-based standard-cell power estimates

PrimeTime **R-2020.09-SP5-1** with PrimePower completed **12 accepted analyses**:
three preserved N64/d16 mapped tops × canonical noncausal/causal, random seed 7,
and distinct saturated logits. Each activity capture independently passes a
complete 1,024-word fixed-point output/accounting check in VCS. Library, netlist,
SDC and RTL hashes are checked against the original synthesis provenance;
there is no new synthesis or mapped-equivalence claim.

The comparison uses the typical `gscl45nm` library, **100 MHz**, zero modeled AXI
read latency and averaged RTL VCD activity mapped to standard cells. No wire-load
model is set. SRAM/ROM remain logical blackboxes. Capture encloses accepted start
through sampled completion, including the accepted-start cycle; initialization,
common reset and output readout are outside the interval.

### Canonical noncausal N64/d16

| Lanes | Dynamic estimate | Leakage estimate | Total estimate | Active interval | Energy estimate | Energy reduction vs 256 |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 102.197 mW | 25.037 mW | 127.234 mW | 78.010 µs | 9.926 µJ | — |
| 32 | 94.710 mW | 10.620 mW | 105.331 mW | 79.290 µs | 8.352 µJ | 15.86% |
| 16 | 93.448 mW | 9.585 mW | 103.033 mW | 80.570 µs | 8.301 µJ | 16.36% |

Energy is `total_power_W × duration_ns` in nJ. The intervals use
**7,801 / 7,929 / 8,057** cycles; the earlier README's **7,800 / 7,928 / 8,056**
TB counts omit the accepted-start edge. That one-cycle accounting distinction is
explicitly checked in both the bench and the collector.

### All four recorded workloads

| Workload | 256-lane energy | 32-lane energy | 16-lane energy | 32-lane reduction | 16-lane reduction |
|---|---:|---:|---:|---:|---:|
| Canonical noncausal | 9.926 µJ | 8.352 µJ | 8.301 µJ | 15.86% | 16.36% |
| Canonical causal | 6.762 µJ | 5.708 µJ | 5.667 µJ | 15.59% | 16.19% |
| Random seed 7, noncausal | 9.950 µJ | 8.368 µJ | 8.318 µJ | 15.90% | 16.40% |
| Distinct saturated logits, noncausal | 9.301 µJ | 7.946 µJ | 7.917 µJ | 14.57% | 14.88% |

These are the four named fixtures, not a representative model workload average.
Saturated cases still follow the fixed-point contract; their floating-point
accuracy limitations remain in the [numerical report](NUMERICAL_ACCURACY.md).
Power-capture jobs are counted separately from both coverage experiments.

### Annotation quality and interpretation

The VCD's generate-scope names and unpacked-vector ranges are normalized to
DC's mapped naming. Timestamp/value records are preserved and both waveform
hashes are retained. `read_vcd -rtl` supplies activity on matching signals;
PrimeTime propagates activity to the remaining internal logic.

| Driver/activity category | 256 lanes | 32 lanes | 16 lanes |
|---|---:|---:|---:|
| Primary-input nets directly from VCD | 100% | 100% | 100% |
| Sequential nets directly from VCD | 54.35% | 62.30% | 62.69% |
| Sequential nets with implied activity | 40.86% | 32.94% | 32.52% |
| Sequential nets with propagated activity | 4.80% | 4.76% | 4.79% |
| All nets with default activity | 2.16% | 1.99% | 1.95% |

Implied activity is derived from annotated points without random-vector
propagation; it is kept separate from direct VCD annotation. The sequential
file-plus-implied fraction is **95.20–95.24%**. The remaining **1,920** propagated
sequential nets in each design are the array controller's `a_sr`/`b_sr` operand
delay registers, whose pruned mapped bits have different names. Sequential-driver
nets have zero default/unannotated activity. All-net defaults remain visible,
including arithmetic hierarchy ports; they are not waived or called direct
workload activity. Detailed classification is stable across the four fixtures
for each elaboration and is recorded for all 12 analyses.

The area reduction is much larger than the estimated energy reduction. Clock
pin internal power remains substantial: the baseline report groups **84.201 mW**
as `clock_network`, including register clock-pin internal power. This is not a
physically implemented clock-tree measurement. Smaller dequantizer hardware
reduces leakage and some dynamic power, while extra cycles still clock the
shared compute and storage registers. The measured area percentage must not be
reused as a power or energy percentage.

### PrimeTime clock-warning review

The initial `check_timing` reports retain a warning for **29,781 / 25,984 /
25,714** purported unclocked register clock pins. A separate full verbose audit,
after `update_timing`, identifies every listed pin as a `DFFSR` asynchronous
**S or R control**, with no `/CLK` entries. This library has recovery/removal
relationships between the asynchronous controls; those related pins are also
considered by PrimeTime's clock checks. The warning is not removed by a timing
update and is not silently waived.

The [clock audit](../syn/scripts/check_pt_clock_scope.tcl) checks every actual
register clock pin: **40,033 / 40,332 / 40,062** pins for 256/32/16 lanes,
all with exactly the `clk` domain, and **zero actual clock pins outside it**.
This also matches the mapped `.CLK(clk)` connection counts and the sequential
net totals. The audit preserves [counts and report hashes](analysis/2026-10-04/clock_scope.json)
and [portable excerpts](analysis/2026-10-04/clock_scope.txt). It establishes the
clock scope used by these estimates; physical reset recovery/removal closure
is not claimed. The original DC postcheck and this PrimeTime warning review
remain separate tool results.

These are **pre-layout standard-cell estimates** with partial RTL annotation
and statistical propagation. Physical SRAM/ROM energy, external DRAM, host
setup/readout, interconnect parasitics, clock-tree implementation and mapped
waveform/glitch behavior are not established. No chip power, whole-system energy,
physical timing closure or power signoff is claimed. Earlier vectorless DC
numbers remain a separately labeled historical estimate.

[Power/energy CSV](analysis/2026-10-04/power.csv) ·
[Source, input, waveform and report hashes](analysis/2026-10-04/power.json) ·
[Per-case annotation fractions](analysis/2026-10-04/activity_annotation.csv) ·
[Portable power reports](analysis/2026-10-04/power_reports.txt) ·
[Annotation reports](analysis/2026-10-04/activity_reports.txt) ·
[Warning/residual review](analysis/2026-10-04/power_review.json) ·
[Reproduction](../syn/README.md#workload-activity-on-preserved-mapped-tops)

Raw waveforms, mapped netlists, VDBs and logs stay in private run bundles.
Public report excerpts normalize whitespace and replace the library path with
its basename; recorded raw-report SHA256 values describe the original reports.
The power captures use flow revision `512377e`; the preserved mapped RTL identity
remains `d5f9123` / equivalent `dd68964`.
