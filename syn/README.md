# Design Compiler synthesis flow

The preserved matched pair is **N64/d16 standalone core and DMA-integrated top**
at `dfbd28e`. The newer shared-dequantizer sweep measures **three integrated tops**
at `d5f9123`, using the same source, library, clock and flow within that sweep. The current measured
results and provenance are in [results evidence](../docs/RESULTS_EVIDENCE.md).
The equivalent measured source after history consolidation is `dd68964`;
see the [revision map](../docs/README.md#october-3-revision-identities). Original
run manifests and source hashes are preserved.

## Profiles and explicit geometry

| Profile | Top / purpose | Memory model |
|---|---|---|
| `core` | `flash_attn_core_banked_prefetch` | Logical SRAM/ROM blackboxes |
| `top` | `flash_attn_top_dma_banked_prefetch` | Logical SRAM/ROM blackboxes |
| `core_rtl`, `top_rtl` | Same cores, elaboration/debug | Behavioral memories |
| `systolic` | `systolic_array` | None |
| `dma` | `dma_engine_vec` | None |
| `loader` | `banked_tile_loader` | None |

Core/top profiles explicitly default to `TILE_SIZE=16,HEAD_DIM=16,SEQ_LEN=64,
SRAM_DEPTH=4096,DEQUANT_LANES=256`; top also selects AXI widths 32/64. `ELAB_PARAMETERS` provides
partial overrides, expanded by `run_metadata.py`. Unsupported geometry is rejected
before invoking DC. The standalone RTL module's default N16 is **not** the flow's
canonical setting.

`DEQUANT_LANES=32` or `16` selects an optional batch-dequantizer architecture;
`256` preserves the parallel default. Compare variants using a fresh matched
256-lane run from the same source. Functional scope and cycle counts are in the
[experiment guide](../docs/DEQUANT_EXPERIMENT.md).

The optimized core/top profiles require `SRAM_DEPTH=4096`, matching their fixed
12-bit internal interfaces and RTL guards. Smaller depths are not supported;
overrides are rejected before DC, even when `SEQ_LEN*HEAD_DIM` would fit.

Macro filelists substitute `syn/blackboxes/sram_1r1w_bb.sv` and `exp_lut_bb.sv` for
behavioral memories. These stubs have no characterized area, access timing or
power. Avoid behavioral full-memory mapping as a substitute for real memory macros.

## Run a matched pair

Provide a licensed DC executable and the intended standard-cell library through
the environment; no machine-specific paths belong in the repository:

```bash
export DC_SHELL_BIN=dc_shell
export DC_TARGET_LIBRARY=/path/to/gscl45nm.db

CLK_PERIOD=10.0 IO_DELAY=1.0 ELAB_PARAMETERS='HEAD_DIM=16,SEQ_LEN=64' \
  syn/scripts/run_server.sh core unique_n64d16 compile
CLK_PERIOD=10.0 IO_DELAY=1.0 ELAB_PARAMETERS='HEAD_DIM=16,SEQ_LEN=64' \
  syn/scripts/run_server.sh top unique_n64d16 compile
```

Use a new run tag for each experiment. Existing run directories are never
reused or overwritten. `elab` stops after analyze/elaborate/link and writes DDC;
`compile` performs ordinary mapping; `compile_ultra` is a separate flow choice
and must not silently replace one endpoint of a comparison.

`PYTHON_BIN` can select an existing Python 3 runtime. For a supported d64 run,
use `ELAB_PARAMETERS='HEAD_DIM=64,SEQ_LEN=64'` and a separate run tag.

## Constraints

The shared Tcl creates one `clk`, normally 10 ns, and sets 1 ns input/output
delays. `rst_n` is ideal and false-pathed from the input. No input driving cell,
output load, clock uncertainty, physical placement, CTS or extracted parasitics
are specified. Preserve these assumptions across any controlled comparison.

Positive setup margin is `qor.rpt`'s **Critical Path Slack**, checked against
`timing_setup.rpt`. The design-level WNS summary clamps positive margin to zero.
Area means **Total cell area** in library units, excluding physical memory area;
it does not mean die area or mm². Vectorless `report_power` is not measured or
activity-annotated system power.

## Identity and acceptance checks

Each profile creates `syn/runs/<tag>/<profile>/` with:

```text
manifest.txt
source_sha256.json
SUCCESS or FAILED
logs/dc_shell.log
reports/{check_design,hierarchy,reference,area,qor,timing_setup,
         constraint_violators,power,resources,clocks}.rpt
artifacts/{mapped.v,mapped.ddc,constraints.sdc}
work/
```

The manifest records the base Git commit, dirty status, complete requested
parameters, effective elaborated design, library path, mode, clock/I/O constraints
and timestamps. `source_sha256.json` records filelists, every listed source and
flow scripts. A dirty snapshot must be identified by these hashes, not its base
commit alone. Use an isolated snapshot so unrelated checkout changes cannot
silently enter a run; committing or pushing is not required to launch a run.

The Tcl preserves the design selected by `elaborate` and checks `link`. The wrapper
rejects source drift, textual tool `Error:` messages, missing completion markers
and missing mapped artifacts even if DC exits zero. `SUCCESS` establishes flow
completion, **not timing closure or functional correctness**.

Before accepting PPA, check generated hierarchy and mapped declarations for both
`HEAD_DIM16_SEQ_LEN64` core and controller specializations. Also inspect actual
SDC, area, timing, TNS, design-rule warnings and critical-path endpoints. A run
label or requested parameter string alone is insufficient evidence.

For integration area, use only the canonical matched pair and explicitly define
`(top − standalone core) / standalone core`. Record in-context `u_core` separately;
the difference is not necessarily pure DMA area because mapping context matters.

After compilation, recheck each accepted mapped run sequentially:

```bash
MAPPED_RUN_DIR=syn/runs/experiment_l32_n64d16/top \
  dc_shell -f syn/scripts/check_mapped.tcl \
  > syn/runs/experiment_l32_n64d16/top/logs/postcheck.log 2>&1
```

Keep `DC_TARGET_LIBRARY` set to the same library. Repeat for 16 and 256 lanes.
The recheck reads the saved DDC and preserves the mapped netlist; it records
high-precision constraints, min-delay timing, mapped lint, latch count and
fanout of unused low product bits. Require the completion marker and no tool
errors. These structural checks do not establish mapped functional equivalence.

For a completed three-way top comparison, `report_dequant_sweep.py` accepts
run tags `<prefix>_l256_n64d16`, `<prefix>_l32_n64d16`, and `<prefix>_l16_n64d16`:

```bash
python3 syn/scripts/report_dequant_sweep.py --runs-root syn/runs \
  --run-prefix experiment --output-dir /tmp/flashattn-dequant-evidence
```

It requires clean successful runs, identical source/library hashes and settings,
explicit geometry, actual mapped dequantizer counts and completed mapped rechecks. It emits portable
metric excerpts and provenance with library basenames, retaining raw report and
artifact hashes. This collector does not replace warning review or functional
verification.

## Historical corrections and public artifacts

The September 4 standalone core used N16, while its top used N64. Their 0.23743%
difference is invalid as a controlled integration overhead. The initial divider
baseline likewise used N16 versus the later N64 run. Raw reports remain unchanged;
these pairs must not be reused to claim isolated total-area/power reductions.

The separate historical shared-scale comparison used two clean N64/d16 commits;
its 7.3549917% reduction (7.35% directly rounded; formerly reported as 7.36%)
retains its own before/after provenance. It is not recomputed
using the final repaired RTL. See the public results document for claim disposition.

Raw runs, work libraries, DDC, netlists, notes and verification logs stay ignored.
Public `docs/RESULTS_EVIDENCE.md` and compact `docs/evidence/` extracts preserve
results, source/report hashes and limitations without host paths, license details
or large generated artifacts. Supply library/tool access separately when reproducing.

A passing logical synthesis run does not establish memory timing, physical setup/
hold closure, signoff power, or mapped-netlist functional equivalence.

## Workload activity on preserved mapped tops

The [activity power runner](scripts/run_activity_power.py) checks preserved
mapped netlist/SDC/library/RTL hashes against matched synthesis provenance,
captures VCS RTL VCD activity for an exact-checked job, then uses
[PrimeTime/PrimePower](scripts/pt_activity_power.tcl) in averaged mode. It does
not synthesize another implementation. Results are N64/d16 only, at the same
10 ns clock, with zero modeled AXI read latency.

First complete the [VCS coverage flow](../sim/vcs/README.md). For the published
October 3 mapped artifacts, set the tool/library environment through your EDA
installation and run from the repository root:

```sh
# PT_SHELL_BIN selects a working licensed pt_shell. DC_TARGET_LIBRARY must
# point to the same .db used for synthesis. Some installations also require
# SYNOPSYS_LC_ROOT and its bin directory on PATH.
python3 syn/scripts/run_activity_power.py \
  --run-dir "$PWD/syn/runs/activity_new" \
  --vcs-run "$PWD/sim/vcs/runs/coverage_new" \
  --mapped-root "$PWD/syn/runs" \
  --provenance docs/evidence/2026-10-03/provenance.json \
  --fixtures /tmp/attention-extra-fixtures
```

This example requires the preserved `dequant_20261003_l{256,32,16}_n64d16/top`
artifacts; public git does not contain licensed netlists/databases. For your own
fresh matched synthesis runs, first collect their provenance with
`report_dequant_sweep.py`, then pass that JSON to `--provenance` and the matching
`--mapped-tag 'experiment_l{lanes}_n64d16'`. Rerunning synthesis need not produce
byte-identical artifacts, so it must have its own provenance.

The runner rebuilds d16 VCS executables with array dumping enabled, rather than
modifying the accepted coverage databases. Four workloads per lane configuration
are checked: canonical noncausal/causal, random seed 7, and distinct saturated
logits. VCD capture covers accepted start through completion; host initialization
and output readout are excluded. The activity duration includes the accepted
start cycle (`perf_total_cycles = tb_cycles + 1`).

Only VCD header names/ranges are normalized for DC's generate-scope renaming and
packed array buses; timestamp/value records are preserved. Both original and
normalized VCD hashes are retained. `read_vcd -rtl` maps activity to the preserved
netlist, and the tool propagates activity into unmapped combinational logic.
The runner requires no PrimeTime errors, an explicit completion marker, consistent
power units/sums and at least 95% file-or-implied activity on primary-input and
sequential-driver nets. File, implied, propagated, default and unannotated counts
remain separate in the result. Implied activity is derived without random-vector
propagation from annotated points; it is not counted as direct VCD annotation.

`power.json` records per-workload input/report/activity hashes and annotation
counts; `power.csv` records W and nJ/transaction. The conversion is
`energy_nJ = total_power_W × activity_duration_ns`. Inspect residual activity
reports and warning logs as well as PASS markers. These are pre-layout
standard-cell estimates, with logical SRAM/ROM blackboxes, not physical memory
or whole-system energy. See [coverage and power evidence](../docs/COVERAGE_POWER.md).

Collect a complete accepted 12-case summary (or disjoint per-lane summaries):

```sh
python3 syn/scripts/summarize_activity_power.py syn/runs/activity_new/power.json \
  --output-dir /tmp/attention-power-export \
  --flow-commit YOUR_CAPTURE_REVISION \
  --pt-version YOUR_VERIFIED_PT_VERSION --vcs-version YOUR_VERIFIED_VCS_VERSION
```

The collector requires the full three-lane/four-workload grid, identical source,
flow/library/fixture hashes across the relevant comparisons, consistent timing
windows and energy arithmetic, and accepted annotations. It exports matched
energy reductions and annotation fractions without assuming area equals energy.

For the library's asynchronous-control clock warnings, use the same mapped
inputs with a **new** `PT_RUN_DIR` and run
`pt_shell -f syn/scripts/check_pt_clock_scope.tcl`. It checks all actual register
CLK pins have exactly the `clk` domain, and preserves the verbose related-pin
warning list. The recorded warning entries are DFFSR S/R controls; adding a fake
clock to those controls would not establish physical reset timing closure.
