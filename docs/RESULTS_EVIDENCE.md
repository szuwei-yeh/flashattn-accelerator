# FlashAttention results evidence — 2026-09-24

The corrected **N64/d16 standalone core and DMA-integrated top** were mapped with
the same RTL snapshot, library and synthesis flow. These are pre-layout logical
synthesis results; SRAM/ROM area and access timing are excluded.

| Metric | Standalone core | DMA-integrated top |
|---|---:|---:|
| Total standard-cell area (library units) | 5,061,521.160437 | 5,073,910.211095 |
| Critical path length | 6.01 ns | 6.01 ns |
| Worst setup slack at 10 ns / 100 MHz | +3.90 ns | +3.91 ns |
| Setup TNS | 0.00 ns | 0.00 ns |
| Setup violating paths | 0 | 0 |
| Max-capacitance violations | 105,926 | 106,988 |

**Matched integration area difference:** 12,389.050658 units,
**0.24477%** relative to this standalone core.
Report extracts: [core](evidence/2026-09-24/core.txt),
[top](evidence/2026-09-24/top.txt). Machine-readable source/report fingerprints,
metrics and verification identity: [provenance.json](evidence/2026-09-24/provenance.json).

## Source and parameter identity

At synthesis time, the snapshot was base commit
`36ec6158da5490ba1baea04081b311ad86d085f5` **plus then-uncommitted audit fixes**.
The release commit containing this evidence preserves those measured source bytes;
the original run manifests retain their actual base commit and dirty status.
The base commit alone does not identify the measured RTL. No new RTL change was
made during PPA closure or publication. The source-union SHA-256 is:

```text
27e668eb053edfbdc7d5c914571a81847ea59253d58e24a617f392217aaade07
```

It hashes the sorted compact JSON mapping of relative filenames to SHA-256 for
the union of both run source manifests (`json.dumps(..., sort_keys=True,
separators=(',', ':'))`, UTF-8). The linked JSON contains every source hash;
this identity remains valid when documentation is committed later.

Both runs explicitly use `TILE_SIZE=16, HEAD_DIM=16, SEQ_LEN=64, SRAM_DEPTH=4096`.
Top additionally uses `AXI_ADDR_W=32, AXI_DATA_W=64`.
Generated hierarchy reports and mapped module declarations independently show:

```text
Standalone:
flash_attn_core_banked_prefetch_TILE_SIZE16_HEAD_DIM16_SEQ_LEN64_SRAM_DEPTH4096
  tile_controller_banked_prefetch_TILE_SIZE16_HEAD_DIM16_SEQ_LEN64

Integrated:
flash_attn_top_dma_banked_prefetch_SEQ_LEN64_HEAD_DIM16_TILE_SIZE16_SRAM_DEPTH4096_AXI_ADDR_W32_AXI_DATA_W64
  flash_attn_core_banked_prefetch_TILE_SIZE16_HEAD_DIM16_SEQ_LEN64_SRAM_DEPTH4096
    tile_controller_banked_prefetch_TILE_SIZE16_HEAD_DIM16_SEQ_LEN64
```

The report headers, controller specialization and netlist evidence establish
N64/d16; the run tag or requested parameters alone were not used as proof.

## Synthesis configuration and metric definitions

| Setting | Both runs |
|---|---|
| Tool | Synopsys Design Compiler R-2020.09-SP4 |
| Library / corner | FreePDK45 `gscl45nm.db` / typical |
| Library SHA-256 | `4968d1dba7ff9911cc51dfac7d8ea8b94fbab59d2f862f0531d7c774fb0a5791` |
| Flow | `syn/scripts/run_server.sh`, `dc_run.tcl`; `compile` |
| Run / profiles | `ppa_20260924_n64d16` / `core`, `top` |
| Clock / I/O | One `clk`, 10 ns; 1 ns input and output delays |
| Reset | `rst_n` ideal, false path from reset |
| Memory | Logical `sram_1r1w` / `exp_lut` blackboxes |
| Physical assumptions | No driving cell, output load, clock uncertainty, CTS, placement, routing or extracted parasitics |

The emitted SDC files have different port lists because the interfaces differ;
their clock, I/O-delay policy and reset exception are identical. Exact SDC hashes
and extracted commands are retained in the public provenance.

- **Area:** `area.rpt: Total cell area`, cross-checked with `qor.rpt: Cell Area`.
  Library area units, not die area or mm². Macro/blackbox area is zero.
- **Critical path:** `qor.rpt: Critical Path Length` in the `clk` max/setup group.
  It is not an Fmax measurement; I/O delays and setup time prevent simply
  equating it with `10 ns − slack`.
- **Worst setup slack:** `Critical Path Slack`, cross-checked with the first
  `timing_setup.rpt` path. Positive slack is reported directly; DC's separate
  design-level WNS summary clamps positive margin to zero.
- **TNS / violations:** `Total Negative Slack` / `No. of Violating Paths` in
  `qor.rpt`. These are DC-reported path-group metrics, not a gate-level proof.
- **Integration difference:** `(top cell area − standalone core cell area) /
  standalone core cell area`. Both endpoints are this run's N64/d16 pair.

## Hierarchy and critical paths

| Hierarchy (includes descendants) | Standalone | In integrated top |
|---|---:|---:|
| 256 dequantizers | 3,691,327.0698 | 3,691,327.0698 |
| Array controller + systolic array | 590,842.1344 | 590,834.6256 |
| 16 online-softmax lanes | 424,565.8596 | 424,565.8596 |
| Output-buffer logic | 20,298.6333 | 20,298.6333 |
| Address generator | 12,770.1225 | 12,770.1225 |
| Tile controller | 1,682.9098 | 1,691.8265 |
| Tile loader | 940.9465 | 940.9465 |
| Q/K/V scratchpad peripheral logic | 4,740.3993 | 4,740.3993 |

Rows are selected blocks, not an exhaustive area sum. In-context `u_core` is
5,061,564.3360; `u_dma` is 4,139.2260. Top-local control/configuration/
counters/glue/arithmetic account for 8,206.6491.
Top minus in-context core is 12,345.8751; it differs from
top minus standalone core because mapping context can change the core's area.
Do not describe the net integration delta as isolated DMA area.

| Worst path | Startpoint | Endpoint |
|---|---|---|
| Core | `combined_scale_reg_reg[31]` | `gen_dequant[54].u_deq/data_out_reg[14]` |
| Top | `u_core/combined_scale_reg_reg[31]` | `u_core/gen_dequant[51].u_deq/data_out_reg[14]` |

See the timing excerpts for the exact measured endpoints. Hierarchy area comes
from `reports/area.rpt`; parameters from `reports/hierarchy.rpt`, elaboration
manifest and mapped declarations; timing/TNS from `qor.rpt` and `timing_setup.rpt`.

## Disposition of older claims

| Older claim | Disposition | Evidence / replacement |
|---|---|---|
| Final core 5,061,246.151; 5.96 ns; +3.96 ns | Update; old core was N16 | Use this N64 row; do not attribute its delta solely to the fixes |
| Final top 5,073,263.046; 6.02 ns; +3.90 ns | Update current result; retain old result only as September 4 history | Now 5,073,910.211; 6.01 ns; +3.91 ns |
| 100 MHz logical-synthesis target | Retain | New N64 pair above; not physical signoff |
| 0.23743% / approximately 0.24% September 4 integration overhead | Invalid configuration comparison | N16 core versus N64 top; replace with 0.24477% from this pair |
| September 3 matched N64 integration delta 0.23364% | Historical only | Do not combine its +3.93 ns slack with another run; current overhead uses only this new pair |
| Shared Q/K scaling reduced core area 7.36% | Update rounding to **7.35%**; retain the historical optimization experiment | Exact reduction 7.3549917%; clean N64/d16 commits `4aa075f` → `bacdec6`; never substitute today's endpoint |
| Divider removal reduced total area 29.25%, power 10.69% | Invalid as isolated optimization deltas | N16 → N64; vectorless power also excludes physical memories |
| Dequantizers approximately 72.8% of final top | Update current hierarchy share | Now 72.751%; selected hierarchy sum |
| DMA fill 312→56 (5.57×), drain 256→17 (15.06×) | Retain | Audit-fix regression remeasured both; 256 B, modeled latency 10 for fill |
| Memory-path E2E 15332→12140 / 60421→46412 (1.263× / 1.302×) | Retain with stage definition | Audit-fix regression remeasured flat/byte versus banked/vector d16/d64, latency 0 |
| Prefetch −1.878% / −1.732%; fusion −34.52% / −36.06% | Historical milestones only | Historical before counts 11912/45608; current fused 7800/29160 revalidated; not new PPA or DMA-only causal claims |
| Alternative d64 baseline 46,524 cycles | Remove: insufficient provenance | No supporting report; verified banked baseline is 46,412 |
| Signoff power, physical memory-inclusive area, measured Fmax or exhaustive coverage | Unsupported | Not established by these reports or the verification suite |

**+3.90 ns:** is supported by the September 4 N64 top report. The corrected DMA-integrated top measures **+3.91 ns**. Use that value for the current top; the new standalone core separately measures +3.90 ns.

**7.36%:** the historical shared-scale benefit is supported, but this rounded
number should be **7.35%** (or approximately 7.4%). The raw areas are
5,463,487.889260 → 5,061,648.809908, a 7.354991674% reduction.
The old 7.36% matches rounding first to 7.355% and then to two decimals; direct
rounding of the unrounded ratio gives 7.35%. This is a reporting correction, not
a new optimization experiment.
Both clean-commit manifests, actual N64/d16 hierarchies and reports are retained;
the synthesis flow did not change between those commits. Library identity was
recorded by name/path historically, without a .db content hash. The next
registered-scale stage, September 4 cleanup, and September 24 repairs are distinct.
Both historical endpoints predate the current correctness fixes. Their older flow
logged a redundant unparameterized-design selection error; the actual elaborated
hierarchies and mapped reports independently confirm the intended N64/d16 designs.
The corrected current flow rejects such tool errors.
Public extracts: [historical shared-scale evidence](evidence/2026-09-24/historical_shared_scale.txt).

## Verification linked to this RTL

At publication commit `a121d7cda131a3279d2d32eff92c5dd7123b4641`, the prior audit-fix
source fingerprints matched its RTL, formal and test inputs. The linked original
provenance preserves 108 source hashes and ten PASS-log hashes for that snapshot.
Selected [verification excerpts](evidence/2026-09-24/verification.txt) are public.
No large regression was rerun during this documentation/synthesis-only closure.

| Check | Retained result and scope |
|---|---|
| Full Verilator regression | PASS, including legacy/reference and DRAM writeback tests under their individual acceptance criteria |
| Optimized N64 d16/d64 | Exact core/top comparisons; core cycles 7589/28337, top cycles 7800/29160 at latency 0 |
| Extreme fixed-point checks | Six full-core cases PASS; signed scales and INT16 saturation |
| Four-state SRAM initialization | Icarus and VCS block tests PASS, including unknown power-up, stale contents, reset, wrap and normalization |
| DMA / transaction contract | Boundary/stall/fault checks, same-DUT restart and common-reset recovery PASS |
| Controller formal | BMC48 and cover PASS; abstract loader/datapath completion, bounded safety only |

Formal depth is in solver steps, not full accelerator transactions. VCS evidence
is a focused block test, not a VCS full-system regression. No mapped-netlist
functional-equivalence proof or gate-level simulation is claimed.

A subsequent verification-only follow-up adds optimized N64/d64 causal core/top
targets, a directed N64/d16 mid-output reset/restart case, and scenario-report
failure propagation checks. See [follow-up evidence](evidence/2026-09-24/verification_followup.txt)
for results and changed test-source hashes. These TB/Makefile changes do not
replace the archived audit hashes. RTL, formal sources and synthesis inputs
remain identical to the measured PPA snapshot; no new PPA run is implied.

The optimization claims were checked individually:

| Claim | Disposition |
|---|---|
| Shared 16×16 array, 16 softmax lanes, zero exact mismatches at N64 d16/d64 | Retain; fixed-point reference, not FP32 equivalence |
| 5.57× / 15.06× / 1.26–1.30× memory improvements | Retain numbers; describe vectorized AXI DMA over the existing scalar AXI baseline |
| Active/shadow prefetch and bounded readiness/promotion/index checks | Retain; identify an abstract controller model, not full datapath formal proof |
| Fused update, −34.5–36.1% E2E cycles | Retain as its distinct historical fusion stage, d16/d64, latency 0; no new optimization claim |
| 256 removed dividers, shared scaling, −7.36% area, 100 MHz | Update to 7.35% and separate attribution: divider removal is structural; the area delta belongs to the historical shared-scale pair; final timing uses this new run |

Supported PPA wording:

> Optimized softmax/dequantization using Synopsys DC synthesis and static timing
> analysis; sharing Q/K scaling across 256 dequantization lanes reduced N64/d16
> core area by 7.35% in a matched historical comparison. The final N64/d16
> DMA-integrated top met 100 MHz with +3.91 ns setup slack in pre-layout synthesis.

The historical area comparison and final timing result are separate experiments,
both using logical SRAM/ROM blackboxes. Keep the integration difference in
technical evidence rather than standalone claims. No power-reduction or
physical-signoff claim is added.

## Reproduction and artifact policy

Use the source manifest to identify the exact RTL; supply licensed DC and the
matching library in your environment, then use fresh run tags:

```bash
CLK_PERIOD=10.0 IO_DELAY=1.0 ELAB_PARAMETERS='HEAD_DIM=16,SEQ_LEN=64' \
  syn/scripts/run_server.sh core unique_n64d16 compile
CLK_PERIOD=10.0 IO_DELAY=1.0 ELAB_PARAMETERS='HEAD_DIM=16,SEQ_LEN=64' \
  syn/scripts/run_server.sh top unique_n64d16 compile
```

Raw logs, full reports, DDC, mapped netlists and server-specific metadata remain
ignored. Public evidence includes relative report names, sanitized excerpts and
hashes. Timing remains limited by absent memory timing/physical modeling and
reported max-capacitance violations; no silicon signoff is implied.
