# FlashAttention Accelerator — RTL / ASIC Portfolio

**Tiled INT8 attention in SystemVerilog — from AXI reads to host-visible results.**

A host-programmable attention accelerator with one reused systolic compute engine, online softmax, persistent KV cache, physical grouped-query sharing, AXI DMA and external-memory writeback. V2 extends the verified V1 datapath into a bounded inference SoC; it has not been deployed in silicon or an FPGA.

The matched d16 full-SoC experiment demonstrates **58.394905% lower standard-cell area** with 16 shared dequantizers, for **5.21% more warm-job cycles**, versus the 256-lane baseline. A separate matched GQA experiment halves K/V read payload. Independent fixed-point references check every valid external output; the recorded bounded SoC qualification safety campaign passed **4,224 successful commands / 3,225,600 exact words**.

[Architecture](#architecture-and-supported-scope) · [Measured results](#key-results--separate-experiments) · [Verification](#verification) · [Quick start](#quick-start--bounded-portable-v2-verification) · [V1 history](#earlier-optimization-milestones--v1) · [Repository guide](#repository-guide)

## V1 to V2 — architecture evolution

V1 established the tiled QK/PV datapath, online softmax, banked data movement and fused output updates. V2 reuses those arithmetic blocks and adds persistent state, serial head scheduling and a host-controlled job lifetime.

| Boundary | Verified V1 main path | Current V2 host SoC |
|---|---|---|
| Workload | Single-head N64 prefill, d16/d64; causal/noncausal | N16 prefill and latest-position decode at L16/17/18, d16/d64; causal/noncausal |
| State and heads | One transaction per common reset | Four serial Q heads, two persistent physical KV groups; APPEND, CLEAR and ACK |
| Host/results | Start/done and output read port | AXI4-Lite configuration/status, AXI4 output writeback and IRQ |
| Tile loading | Banked staging with local K/V prefetch overlap | Foreground loads from the selected persistent cache; prefetch disabled in this controller |

Both versions share a **16 × 16 array (256 PEs)** for QKᵀ and PV, with 16 online-softmax lanes. Their measurements use different geometries and interfaces; the V1 and V2 results below retain their own baselines. [V1 design](docs/DESIGN.md) · [V2 architecture](docs/ARCHITECTURE.md)

## Architecture and supported scope

![Host-controlled GQA SoC: CSR and job ownership, read DMA, persistent cache, shared QK/PV compute, normalization and AXI writeback](docs/v2/figures/soc_architecture.svg)

The recommended top is [V2 host SoC](rtl/v2/soc/flash_attn_soc.sv), selected by [`soc.f`](rtl/v2/soc/soc.f). Four serial query heads share two physical KV groups and one 16×16 array. QK and PV reuse the same PEs. Dequantization selects 256 lanes (default) or 16 real, time-multiplexed lanes; output traversal selects full-row or query-row-aware decode.

- **Data movement:** 64-bit AXI4 reads load Q scratchpads and write K/V into the selected SRAM-backed cache group. Banked tile reads feed the shared compute engine.
- **Shared compute:** signed INT8 QK products accumulate in INT32. d64 uses four feature chunks; PV reuses the array with unsigned 8-bit exponential weights and signed INT8 V.
- **Online attention:** masked score tiles update running maxima, sums and rescaled outputs without writing the full attention matrix to external memory. Final normalization produces signed 32-bit fixed-point results.
- **Host integration:** captured descriptors drive the serial engine; the packer sends results to the AXI write DMA. The CSR/adapter owns completion and result lifetime.

The C++ host and external-memory responder are simulation-only. The synthesizable design includes 81 SRAM and 16 ROM instances; their physical implementations remain logical blackboxes in the published PPA.

Supported host jobs are N16 prefill, single-query decode at L16/17/18 (absolute position L−1), APPEND, CLEAR and result ACK. HEAD_DIM is 16 or 64; cache capacity is 64.

INT8 inputs, Q8.8 scales and the preserved V1 fixed-point rounding, saturation and online-softmax rules are used. Exact fixed-point equality is not an FP32/FP16/BF16 accuracy claim. [Architecture](docs/ARCHITECTURE.md) · [Numerical limits](docs/NUMERICAL_ACCURACY.md)

### Stateful prefill, append and decode

PREFILL establishes 16 K/V tokens and computes 16 query rows per head. APPEND adds one token to each physical KV group; DECODE then computes one query per head against the resident sequence. The current host contract supports decode lengths 16, 17 and 18, although cache storage and append checks extend to capacity 64.

![V2 separate PREFILL and DECODE paths share one serial compute engine; independent APPEND atomically extends the persistent KV cache](docs/v2/figures/stateful_dataflow.svg)

The token sketches show logical sequence positions, not SRAM banks. PREFILL shows the valid cache after publication; its compute can consume complete K/V pairs before sequence metadata is published. APPEND is a separate host command with no Q load, compute or output writeback. Both KV groups must complete their paired stores before the new valid length is published.

For PREFILL/DECODE, successful host DONE and enabled success IRQ wait for source retirement, the final accepted OKAY B response and no outstanding writes. ACK releases completed ownership; IRQ clear is separate. Fatal AXI faults require common DUT/endpoint reset; completed external writes are not rolled back. CLEAR invalidates metadata without zeroing SRAM. Rejected or faulted operations cannot publish provisional cache contents. [Operating and safety contract](docs/v2/VERIFICATION.md) · [Reproducible figure source](tools/v2/generate_stateful_dataflow.py)

<details>
<summary>Stateful sequence and ownership diagram</summary>

![Stateful prefill, append and decode with complete-pair publication and ACK](docs/v2/figures/stateful_sequence.svg)

</details>

### GQA with physically shared KV cache

Q0/Q1 use KV0; Q2/Q3 use KV1. Two physical cache groups store and load K/V once per group, while four query heads execute serially on the same array. This saves K/V storage/load redundancy; it does not halve Q computation or create four parallel compute engines.

The earlier matched five-command serial GQA experiment measures **2,304 → 1,152 K/V read bytes (−50%)** and **3,456 → 2,304 total read bytes (−33.33%)** for 4Q:4KV → 4Q:2KV. The Q payload is unchanged. This read-payload comparison excludes later SoC writeback and AXI control overhead. [Workload and records](docs/RESULTS.md#physical-gqa-reuse--measured)

<details>
<summary>Matched GQA payload comparison</summary>

![Measured K/V and total read payload for the separate serial GQA workload](docs/v2/figures/gqa_traffic.svg)

</details>

### Dequantizer sharing and decode traversal

`DEQUANT_LANES=256` remains the parallel default. The 16-lane option assembles each 256-score tile in 16 ordered batches before softmax consumes it, preserving the arithmetic and the 256-PE array. Lane count is checked independently in the elaborated hierarchy. Sharing reduces mapped cell area while adding RTL cycles and reducing mapped timing margin.

Query-row-aware decode visits only the owned row during output update and normalization. Prefill retains full traversal; pending writes still drain before completion. The single-head 706 → 226 and 2,659 → 739 cycle comparisons are separate from four-head host latency. [Traversal experiment](docs/RESULTS.md#single-head-traversal-experiment--measured--verified)

<details>
<summary>Parallel and shared dequantizer datapaths</summary>

![256 parallel dequantizers versus 16 shared lanes with complete-tile publication](docs/v2/figures/dequantizer_sharing.svg)

</details>

### AXI, writeback and IRQ semantics

The AXI4-Lite slave exposes 32-bit configuration/status registers. Accepted commands capture their descriptor; busy attempts cannot replace the active job. Read and write DMA use 64-bit AXI4 payloads with response checking and backpressure handling.

For PREFILL/DECODE output jobs, compute completion starts packing/writeback. **Successful host DONE and success IRQ occur only after source retirement, the final accepted OKAY B response, and no outstanding writes.** ACK releases the result owner; IRQ status/enable/clear and error/reject events are explicit CSR operations. Fatal AXI faults require common DUT/endpoint reset; completed external writes are not rolled back. [CSR implementation](rtl/v2/soc/axi_lite_csr.sv) · [Job adapter](rtl/v2/soc/soc_job_adapter.sv)

## Key results — separate experiments

| Result | Recorded comparison | Scope |
|---|---|---|
| Decode traversal, d16 | 706 → 226 cycles; 3.12× | Single head, warm L16, modeled AXI0, 256 lanes |
| Decode traversal, d64 | 2,659 → 739 cycles; 3.60× | Same single-head workload; four dimension chunks |
| Shared dequantizer area | 5,758,250.967611 → 2,395,725.809614 library units; 58.394905% lower | Matched d16 complete SoC, 256 → 16 lanes, DC pre-layout, 10 ns |
| Sharing latency cost | 1,228 → 1,292 cycles; +5.21% | Four-head warm d16/L16 host job, including writeback, AXI0 |
| Physical GQA reuse | 50% fewer KV bytes; 33.33% fewer total read bytes | Matched d16 serial GQA five-command workload, 4Q:4KV → 4Q:2KV |
| Standard-cell job energy estimate | 2.246 → 2.059 µJ; 8.34% lower | Matched warm d16 SoC activity; dynamic energy only 0.30% lower |

These comparisons retain separate workload and source identities. SRAM/ROM are logical blackboxes; area and energy exclude physical memory and routed parasitics. d64 function and DC analysis/elaboration/link passed, but matched d64 PPA is unavailable. [Results and exact boundaries](docs/RESULTS.md)

### Matched d16 area, timing and activity-based power

![Matched d16 full-SoC standard-cell area, with logical memory blackboxes](docs/v2/figures/soc_area.svg)

| Measurement | 256 lanes | 16 lanes | Interpretation |
|---|---:|---:|---|
| Standard-cell area, library units | 5,758,250.967611 | 2,395,725.809614 | **58.394905% lower**, pre-layout DC |
| Setup slack at 10 ns, ns | +3.918528557 | +0.459291458 | Zero setup TNS/violating paths in both |
| Nonzero-limit max-cap violations | 2 | 1 | Electrical signoff remains open |
| Warm L16 host latency, cycles | 1,228 | 1,292 | Modeled AXI0; includes packing/writeback |
| Estimated total standard-cell power, mW | 182.763174 | 159.234151 | One matched warm workload |
| Estimated total job energy, µJ | 2.24615940846 | 2.05889757243 | **8.336979% lower** |
| Estimated dynamic job energy, µJ | 1.89573309201 | 1.88997247972 | Only **0.303873% lower** |

Area/timing use the earlier **frozen d16 PPA baseline d16 top**, checkpoint `5b486419`, with DC R-2020.09-SP4, typical gscl45nm, 10 ns clock, 1 ns I/O delays and ordinary `compile`. Both saved-DDC checks find zero mapped latches and unresolved references. These endpoints do not establish mapped equivalence or mapped PPA for the current d64 integration.

Power uses PrimeTime/PrimePower with the original matched d16 synthesis artifacts and measured VCS activity. The warm noncausal 4Q:2KV/L16 job excludes cache population, CSR setup and ACK. Its window includes the full DONE cycle: `(host latency + 1) × 10 ns`. Lower estimated total energy largely reflects leakage savings; dynamic energy is nearly unchanged. Physical memory, routed parasitics, clock-tree and system energy are excluded, and default/propagated activity remains disclosed. [PPA evidence](docs/v2/evidence/results/ppa.json) · [Power evidence](docs/v2/evidence/results/power.json)

## Verification

Independent bit-exact oracles check every valid external-memory output. Recorded Verilator and genuine VCS four-state campaigns cover stateful jobs, partial tiles, poisoned tails, ownership, AXI backpressure/faults, reset boundaries and meaningful negative controls.

| Layer | Recorded result and scope |
|---|---|
| Bounded actual-SoC safety regression | **120 executions / 4,224 successful commands / 3,225,600 exact words**, d16/d64 × 256/16 lanes; modeled latencies 0/20, three timing seeds and final-B hold |
| Safety/fault/reset checks | 504 fault cases and 264 reset checkpoints, recorded separately from successful commands; output ownership, complete-pair publication, head isolation and common-reset recovery |
| Meaningful negative controls | Eight rejected controls in that campaign: wrong final expected word and missing fixtures, with the specific detector, nonzero exit and no simulator PASS |
| Historical genuine VCS, d64 | **1,098 commands / 1,317,888 exact words**, plus 126 faults and 78 reset checkpoints; separate current SoC campaign |
| Fresh compilation and source identity | Strict lint, full production filelist rebuild, physical lane hierarchy and source/fixture/binary hashes; preserved V1 smoke and unit results remain separate |
| Linux / current layout CI | Four-profile Ubuntu matrix configured; **this revised layout has not run in Linux/V2 Actions** |

[Recorded SoC campaign](docs/v2/evidence/verification/verification.json) · [Safety scope](docs/v2/VERIFICATION.md) · [Historical VCS provenance](docs/v2/evidence/results/soc_provenance.json) · [Regression receipts](docs/v2/evidence/verification/regression.json)

Verification closure remains partial: cache formal is bounded to 24 steps under legal-host/complete-pair assumptions; the d16 host adapter passed 12 normalized steps with abstract child/DMA inputs. Deeper/integrated attempts are INCONCLUSIVE. Code, FSM and functional coverage are distinct and remain incomplete. No unbounded integrated proof or mapped equivalence is claimed. [Verification and gaps](docs/LIMITATIONS.md)

## Quick start — bounded portable V2 verification

Prerequisites: Verilator 5.046, C++ compiler/make, and a compatible Python 3 interpreter with NumPy. The runner stages the complete selected RTL, bench, oracle and data, freshly lints/builds, checks physical hierarchy, exact outputs and two required simulator failure controls. It installs nothing and fails on any failed verification step.

```sh
python3 -B tools/v2/verify.py --dim 16 --lanes 256 --compact 1 --scope prefill --latencies 0 --output-dir sim/verilator/obj_v2_prefill
python3 -B tools/v2/verify.py --dim 16 --lanes 16 --compact 1 --scope focus --latencies 0 --output-dir sim/verilator/obj_v2_sequence
# Optional d64: --dim 64; both 256/16 lanes are supported.
```

Use a fresh output directory. `--scope focus` executes prefill→append→decode→append→decode on one actual SoC. `--scope warm` checks changed populations and warm L16 jobs. `--latencies 0,20,100` selects simulation latency models, not HBM/DDR measurements. Linux/macOS execution receipts and CI execution status are reported separately in [release scope](docs/RELEASE.md). The inherited V1 jobs remain separate; the configured V2 CI matrix will rebuild all four dim/lane profiles and checks wrong-final-word/missing-fixture rejection.

On this Mac the existing NumPy requires Intel Python; use `arch -x86_64 /usr/local/bin/python3 -B` in place of `python3 -B`. The oracle defaults to the caller's interpreter; `--python-command` or `V2_PYTHON` can specify an existing compatible launcher when required. [Reproduction and contracts](docs/REPRODUCIBILITY.md)

For the accepted warm-cycle conditions, use `--scope warm --latencies 0 --seeds 0 --final-b-hold 0 --compact 1` with a new output directory. Canonical d16 cycles are 1,228/1,292 and d64 cycles 4,210/4,272 for 256/16 lanes. Nonzero seeds or extra final-B hold create different timing experiments. [Canonical receipts](docs/v2/evidence/verification/canonical_cycles.json)

The public package supports fresh RTL verification. Not every historical standalone benchmark is bundled; independent PPA/power reruns require licensed tools, the original library and mapped/activity artifacts. The runner's mandatory negative controls must pass by rejecting the injected failures, and any failed positive verification exits nonzero.

## Earlier optimization milestones — V1

V1 remains the verified single-head N64 foundation: vector DMA, banked scratchpads, local K/V prefetch, fused output updates and configurable dequantization. Its preserved source, verification and PPA are a separate experiment from the V2 host SoC.

<details>
<summary>V1 architecture, matched results and engineering evolution</summary>

![Preserved V1 architecture with local K/V prefetch and shared QK/PV compute](docs/figures/architecture.svg)

The V1 optimized main path overlaps local K/V prefetch with compute, reads results through an output port after `done`, and accepts one transaction per common reset. V2 adds persistent cache ownership and host writeback; the V1 diagram and results retain their original interface scope.

| V1 dequantizer lanes | Standard-cell area, library units | Area reduction | N64/d16 cycles | Setup slack |
|---|---:|---:|---:|---:|
| 256 (default) | 5,074,665.315 | — | 7,800 | +3.92 ns |
| 32 | 1,940,361.688 | 61.76% | 7,928 (+1.64%) | +0.66 ns |
| 16 | 1,715,678.209 | 66.19% | 8,056 (+3.28%) | +2.12 ns |

This is the matched N64/d16 DC sweep at the same 10 ns constraint, with logical SRAM/ROM blackboxes. Cycles are noncausal `start → done`, zero modeled read latency, excluding host setup/readout. Original `d5f9123` maps to equivalent implementation `dd68964`; d64 is functional-only for this synthesis sweep. The recorded RTL sweep passed 312 invocations / 624 transactions. [V1 experiment](docs/DEQUANT_EXPERIMENT.md) · [Original provenance](docs/evidence/2026-10-03/provenance.json)

![Preserved V1 optimization milestones and alternative dequantizer configurations](docs/figures/cycle-optimization-stages.png)

| Optimization | Improvement | Test conditions / baseline |
|---|---|---|
| Vector DMA | 312 → 56 cycles (5.57× faster fill) | 256 B, scalar → vector DMA, modeled read latency 10 |
| Banked scratchpad | 256 → 17 cycles (15.06× faster drain) | 256 B, byte-serial → 16-bank stripe read |
| Vector DMA + banked memory | 1.26× / 1.30× E2E speedup | N64 d16/d64, scalar memory → vector/banked memory |
| K/V prefetch | 1.88% / 1.73% fewer cycles | N64 d16/d64, vector/banked top → resident-tile prefetch top |
| Fused output update | 34.52% / 36.06% fewer cycles | N64 d16/d64, unfused → fused prefetch top |
| Shared Q/K scale | 7.35% less core cell area | N64/d16, per-lane → shared scale product, matched synthesis conditions |

Each row has its own baseline; reductions are not additive. E2E rows use noncausal `start → done` at zero modeled read latency. The historical 7.35% result concerns shared scale multiplication, not dequantizer lane reduction. [Historical evidence](docs/RESULTS_EVIDENCE.md) · [V1 coverage/power and limitations](docs/COVERAGE_POWER.md)

V1 reproduction uses the preserved [verification guide](docs/VERIFICATION.md#reproducing-checks); its build checkout must have no spaces. The [V1 synthesis guide](syn/README.md) records licensed tool prerequisites. V1 CI history does not certify V2 execution.

</details>

## Repository guide

| Directory | Contents |
|---|---|
| `rtl/` | Preserved reusable V1 datapath; [V2 production filelist](rtl/v2/soc/soc.f) selects the complete host SoC |
| `sim/` | V1 regression benches and [actual-SoC safety bench](sim/v2/soc/tb_soc.sv) with its [host scoreboard](sim/v2/soc/tb_soc.cpp) |
| `tools/v2/` | Current fresh-build runner, fixture support and release checks; [tool scope and dependencies](tools/v2/README.md) |
| `golden/`, `data/` | Independent fixed-point references, accepted fixtures and LUT inputs |
| `formal/` | Bounded properties with their original abstraction and assumption limits |
| `syn/` | Preserved synthesis/power flows and logical memory models; licensed inputs required |
| `docs/` | Architecture, results, reproduction and limitations; [V2 evidence index](docs/v2/README.md) |

Complete development history and large/private raw artifacts remain local. The public package retains the evidence needed to inspect its claims and reproduce the supported RTL checks.

## Known limitations and measurement scope

- **Bounded geometry:** fixed 4Q:2KV, d16/d64, N16 prefill and host decode L16/17/18. No arbitrary shape or model-level floating-point accuracy claim.
- **Verification:** two-state bounded SoC qualification tests supplement historical four-state campaigns; full integrated formal and coverage closure, mapped equivalence and exhaustive fault interleavings remain open.
- **PPA:** d16 mapped standard-cell results only; d64/16 mapping stopped at the 12 GiB resource guard, d64/256 mapping not run. Logical memories, electrical violations and absent physical/reset signoff preclude silicon or achieved-Fmax claims.
- **Reproducibility:** Linux/new V2 CI remains unverified; licensed PPA artifacts and the complete historical archive are not public quick-start dependencies.

[Detailed limits](docs/LIMITATIONS.md) · [Bounded release verdict](docs/RELEASE.md)

## Evidence and navigation

[Architecture](docs/ARCHITECTURE.md) · [Measurements / PPA / power](docs/RESULTS.md) · [Limitations](docs/LIMITATIONS.md) · [Revision history](docs/HISTORY.md) · [Release audit](docs/RELEASE.md)

<details>
<summary>Release provenance and preserved V1 references</summary>

V1 implementation, CI and N64 results remain documented in [V1 design](docs/DESIGN.md), [V1 verification](docs/VERIFICATION.md) and [V1 synthesis](syn/README.md). V1 CI does not certify V2. The [claim matrix](docs/v2/evidence/README.md) ties results to source hashes and original reports. `main` is the public primary branch; `git rev-parse HEAD` identifies the committed checkout. [Source and release identity](docs/HISTORY.md#release-identity) · [Verification scope](docs/RELEASE.md). Development preparation records are preserved locally under ignored `private/`.

Recorded SoC qualification checks passed the bounded Mac-validated portfolio scope. Current curation checks are separately recorded in [verification-v1](release/verification-v1.json). Linux execution and a new V2 Actions run remain unverified. [Release readiness report](docs/RELEASE.md).

</details>

Distribution is for **portfolio review; project reuse licensing remains pending**. No project-level open-source reuse license is granted; existing third-party notices retain their original scope. [Distribution status](DISTRIBUTION.md).
