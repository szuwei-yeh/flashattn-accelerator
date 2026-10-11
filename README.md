# FlashAttention Accelerator — RTL / ASIC Portfolio

**Tiled INT8 attention in SystemVerilog — from a shared compute datapath to a stateful host SoC.**

## Project Overview

A **16 × 16 systolic array** performs QKᵀ and PV, while **16 online-softmax lanes**
carry state across tiles. V1 establishes the compute and memory foundation;
**V2 is the flagship implementation**, adding persistent KV cache, physical GQA
sharing, prefill/decode, host control and AXI output writeback.

The matched d16 V2 synthesis experiment reduces **standard-cell area by 58.39%**
with 16 shared dequantizers, for **5.21% more warm-job cycles** against the
256-lane baseline. A separate matched GQA experiment halves K/V read payload.
These are bounded RTL/pre-layout results; the design has not been deployed in
silicon or an FPGA.

[Overview](#project-overview) · [V2 SoC](#v2-soc-architecture) ·
[Datapath](#shared-attention-datapath) · [Prefill & decode](#stateful-prefill--decode) ·
[Optimizations & PPA](#key-optimizations--ppa) ·
[Verification & reproduction](#verification--reproduction) ·
[V1 & documentation](#v1-evolution--documentation)

## V2 SoC Architecture

![Host-controlled GQA SoC: CSR and job ownership, read DMA, persistent cache, shared QK/PV compute, normalization and AXI writeback](docs/v2/figures/soc_architecture.svg)

The [V2 host SoC](rtl/v2/soc/flash_attn_soc.sv), selected by [`soc.f`](rtl/v2/soc/soc.f), integrates host control, DMA, persistent KV storage and external output writeback. Four query heads execute serially over two physical KV groups on one shared compute engine.

- **Host control:** 32-bit AXI4-Lite configuration/status captures accepted descriptors; busy attempts cannot replace the active job.
- **AXI DMA:** 64-bit AXI4 reads load Q scratchpads and store K/V in the selected cache group; the write DMA returns results to external memory.
- **Persistent KV cache and GQA:** Q0/Q1 share KV0 and Q2/Q3 share KV1. Foreground banked reads feed compute, reducing KV storage/load redundancy while retaining all four query-head computations.
- **Job ownership:** the CSR/adapter captures configuration and owns completion, IRQ and result lifetime through compute, packing and writeback.

The C++ host and external-memory responder are simulation-only. The synthesizable design includes 81 SRAM and 16 ROM instances; their physical implementations remain logical blackboxes in the published PPA.

| Supported host contract | Scope |
|---|---|
| Heads / dimension | Fixed 4Q:2KV; HEAD_DIM 16 or 64 |
| PREFILL | 16 query rows per head; establishes 16 cached KV tokens |
| DECODE | One query per head at absolute position L−1; L16/17/18 |
| Cache / lifecycle | Capacity 64; separate APPEND, CLEAR and result ACK |

[Architecture and arithmetic](docs/ARCHITECTURE.md) · [Numerical limits](docs/NUMERICAL_ACCURACY.md)

## Shared Attention Datapath

**QKᵀ and PV reuse the same 16 × 16 array (256 PEs).** Sixteen online-softmax
lanes maintain per-query-row running maxima, sums and output-rescale factors as
KV tiles are processed. Only a score tile is staged at a time; the full attention
matrix is not written to external memory.

![Online softmax tile flow: eligible-key running max, masked exponential weights, shared QK/PV array phases, fused output update and final normalization](docs/v2/figures/online_softmax.svg)

The diagram shows an ideal recurrence with explicit fixed-point RTL notes.
QK and PV are scheduled phases of one physical array; arrows summarize tile
values and state factors rather than simultaneous hardware activity.

1. **QKᵀ:** signed INT8 Q and K produce INT32 dot-product accumulations. d64 uses four 16-feature chunks.
2. **Scale and mask:** dequantization applies signed Q8.8 scales, rounding, INT16 saturation and the supported 1/√d shift. Invalid and causal-masked keys are excluded from softmax maxima and sums.
3. **Online softmax:** each row updates its running maximum and sum, generates LUT-based exponential weights and rescales previous output state.
4. **PV:** unsigned 8-bit exponential weights and signed INT8 V reuse the array; the PV contribution applies Q8.8 `scale_v`. d64 processes four output-feature chunks.
5. **Fused output update:** one output-buffer traversal rescales the prior accumulator and adds the current PV contribution. First-tile updates ignore prior output contents.
6. **Final normalization:** after the last KV tile, the running output accumulator is divided by the running sum to produce signed 32-bit fixed-point results.

For supported L17 decode, the second 16-slot KV tile has one valid key; masks
exclude its 15 tail slots from attention.

Persistent state across host commands is the **KV cache**. Per-query online-softmax
state is initialized at the first KV tile of a new query pass. LUT approximation, saturation, weight
clamping and fixed-point rounding remain part of the numerical contract;
bit-exact reference agreement does not establish floating-point equivalence.

[Compute core](rtl/v2/compute/flash_attn_core_decode.sv) ·
[Masked online softmax](rtl/v2/compute/online_softmax_masked.sv) ·
[Fused output buffer](rtl/memory/output_buffer.sv) ·
[Reproducible figure source](tools/v2/generate_online_softmax.py) ·
[Arithmetic and numerical limits](docs/NUMERICAL_ACCURACY.md)

## Stateful Prefill & Decode

**PREFILL builds the cache; DECODE reuses it.** PREFILL establishes 16 K/V tokens and computes 16 query rows per head. A separate APPEND command adds one token per physical KV group; DECODE computes one query per head against the resident sequence.

![V2 separate PREFILL and DECODE paths share one serial compute engine; independent APPEND atomically extends the persistent KV cache](docs/v2/figures/stateful_dataflow.svg)

**Cache state persists across healthy jobs and ACK.** For PREFILL/DECODE,
successful DONE and enabled success IRQ wait for source retirement, the final
accepted OKAY AXI B response and no outstanding writes. ACK releases result
ownership; clearing IRQ is a separate operation.

<details>
<summary>Cache publication and fault-recovery contract</summary>

The sketch shows logical token positions. PREFILL compute may consume complete
K/V pairs before final metadata publication. APPEND performs no Q load, compute
or output writeback; both physical KV groups must complete paired K/V stores
before the new valid length is published.

CLEAR invalidates metadata without zeroing SRAM. Rejected or faulted operations
cannot publish provisional cache contents. Fatal AXI faults require common
DUT/endpoint reset; completed external writes are not rolled back.
[Operating and safety contract](docs/v2/VERIFICATION.md) ·
[Reproducible figure source](tools/v2/generate_stateful_dataflow.py)

</details>

## Key Optimizations & PPA

### Dequantizer sharing and query-row traversal

`DEQUANT_LANES=256` remains the parallel default. The 16-lane option assembles each 256-score tile in 16 ordered batches before softmax consumes it, preserving the arithmetic and the 256-PE array. Lane count is checked independently in the elaborated hierarchy. Sharing reduces mapped cell area while adding RTL cycles and reducing mapped timing margin.

Query-row-aware decode visits only the owned row during output update and normalization. Prefill retains full traversal; pending writes still drain before completion. The single-head 706 → 226 and 2,659 → 739 cycle comparisons are separate from four-head host latency. [Traversal experiment](docs/RESULTS.md#single-head-traversal-experiment--measured--verified)

<details>
<summary>Parallel and shared dequantizer datapaths</summary>

![256 parallel dequantizers versus 16 shared lanes with complete-tile publication](docs/v2/figures/dequantizer_sharing.svg)

</details>

### Physical KV sharing — matched traffic experiment

The earlier matched five-command serial GQA experiment measures **2,304 → 1,152 K/V read bytes (−50%)** and **3,456 → 2,304 total read bytes (−33.33%)** for 4Q:4KV → 4Q:2KV. The Q payload is unchanged. This read-payload comparison excludes later SoC writeback and AXI control overhead. [Workload and records](docs/RESULTS.md#physical-gqa-reuse--measured)

<details>
<summary>Matched GQA payload comparison</summary>

![Measured K/V and total read payload for the separate serial GQA workload](docs/v2/figures/gqa_traffic.svg)

</details>

### Measured results — separate experiments

| Result | Recorded comparison | Scope |
|---|---|---|
| Decode traversal, d16 | 706 → 226 cycles; 3.12× | Single head, warm L16, modeled AXI0, 256 lanes |
| Decode traversal, d64 | 2,659 → 739 cycles; 3.60× | Same single-head workload; four dimension chunks |
| Shared dequantizer area | 58.394905% lower standard-cell area | Matched d16 complete SoC, 256 → 16 lanes, DC pre-layout, 10 ns |
| Sharing latency cost | 1,228 → 1,292 cycles; +5.21% | Four-head warm d16/L16 host job, including writeback, AXI0 |
| Physical GQA reuse | 50% fewer KV bytes; 33.33% fewer total read bytes | Matched d16 serial GQA five-command workload, 4Q:4KV → 4Q:2KV |
| Standard-cell job energy estimate | 2.246 → 2.059 µJ; 8.34% lower | Matched warm d16 SoC activity; dynamic energy only 0.30% lower |

These comparisons retain separate workload and source identities. SRAM/ROM are logical blackboxes; area and energy exclude physical memory and routed parasitics. d64 function and DC analysis/elaboration/link passed, but matched d64 PPA is unavailable. [Results and exact boundaries](docs/RESULTS.md)

![Matched d16 full-SoC standard-cell area, with logical memory blackboxes](docs/v2/figures/soc_area.svg)

<details>
<summary>Exact d16 area, timing and activity-based power measurements</summary>

| Measurement | 256 lanes | 16 lanes | Interpretation |
|---|---:|---:|---|
| Standard-cell area, library units | 5,758,250.967611 | 2,395,725.809614 | **58.394905% lower**, pre-layout DC |
| Setup slack at 10 ns, ns | +3.918528557 | +0.459291458 | Zero setup TNS/violating paths in both |
| Nonzero-limit max-cap violations | 2 | 1 | Electrical signoff remains open |
| Warm L16 host latency, cycles | 1,228 | 1,292 | Modeled AXI0; includes packing/writeback |
| Estimated total standard-cell power, mW | 182.763174 | 159.234151 | One matched warm workload |
| Estimated total job energy, µJ | 2.24615940846 | 2.05889757243 | **8.336979% lower** |
| Estimated dynamic job energy, µJ | 1.89573309201 | 1.88997247972 | Only **0.303873% lower** |

Area/timing use the earlier **frozen d16 PPA baseline**, checkpoint `5b486419`, with DC R-2020.09-SP4, typical gscl45nm, 10 ns clock, 1 ns I/O delays and ordinary `compile`. Both saved-DDC checks find zero mapped latches and unresolved references. These endpoints do not establish mapped equivalence or mapped PPA for the current d16/d64 integration.

Power uses PrimeTime/PrimePower with the original matched d16 synthesis artifacts and measured VCS activity. The warm noncausal 4Q:2KV/L16 job excludes cache population, CSR setup and ACK. Its window includes the full DONE cycle: `(host latency + 1) × 10 ns`. Lower estimated total energy largely reflects leakage savings; dynamic energy is nearly unchanged. Physical memory, routed parasitics, clock-tree and system energy are excluded, and default/propagated activity remains disclosed. [PPA evidence](docs/v2/evidence/results/ppa.json) · [Power evidence](docs/v2/evidence/results/power.json)

</details>

## Verification & Reproduction

Independent bit-exact oracles check every valid external-memory output. Recorded Verilator and genuine VCS four-state campaigns cover stateful jobs, partial tiles, poisoned tails, ownership, AXI backpressure/faults, reset boundaries and meaningful negative controls.

| Layer | Recorded result and scope |
|---|---|
| Bounded actual-SoC safety regression | **120 executions / 4,224 successful commands / 3,225,600 exact words**, d16/d64 × 256/16 lanes; modeled latencies 0/20, three timing seeds and final-B hold |
| Safety/fault/reset checks | 504 fault cases and 264 reset checkpoints, recorded separately from successful commands; output ownership, complete-pair publication, head isolation and common-reset recovery |
| Meaningful negative controls | Eight rejected controls in that campaign: wrong final expected word and missing fixtures, with the specific detector, nonzero exit and no simulator PASS |
| Historical genuine VCS, V2 d64 | **1,098 commands / 1,317,888 exact words**, plus 126 faults and 78 reset checkpoints; separate current SoC campaign |
| Fresh compilation and source identity | Strict lint, full production filelist rebuild, physical lane hierarchy and source/fixture/binary hashes; preserved V1 smoke and unit results remain separate |
| Linux / current layout CI | Four-profile Ubuntu matrix configured; **this revised layout has not run in Linux/V2 Actions** |

[Recorded SoC campaign](docs/v2/evidence/verification/verification.json) · [Safety scope](docs/v2/VERIFICATION.md) · [Historical VCS provenance](docs/v2/evidence/results/soc_provenance.json) · [Regression receipts](docs/v2/evidence/verification/regression.json) · [V1 verification](docs/VERIFICATION.md)

Verification closure remains partial: cache formal is bounded to 24 steps under legal-host/complete-pair assumptions; the d16 host adapter passed 12 normalized steps with abstract child/DMA inputs. Deeper/integrated attempts are INCONCLUSIVE. Code, FSM and functional coverage are distinct and remain incomplete. No unbounded integrated proof or mapped equivalence is claimed. [Verification and gaps](docs/LIMITATIONS.md)

### Quick start

Use **Verilator 5.046**, a C++ compiler/make and Python 3 with compatible NumPy.
From the repository root:

```sh
v2_run_dir="$(mktemp -d)"

# V2 prefill: four query heads, two physical KV groups.
python3 -B tools/v2/verify.py --dim 16 --lanes 256 --scope prefill \
  --latencies 0 --output-dir "$v2_run_dir/prefill"

# V2 stateful sequence: prefill → append → decode → append → decode.
python3 -B tools/v2/verify.py --dim 16 --lanes 16 --scope focus \
  --latencies 0 --output-dir "$v2_run_dir/sequence"
```

Use `--dim 64` for d64; V2 supports 256 or 16 dequantizer lanes. Every run stages
sources in a temporary path without spaces, freshly lints/builds, checks physical
hierarchy and exact outputs, and rejects wrong-final-word/missing-fixture controls.
Failed verification exits nonzero. The runner installs nothing; reuse no output directory.

`--scope warm` checks changed Q/K/V/scale populations and warm L16 jobs.
`--latencies 0,20,100` selects simulation models, not HBM/DDR measurements.
On macOS, use an existing compatible Python/NumPy launcher; the validated setup
uses Intel Python. [Dependencies, platform notes and full commands](docs/REPRODUCIBILITY.md)

<details>
<summary>Canonical warm-cycle conditions and reproduction boundaries</summary>

For accepted warm-cycle conditions, use `--scope warm --latencies 0 --seeds 0
--final-b-hold 0 --compact 1` with a new output directory. Canonical host
command-to-DONE cycles are d16 **1,228/1,292** and d64 **4,210/4,272** for
256/16 lanes. Nonzero timing seeds or extra final-B hold define different experiments.
[Canonical receipts](docs/v2/evidence/verification/canonical_cycles.json)

The public package supports fresh RTL verification. Historical standalone
benchmarks are not all bundled; independent PPA/power reruns require licensed
tools, the original library and mapped/activity artifacts.

</details>

### Scope and remaining limitations

- **Bounded geometry:** fixed 4Q:2KV, d16/d64, N16 prefill and host decode L16/17/18. No arbitrary shape or model-level floating-point accuracy claim.
- **Verification:** two-state bounded SoC qualification tests supplement historical four-state campaigns; full integrated formal and coverage closure, mapped equivalence and exhaustive fault interleavings remain open.
- **PPA:** d16 mapped standard-cell results only; d64/16 mapping stopped at the 12 GiB resource guard, d64/256 mapping not run. Logical memories, electrical violations and absent physical/reset signoff preclude silicon or achieved-Fmax claims.
- **Reproducibility:** Linux/V2 CI validation of this revised layout remains pending; licensed PPA artifacts and the complete historical archive are not public quick-start dependencies.

[Detailed limits](docs/LIMITATIONS.md) · [Bounded release verdict](docs/RELEASE.md)

## V1 Evolution & Documentation

V1 is the verified single-head N64 foundation: vector DMA, banked staging,
local K/V prefetch, fused output updates and configurable dequantization.
Its matched d16 top uses **66.19% less standard-cell area for 3.28% more cycles**
with 16 dequantizers versus the 256-lane baseline. The recorded RTL sweep passed
**312 invocations / 624 transactions**. These measurements have their own source,
geometry and interface scope, separate from the V2 host SoC.

[V1 design](docs/DESIGN.md) · [V1 verification](docs/VERIFICATION.md) ·
[V1 results and evidence](docs/RESULTS_EVIDENCE.md) ·
[Engineering documentation](docs/README.md)

### Historical V1 — cycle milestones and dequantizer tradeoffs

Both figures show **single-head N64 V1 measurements**, noncausal, at zero
modeled read latency. Cycles cover `start → done`, excluding host setup,
output readout and AXI output writeback. Each d16/d64 panel uses its own axis scale.

![Historical V1 cycle milestones: scalar memory, vector and banks, K/V prefetch and fused update for N64 d16/d64](docs/figures/cycle-milestones.svg)

The four successive 256-lane stages show how data movement and fused output
updates reduce cycles. [Cycle data](docs/figures/cycle-milestones.csv) ·
[Historical measurement scope](docs/RESULTS_EVIDENCE.md)

![Historical V1 optimization stages and alternative 32/16-lane dequantizer configurations, with a separate N64/d16 area tradeoff](docs/figures/cycle-optimization-stages.png)

The 32- and 16-lane rows are **alternative configurations of the fused design**.
The area tradeoff is the separate matched **N64/d16** synthesis experiment;
d64 bars show RTL cycles only. [Dequantizer experiment](docs/DEQUANT_EXPERIMENT.md)

<details>
<summary>V1 architecture and the evolution to V2</summary>

### Original V1 architecture

![V1 architecture: banked staging and local K/V prefetch feed shared QK/PV compute, online softmax and fused output updates](docs/figures/architecture.svg)

- **Data movement:** 64-bit AXI reads fill 16-bank Q/K/V scratchpads; V1 overlaps local K/V shadow-tile prefetch with compute.
- **Shared compute:** QK uses signed INT8 inputs and INT32 accumulation; PV reuses the same 256 PEs with unsigned 8-bit exponential weights and signed INT8 V. d64 uses four dimension chunks.
- **Online attention:** score tiles update running maxima, sums and rescaled outputs without writing the full attention matrix to external memory.
- **Fused output update:** rescaling and accumulation share one buffer traversal, followed by final normalization.

This diagram describes the single-head V1 path. Its results are read through an
output port after `done`; it has no AXI output writeback. V2 reuses the arithmetic
blocks but uses foreground cache loads with prefetch disabled in its controller.
INT8 inputs, Q8.8 scales and fixed-point rounding/saturation are preserved;
bit-exact reference agreement is not floating-point equivalence.
[V1 design](docs/DESIGN.md) · [V1 results and source identity](docs/RESULTS_EVIDENCE.md)

### From V1 to V2

V2 extends the same arithmetic foundation with persistent state, serial head scheduling and a host-controlled job lifetime.

| Boundary | Verified V1 main path | Current V2 host SoC |
|---|---|---|
| Workload | Single-head N64 prefill, d16/d64; causal/noncausal | N16 prefill and latest-position decode at L16/17/18, d16/d64; causal/noncausal |
| State and heads | One transaction per common reset | Four serial Q heads, two persistent physical KV groups; APPEND, CLEAR and ACK |
| Host/results | Start/done and output read port | AXI4-Lite configuration/status, AXI4 output writeback and IRQ |
| Tile loading | Banked staging with local K/V prefetch overlap | Foreground loads from the selected persistent cache; prefetch disabled in this controller |

The two versions use different workload geometries and interfaces; each reported experiment retains its own matched baseline. [V1 design](docs/DESIGN.md) · [V2 architecture](docs/ARCHITECTURE.md)

</details>

<details>
<summary>V1 exact optimization results and reproduction</summary>

### Original V1 optimization results

| Optimization | Improvement | Test conditions / baseline |
|---|---|---|
| Vector DMA | 312 → 56 cycles (5.57× faster fill) | 256 B, scalar → vector DMA, modeled read latency 10 |
| Banked scratchpad | 256 → 17 cycles (15.06× faster drain) | 256 B, byte-serial → 16-bank stripe read |
| Vector DMA + banked memory | 1.26× / 1.30× E2E speedup | N64 d16/d64, scalar memory → vector/banked memory |
| K/V prefetch | 1.88% / 1.73% fewer cycles | N64 d16/d64, vector/banked top → resident-tile prefetch top |
| Fused output update | 34.52% / 36.06% fewer cycles | N64 d16/d64, unfused → fused prefetch top |
| Shared Q/K scale | 7.35% less core cell area | N64/d16, per-lane → shared scale product, matched synthesis conditions |

Each row has its own baseline; reductions are not additive. E2E rows use noncausal `start → done` at zero modeled read latency. The historical 7.35% result concerns shared scale multiplication, not dequantizer lane reduction. [Historical evidence](docs/RESULTS_EVIDENCE.md) · [V1 coverage/power and limitations](docs/COVERAGE_POWER.md)

### V1 — matched N64/d16 dequantizer sweep

| V1 dequantizer lanes | Standard-cell area, library units | Area reduction | N64/d16 cycles | Setup slack |
|---|---:|---:|---:|---:|
| 256 (default) | 5,074,665.315 | — | 7,800 | +3.92 ns |
| 32 | 1,940,361.688 | 61.76% | 7,928 (+1.64%) | +0.66 ns |
| 16 | 1,715,678.209 | 66.19% | 8,056 (+3.28%) | +2.12 ns |

This is the matched N64/d16 DC sweep at the same 10 ns constraint, with logical SRAM/ROM blackboxes. Cycles are noncausal `start → done`, zero modeled read latency, excluding host setup/readout. Original `d5f9123` maps to equivalent implementation `dd68964`; d64 is functional-only for this synthesis sweep. The recorded RTL sweep passed 312 invocations / 624 transactions. [V1 experiment](docs/DEQUANT_EXPERIMENT.md) · [Original provenance](docs/evidence/2026-10-03/provenance.json)

V1 uses the preserved runners and a checkout path **without spaces**:

```sh
# Single-head N64/d16 core/top smoke, causal and noncausal.
python3 -B sim/verilator/run_dequant_sweep.py --lanes 16 --dim 16 --smoke

# Full default V1 regression.
make -C sim/verilator regression
```

[V1 verification](docs/VERIFICATION.md#reproducing-checks) · [V1 synthesis](syn/README.md)

</details>

### Documentation and repository guide

| Directory | Contents |
|---|---|
| `rtl/` | Preserved reusable V1 datapath; [V2 production filelist](rtl/v2/soc/soc.f) selects the complete host SoC |
| `sim/` | V1 regression benches and [actual-SoC safety bench](sim/v2/soc/tb_soc.sv) with its [host scoreboard](sim/v2/soc/tb_soc.cpp) |
| `tools/v2/` | Current fresh-build runner, fixture support and release checks; [tool scope and dependencies](tools/v2/README.md) |
| `golden/`, `data/` | Independent fixed-point references, accepted fixtures and LUT inputs |
| `formal/` | Bounded properties with their original abstraction and assumption limits |
| `syn/` | Preserved synthesis/power flows and logical memory models; licensed inputs required |
| `release/` | Versioned public inventory, original evidence locks and source-path mapping |
| `docs/` | Architecture, results, reproduction and limitations; [V2 evidence index](docs/v2/README.md) |

Complete development history and large/private raw artifacts remain local. The public package retains the evidence needed to inspect its claims and reproduce the supported RTL checks.

Distribution is for **portfolio review; project reuse licensing remains pending**. No project-level open-source reuse license is granted; existing third-party notices retain their original scope. [Distribution status](DISTRIBUTION.md).
