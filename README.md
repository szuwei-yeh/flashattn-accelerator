# FlashAttention Hardware Accelerator

A cycle-accurate SystemVerilog implementation of tiled INT8 FlashAttention.
One shared **16 × 16 systolic array** performs both QKᵀ and PV. Sixteen online
softmax lanes preserve running state across tiles. Vector DMA, banked Q/K/V
scratchpads, K/V prefetch, and fused output updates reduce data-movement and
output-processing overhead without storing the full attention matrix in DRAM.

The main portfolio path is the **single-head DMA/banked/prefetch design**.
Earlier AXI4-Stream multihead/GQA/decode and non-prefetch writeback integrations
remain separate reference configurations.

## Repository guide

| Directory | Contents |
|---|---|
| `rtl/` | Synthesizable blocks, organized by function; the optimized entry point is `top/flash_attn_top_dma_banked_prefetch.sv` |
| `sim/verilator/` | Simulation testbenches, regression targets, and runner checks |
| `sim/icarus/` | Four-state output-buffer testbench, also used by VCS |
| `formal/` | Bounded controller safety and reachability checks |
| `golden/` | Numerical references, fixture generators, and numerical checks |
| `data/` | Versioned input fixtures, expected outputs, scales, and exponential LUT |
| `syn/` | Synthesis scripts, filelists, and logical memory blackboxes; see the [flow guide](syn/README.md) |
| `docs/` | Public results documentation and compact evidence in `evidence/2026-09-24/` |

Generated simulator builds remain in ignored `sim/verilator/obj_*` directories;
raw synthesis runs remain in ignored `syn/runs/`. Start with
[results evidence](docs/RESULTS_EVIDENCE.md) for the measured synthesis baseline.

## Measured results and scope

| Optimization | Before | After | Configuration / definition |
|---|---:|---:|---|
| DMA fill | 312 cycles | **56 cycles (5.57×)** | 256 B, scalar vs vector DMA, modeled read latency 10 |
| Scratchpad drain | 256 cycles | **17 cycles (15.06×)** | 256 B, byte-serial vs 16-bank stripe read |
| Memory-path E2E | 15,332 / 60,421 | **12,140 / 46,412** | N64, d16/d64, flat/byte DMA vs banked/vector DMA; 1.263× / 1.302× |
| Incremental K/V prefetch | 12,140 / 46,412 | **11,912 / 45,608** | Same historical memory stage; −1.878% / −1.732% |
| Fused output update | 11,912 / 45,608 | **7,800 / 29,160** | N64, d16/d64 prefetch top; −34.52% / −36.06% |
| Shared Q/K scale | 5,463,487.889 area units | **5,061,648.810 (−7.35%)** | Historical N64/d16 core, identical library/constraints |
| Final DMA-integrated timing | 10 ns target | **+3.94 ns setup slack** | Final N64/d16, `d2bb204`; mapped DC, memory blackboxes |

E2E counts are testbench `start → done`, with zero modeled read latency unless
stated otherwise. They exclude host initialization, output readout, and AXI
writeback. Memory, prefetch, and fusion rows belong to successive RTL milestones;
the total fusion gain is not a DMA-only speedup. RTL performance counters begin
one cycle earlier than the testbench count.

Final regression of the guarded RTL preserves the canonical core and full-top
cycle counts. Both **N64/d16** profiles now have matched mapped-synthesis
evidence from clean commit `d2bb204`. See the public
[results evidence](docs/RESULTS_EVIDENCE.md) for source identity, parameter proof,
report excerpts, metric definitions and the disposition of older claims.

## Architecture

```text
Host configuration + start (locked when accepted)
                         │
External memory ─ AXI AR/R, 64 bits ─ vector DMA
                         │ two beats → 128-bit stripe write
                  ┌──────┼──────┐
                  Q      K      V       scratchpads: 16 banks each
                  └──────┼──────┘
                    shared tile loader
                         │ Q: 16 B/cycle; K and V: 16 B/cycle each
             Q registers + K/V active and shadow registers
                         │
              shared 16×16 systolic array
             QKᵀ → dequantization → online softmax
                         │
               same array reused for PV
                         │
             scale_v → fused rescale + accumulate
                         │
           output SRAM → running-sum normalization
                         │
                   output read port
```

The DMA loads Q first, then streams K/V tile pairs. Compute starts once the first
pair is resident; later DMA transfers overlap compute. `kv_tiles_ready` increases
only after both K and V are complete. Prefetch moves an already resident next
K/V tile from scratchpad to shadow registers during processing of the current
tile. Promotion copies shadow to active. A shared-loader interlock prevents
foreground loading from colliding with prefetch.

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

## Final N64/d16 logical-synthesis PPA

Both profiles were mapped from clean commit `d2bb204` after the DMA
address-range fix, using Design Compiler R-2020.09-SP4, typical `gscl45nm.db`,
10 ns clock, 1 ns input/output delays and the unchanged `compile` flow.
SRAM/ROM remain logical blackboxes. Actual hierarchy and mapped declarations
confirm **SEQ_LEN64 / HEAD_DIM16** in both cores.

| Final mapped result | Standalone core | DMA-integrated top |
|---|---:|---:|
| Standard-cell area (library units) | 5,061,521.160 | 5,074,696.289 |
| Critical path length | 6.01 ns | 5.98 ns |
| Worst setup slack at 100 MHz | +3.90 ns | +3.94 ns |
| Setup TNS / violating paths | 0.00 ns / 0 | 0.00 ns / 0 |

Matched top-minus-core area is **13,175.128 units (0.26030%)**.
This includes mapping-context effects and is not isolated DMA area. The top
critical path starts at `u_core/combined_scale_reg_reg[31]` and ends at `u_core/gen_dequant[251].u_deq/data_out_reg[14]`.
Source manifests, netlist/SDC hashes and report extracts are in
[final provenance](docs/evidence/2026-09-30/provenance.json) and
[results evidence](docs/RESULTS_EVIDENCE.md).

The top reports **107,039 max-capacitance violations**. The
high-precision recheck finds a zero required-capacitance limit on every reported
violation; sampled library output pins have `max_capacitance=0`. This library
constraint issue does not establish physical electrical closure. Memory area
and access timing, placement, CTS, routing and extracted parasitics are absent.
DC vectorless power estimates are retained with their unannotated-activity
warnings; they are not workload or measured system power. No physical signoff
or independently measured Fmax is claimed.

The **7.35% shared-scale area reduction** remains the separate historical
N64/d16 comparison `4aa075f` → `bacdec6`, 5,463,487.889 → 5,061,648.810.
Its exact reduction is 7.3549917%; this release does not replace either endpoint
or attribute that delta to divider removal. The prior September 24 mapping is
retained as [archival evidence](docs/evidence/2026-09-24/provenance.json).

## Verification and operating contracts

Canonical exact regression covers N64 d16/d64. Core counts are 7,589 / 28,337;
DMA-prefetch counts are 7,800 / 29,160 at modeled latency zero. The DMA tests also
exercise latency 20 and 100. Directed checks cover configuration locking,
invalid configuration rejection, busy starts, causal operation, and reset/restart.
Optimized causal core/top regression includes both d16 and d64; the top runs
each at modeled read latencies 0/20/100. The N64/d16 contract test also resets
mid-output update after eight SRAM writes, changes V, then checks the restarted
transaction exactly. This is one directed abort point, not an all-phase reset sweep.

The Python reference models signed scales and INT16 dequantizer saturation.
Additional extreme-value full-core cases exercise saturation and negative scales
for both dimensions. Existing checked-in canonical expected files are preserved.

**Output memory:** first-K/V-tile updates explicitly overwrite old output state.
SRAM itself has no reset. A four-state test, checked under Icarus and VCS, covers unknown initial contents,
stale-data overwrite, reset/restart, 32-bit wrap, and signed normalization.
The optimized interface remains one transaction per reset.

**AXI:** the vector DMA splits bursts at 4 KiB boundaries and checks each accepted
RRESP/RLAST. It rejects descriptors whose source span exceeds the address space
or destination span exceeds the 4096-byte scratchpad. The optimized top rejects
wrapping Q/K/V matrix spans through `cfg_error` before accepting start, including
K/V spans split into separate descriptors. A span ending exactly at the final
legal byte is accepted. Invalid descriptors or bad responses latch `error`; the optimized
top exposes `dma_error`, suppresses successful `done`, and does not promote a
failed tile as resident. The offending stripe is not written. Partial results
are invalid. Recovery requires **common reset of master and slave**; there is
no timeout, outstanding-burst draining, or independent slave recovery. Legacy
non-prefetch wrappers do not expose the new host error status.

A directed scoreboard checks AR payload stability during backpressure, RVALID
gaps, multi-burst data conservation, a stripe crossing two bursts, error handling,
and reset recovery. Real integration monitors check loader stripe count/order,
active-bank ownership, and complete matching shadow promotion.

**Formal:** controller safety uses Yosys/SMT/Z3 BMC to depth 48 and reachability
covers. The harness abstracts datapath/loader completions. It checks resident
matching active tiles, valid shadow promotion, index bounds and loader request
exclusion. It is not an unbounded proof or a proof of SRAM/arithmetic contents.
The integration monitors above are simulation checks, not additional datapath
formal proofs.

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

## Reproducing checks

Install Verilator, a C++ compiler, Python 3 with NumPy, and Icarus Verilog for the
four-state check. Formal additionally needs Yosys, Z3 and SymbiYosys.

```bash
make -C sim/verilator regression
make -C sim/verilator audit_fixes
python3 golden/test_hw_reference.py
# Uses the d16/d64 core binaries built by regression; generates only temp files.
python3 golden/check_rtl_corners.py
python3 syn/scripts/test_run_metadata.py
make -C sim/verilator test_runners
make -C formal all
```

Selected targets:

```bash
make -C sim/verilator dma_vec_bench dma_bench
make -C sim/verilator core_banked_prefetch_N64 core_banked_prefetch_N64_d64
make -C sim/verilator dma_banked_prefetch_top_N64 dma_banked_prefetch_top_N64_d64
make -C sim/verilator core_banked_prefetch_causal_N64_d64 dma_banked_prefetch_causal_top_N64_d64
make -C sim/verilator tb_dma_vec_axi_protocol tb_dma_banked_prefetch_contract
make -C sim/verilator tb_oracle_checks
make -C sim/verilator numerical_checks
```

`regression` also runs the standalone 16×16 array-controller test, LUT sweep,
runner failure-propagation checks, and `tb_oracle_checks`. The latter exercises
the real optimized core/top binaries with mismatched geometry, missing/short
vectors, overflowing hex values, extra/partial tokens, invalid scales, and
one-bit errors in the first/last expected output.
It also corrupts LUT entries and builds a temporary loader with `done` suppressed
to verify that bad results and missing completion fail the test. All mutations
stay in temporary directories. Optimized E2E test geometry is tied to the RTL
build parameters, and all three scale values are required fixture inputs.

`numerical_checks` is also part of regression. It checks 40 temporary fixtures
across d16/d64 core and DMA top (160 transactions including latency 0/20/100),
without importing the hardware golden model. Closed-form cases cover equal
scores, saturation, signed/extreme V scales and causal prefix sums. Exact
permutation checks cover Q rows, paired K/V rows within tiles, and feature
columns across d64 chunk boundaries. Cross-tile K/V permutation is not asserted
bit-identical because fixed-point online rescaling can depend on tile order.

The licensed VCS four-state check is optional:
`make -C sim/verilator vcs_output_buffer_init VCS=/path/to/vcs`. Its runner
requires a fresh explicit PASS, a zero exit status, and no failure diagnostics;
VCS process status alone is insufficient after `$fatal`.

`make coverage` is a compatibility entry point for scenario tests. It builds and
runs its listed test targets, propagates failures, and prints a success summary
only when all prerequisites succeed; it does not report measured code/functional
coverage or a fabricated aggregate case count. Four-state, Python checks and
formal remain separate commands above. Historical full reports and development
notes stay local; public tables state the run/configuration boundaries needed
to interpret claims.

## Source organization

`rtl/core` contains compute variants; `rtl/top` contains integrations;
`rtl/interface` contains DMA/AXI and simulation memory models; `rtl/memory`
contains scratchpads, loaders and output storage; `rtl/systolic`, `rtl/softmax`
and `rtl/quantization` contain arithmetic blocks. `sim`, `formal`, `golden` and
`syn` hold the corresponding checks and flows.
