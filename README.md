# FlashAttention Hardware Accelerator

A cycle-accurate RTL implementation of the FlashAttention algorithm in
SystemVerilog, verified with Verilator across 18 regression targets, 0 mismatches.

---

## Problem Statement

Standard scaled dot-product attention materializes an N×N score matrix,
requiring O(N²) memory bandwidth. For N=1024 this is ~4 MB of intermediate
data written and read back per layer — bandwidth, not compute, is the bottleneck.

FlashAttention tiles the computation into TILE_SIZE×TILE_SIZE blocks and uses
online softmax (running max + running sum carried across KV tiles) so the
intermediate score matrix never leaves on-chip SRAM. This design implements
that algorithm in hardware using INT8 quantization and a systolic array.

---

## Architecture

```
AXI4-Stream IN  (Q: N×d, K/V: N×d' — d'=d for MHA, d'=d/GQA_RATIO for GQA)
         ↓
┌──────────────────────────────────────────┐
│           flash_attn_top_axi             │
│                                          │
│  ┌───────────────────────────────────┐   │
│  │        axi4_stream_slave          │   │
│  │  routes Q/K/V bytes → per-head    │   │
│  │  SRAMs; GQA: K/V broadcast to     │   │
│  │  grouped Q-head pairs             │   │
│  └──────────────┬────────────────────┘   │
│                 ↓  (4 cores, parallel)   │
│  ┌────────────────────────────────────┐  │
│  │       flash_attn_core × 4          │  │
│  │                                    │  │
│  │  Q/K/V SRAM  →  tile_controller    │  │ 
│  │                      ↓             │  │
│  │         ┌────────────────────┐     │  │
│  │         │  16×16 systolic    │     │  │
│  │         │  array (INT8 MAC)  │     │  │
│  │         │  QK^T and PV share │     │  │
│  │         └────────┬───────────┘     │  │
│  │                  ↓                 │  │
│  │         online_softmax × 16        │  │
│  │         (exp LUT, running max/sum) │  │
│  │                  ↓                 │  │
│  │         output_buffer              │  │
│  │         (INT32 accum + rescale     │  │
│  │          + normalize across tiles) │  │
│  │                                    │  │
│  │         kv_cache (prefill/decode)  │  │
│  └────────────────────────────────────┘  │
│                 ↓                        │
│  ┌───────────────────────────────────┐   │
│  │       axi4_stream_master          │   │
│  │  streams INT32 output row-major   │   │
│  └───────────────────────────────────┘   │
└──────────────────────────────────────────┘
         ↓
AXI4-Stream OUT (attention output, INT32)
```

---

## Memory Hierarchy (DDR/HBM → DMA → Banked Scratchpad)

A DMA-managed banked scratchpad hierarchy was added in front of the systolic
core, built up in stages so each layer is independently verified. There are
four coexisting data paths (all produce bit-exact output vs the golden model):

**1. Baseline (original)** — data is resident before compute:
```
AXI-stream / scalar preload → flat Q/K/V SRAM → byte-serial tile load
                            → tile registers → 16×16 systolic array
```

**2. DMA path** — off-chip memory pulled into the flat SRAMs:
```
axi_mem_model → dma_engine → flat core SRAMs → existing core (flash_attn_core)
```
`flash_attn_top_dma` bulk-loads Q then streams KV tiles; the DMA writes one byte
per cycle into the existing per-head SRAMs. The compute core is unchanged.

**3. Banked path (current full demo: `flash_attn_top_dma_banked.sv`)** — vector
DMA into a true banked scratchpad feeding the array at row width:
```
axi_mem_model → dma_engine_vec → banked_scratchpad → banked_tile_loader
             → flash_attn_core_banked → 16×16 systolic array
```
- `banked_scratchpad` = 16 byte-wide banks, `bank = addr[3:0]`, so 16 contiguous
  bytes (a tile row) are a conflict-free stripe.
- `dma_engine_vec` collects two 64-bit AXI beats into one 128-bit stripe and
  writes it with `w_vec=1` (16 bytes/write).
- `banked_tile_loader` fills `Q_reg`/`K_reg`/`V_reg` at one 16-byte stripe per
  cycle instead of byte-serial.

**4. Write-back path (`flash_attn_top_dma_banked_wb.sv`)** — extends path 3 with
an AXI write master so results return to DRAM, closing the full round-trip:
```
DRAM → dma_engine_vec → banked_scratchpad → flash_attn_core_banked
     → output_buffer → output_writeback_packer → dma_write_engine → DRAM
```
The read side is identical to path 3 (same scheduler, residency gate, and
read-side performance counters). After the core finishes, the output buffer is
drained and streamed back to DRAM; top-level `done` asserts only after the write
DMA's final `B` response. This path is **additive** — it wraps path 3 and does
not replace or alter the green read-only DMA tops.

**Shared mechanism.** An open-loop scheduler fetches each KV tile from DRAM once
(K/V is reused across every outer Q-tile) and bumps a monotonic `kv_tiles_ready`
count; a single residency gate stalls `S_LOAD_KV` until the needed tile is
resident. The gate is tied to `16'hFFFF` in the non-DMA tops, so the original 18
regressions are byte-for-byte unchanged.

### AXI write-back DMA
The output half of the round-trip was built as three additive modules (they do
not touch the read-only DMA path, `flash_attn_core_banked`, or `output_buffer`):
- **`dma_write_engine.sv`** — synthesizable AXI4 **write master** (AW/W/B). Takes
  a descriptor `{dest byte base, length}` and a 64-bit source stream, splits it
  into ≤16-beat INCR bursts (full-beat writes, `WSTRB` all ones), and pulses
  `done` after the final `B`. It is the mirror of `dma_engine_vec`; kept separate
  so the read engine is untouched and the two can run on independent channels.
- **`output_writeback_packer.sv`** — drains the output buffer through its 1-cycle
  read port and packs two consecutive 32-bit result words into one 64-bit AXI
  beat (`word[2k]`→`[31:0]`, `word[2k+1]`→`[63:32]`), presenting a valid/ready
  source stream. It holds a beat when the engine back-pressures (between bursts /
  under write latency), so no data is lost.
- **`axi_mem_model_rw.sv`** — behavioral read/write DRAM model: the AR/R side is
  byte-identical to `axi_mem_model`, plus an AW/W/B write slave and a programmable
  `wr_latency` (sim-only).
- **`flash_attn_top_dma_banked_wb.sv`** — the integration top that proves the full
  `DRAM → compute → DRAM` round-trip end to end, adding `cfg_o_base` and the
  `perf_wb_bytes` / `perf_wb_cycles` / `perf_wb_beats` counters alongside the
  preserved read-side counters. Read and write phases are temporally disjoint
  (write starts on core done), so one shared RW model serves AR/R then AW/W/B
  with no contention.

### Synthesizable vs simulation-only
- **Synthesizable:** `dma_engine`, `dma_engine_vec`, `dma_write_engine`,
  `output_writeback_packer`, `banked_scratchpad`, `banked_tile_loader`,
  `stripe_reader`, `tile_controller_banked`, `flash_attn_core_banked`,
  `flash_attn_top_dma`, `flash_attn_top_dma_banked`, `flash_attn_top_dma_banked_wb`.
- **Simulation-only:** `axi_mem_model` (behavioral AXI4 slave DRAM with
  programmable `rd_latency`), `axi_mem_model_rw` (read/write variant, adds AW/W/B
  and `wr_latency`); the `tb_*` harnesses and benchmarks.

### Verification targets
| Target | Stage | Checks |
|---|---|---|
| `make tb_dma_unit`            | DMA   | AXI4 read-master delivers a tile correctly (latency sweep) |
| `make dma_top_N64`            | DMA   | byte DMA + flat core, end-to-end vs golden |
| `make tb_banked_scratchpad`   | 3B-1  | bank routing, stripe R/W, conflict-free invariant |
| `make dma_bench`              | 3B    | scalar vs stripe scratchpad **drain** |
| `make dma_vec_bench`          | 3A    | scalar vs vector DMA **fill** |
| `make tb_banked_tile_loader`  | 3B-1  | stripe→tile-register layout (d=16 and d=64) |
| `make core_banked_N64`        | 3B-2  | banked core end-to-end (TB preload), d=16 |
| `make core_banked_N64_d64`    | 3B-2  | banked core end-to-end (TB preload), d=64 |
| `make dma_banked_top_N64`     | 3B-3  | vector DMA + banked core, end-to-end, d=16 |
| `make dma_banked_top_N64_d64` | 3B-3  | vector DMA + banked core, end-to-end, d=64 |
| `make dma_banked_prefetch_top_N64`     | 3B-3  | + KV double-buffer prefetch, d=16 |
| `make dma_banked_prefetch_top_N64_d64` | 3B-3  | + KV double-buffer prefetch, d=64 |
| `make tb_dma_write_unit`      | WB-1  | AXI4 write master (AW/W/B) + RW DRAM model, multi-burst / WLAST / byte-exact |
| `make wb_drain_bench`         | WB-2  | output-buffer drain → packer → write master, byte-exact + back-pressure |
| `make dma_banked_wb_top_N64`     | WB-3  | full DRAM→compute→DRAM round-trip, d=16 |
| `make dma_banked_wb_top_N64_d64` | WB-3  | full DRAM→compute→DRAM round-trip, d=64 |
| `make regression`             | all   | every target above + the original suite |

### Results
| Target | Meaning | Cycles / Result |
|---|---|---|
| `dma_top_N64`        | byte DMA + flat core            | 15332 cycles |
| `core_banked_N64`    | banked core, TB preload         | 11929 cycles |
| `dma_banked_top_N64` | vector DMA + banked core        | 12140 cycles |
| `dma_vec_bench`      | scalar vs vector DMA fill (256 B) | 312 vs 56 cycles |
| `dma_bench`          | scalar vs stripe scratchpad drain (256 B) | 256 vs 17 cycles |
| `make regression`    | all tests                       | PASS |

The banked path is **1.26×** faster end-to-end than the byte-DMA + flat-core path
(15332 → 12140), all bit-exact.

### Write-back DMA verification results
The full write path is verified bottom-up, then end-to-end, all green:

| Target | Result | Notes |
|---|---|---|
| `tb_dma_write_unit` (WB-1) | PASS | AW/W/B, multi-burst, WLAST, byte-exact placement, `wr_latency` sweep |
| `wb_drain_bench` (WB-2)    | PASS | packer byte-exact + holds under back-pressure, `bytes = words*4` |
| `dma_banked_wb_top_N64` (WB-3, d=16)     | PASS | `rd_bytes=3072`, `wb_bytes=4096`, `wb_beats=512`, `kv_tiles=4`, `max_rel=0` |
| `dma_banked_wb_top_N64_d64` (WB-3, d=64) | PASS | `rd_bytes=12288`, `wb_bytes=16384`, `wb_beats=2048`, `kv_tiles=4`, `max_rel=0` |
| `make regression` | PASS | full suite incl. all read + write-back targets |

`rd_bytes = 3·N·D` (Q+K+V streamed once), `wb_bytes = N·D·4` (int32 output words),
`wb_beats = wb_bytes/8` (two words per 64-bit beat); the O region read back from
DRAM matches the golden `expected.hex` word-for-word (`max_rel = 0`) at every
`{rd,wr}_latency ∈ {0,20,100}`.

### Runtime-configurable DMA scheduler + performance counters
`flash_attn_top_dma_banked` exposes a small set of **runtime** scheduler inputs
(sampled when `start` is accepted) — `HEAD_DIM`/`TILE_SIZE` stay compile-time:
- `cfg_seq_len[15:0]` — number of KV tiles streamed = `cfg_seq_len / TILE_SIZE`.
  Must be `> 0`, `<= SEQ_LEN`, and a multiple of `TILE_SIZE`.
- `cfg_q_base` / `cfg_k_base` / `cfg_v_base[31:0]` — DRAM source byte bases for
  Q / K / V (scratchpad destinations are unchanged: Q@0, KV tile @ `tile_off`).
- `cfg_error` — asserted (combinational) for an invalid `cfg_seq_len`; the
  scheduler then refuses to start.

With `cfg_seq_len = SEQ_LEN` and the default bases the scheduler issues
byte-for-byte the same descriptors as before (identical 12140-cycle result).
This is **runtime-configurable DMA scheduling only** — the compute core's
sequence extent is still the compile-time `SEQ_LEN` (no runtime-length core), so
the valid use today is `cfg_seq_len == SEQ_LEN`.

It also exposes RTL-visible performance counters (reset at run start, held after
`done`). Measured N=64, d=16 (`make dma_banked_top_N64`):

| Counter | Meaning | Value |
|---|---|---|
| `perf_total_cycles`           | accepted `start` → `done`                | 12141 |
| `perf_dma_busy_cycles`        | scheduler active / DMA transferring      | 426 |
| `perf_core_busy_cycles`       | `core_start` → core `done`               | 11930 |
| `perf_dma_bytes`              | +16 per vector scratchpad write          | 3072 |
| `perf_kv_tiles_loaded`        | completed (K,V) tile pairs               | 4 |
| `perf_first_tile_wait_cycles` | `start` → `core_start` (fill before compute) | 211 |

These cross-check: `211 + 11930 = 12141` (core runs continuously after
`core_start`); `perf_dma_bytes = 3 × 64×16 = 3072` (Q+K+V each streamed once);
`perf_kv_tiles_loaded = 64/16 = 4`; and `perf_first_tile_wait_cycles = 211`
matches the gap between this path (12140) and the TB-preloaded `core_banked_N64`
(11929).

**Why `perf_total_cycles` = 12141 but the C++ loop reports 12140:** the RTL
counter starts one cycle earlier — it begins on the cycle `start` is *accepted*
inside the FSM, whereas the C++ testbench begins its tick-count loop the cycle
*after* it de-asserts `start`. The one-cycle offset is just where each side
places `t=0`; both describe the same run.

### DRAM latency sweep (banked path, N=64 d=16)
`make dma_banked_top_N64` runs the same config across `rd_latency ∈ {0,20,100}`,
comparing every run against the same golden `expected.hex`:

| rd_lat | result | tb_cyc | perf_total | dma_busy | core_busy | first_wait | dma_bytes | kv_tiles |
|---|---|---|---|---|---|---|---|---|
| 0   | PASS | 12140 | 12141 | 426  | 11930 | 211  | 3072 | 4 |
| 20  | PASS | 12380 | 12381 | 906  | 11930 | 451  | 3072 | 4 |
| 100 | PASS | 13340 | 13341 | 2826 | 11930 | 1411 | 3072 | 4 |

- **All latencies are bit-exact** against the same `expected.hex` (correctness is
  latency-independent — the residency gate prevents any read-before-load).
- **`perf_core_busy_cycles` stays constant at 11930**: once the core starts, its
  runtime does not depend on DRAM latency — subsequent tiles are already resident.
- **`perf_first_tile_wait_cycles` absorbs the DRAM latency** (211 → 451 → 1411):
  the exposed latency lives entirely in the fill-before-compute window.
- **`perf_total = first_wait + core_busy`** for every row (e.g. 1411 + 11930 =
  13341), so the whole end-to-end increase (+1200 at rd_lat=100) is exactly the
  growth of the first-tile wait.
- **`perf_dma_bytes = 3072` and `perf_kv_tiles_loaded = 4` are invariant** across
  latency — they count work done (Q+K+V streamed once; 4 KV tile pairs), not time.
- **`perf_total = tb_cycles + 1`** every row, for the `t=0`-placement reason above.

This concretely demonstrates **latency hiding for the N=64 d=16 banked DMA path**.
It is not a claim of arbitrary-shape runtime support or real HBM timing — the DRAM
is a behavioral model with a single programmable `rd_latency`.

### KV double-buffer prefetch (banked path, new variant)

The banked core above (`flash_attn_core_banked`) dropped the baseline's KV
prefetch double-buffer, so each inner iteration's KV tile load serializes before
compute. A **new variant** restores baseline-style KV double-buffering on the
banked path *without touching the green banked files* — it is a parallel variant,
not a replacement:

- `rtl/ctrl/tile_controller_banked_prefetch.sv`
- `rtl/top/flash_attn_core_banked_prefetch.sv`
- `rtl/top/flash_attn_top_dma_banked_prefetch.sv`

**Active / shadow registers + copy-style swap.** The core keeps active
`K_reg`/`V_reg` (read by the systolic-array slicing mux during compute) plus a
shadow `K_shadow`/`V_shadow`. While the current KV tile is being computed, the
loader streams the *next* KV tile into the shadow registers. On the inner-tile
boundary the FSM pulses `kv_swap_banks` and the core copies shadow → active in one
cycle — a **copy-style swap**, matching the proven baseline
`flash_attn_core`/`flash_attn_top`. (A lower-area ping-pong SELECT swap that
re-points the slicing mux instead of copying is left as future PPA work.) `Q_reg`
stays a single buffer — Q reloads only once per outer tile, so double-buffering it
buys almost nothing.

**Shared-loader interlock — why `S_PF_WAIT` is required.** The banked path has a
*single* `banked_tile_loader`, shared between the foreground load (`ld_start`:
Q load, first KV load, non-prefetched reloads) and the prefetch load (`pf_start`).
They are mutually exclusive by construction (different FSM states), so one loader
suffices; a `pf_load_active` flag de-multiplexes the loader's `done` into the
foreground `ld_done` vs the prefetch `pf_done`, and routes its stripe writes to
active vs shadow. The consequence: **`S_CHECK_INNER` must not fall back to
`S_LOAD_KV` while a prefetch is in flight.** Doing so would re-start the one loader
on top of the running prefetch, and the FSM would then latch the prefetch's `done`
as a foreground completion and compute on stale active registers. So when a
prefetch was launched (`pf_pending`), the FSM enters a new `S_PF_WAIT` state and
stalls for `pf_rdy` before swapping — it never double-starts the loader. (Baseline
`tile_controller.sv` *can* fall back to `S_LOAD_KV` only because its flat-SRAM path
has two independent load counters.) Prefetch start is still gated by the
`kv_tiles_ready` residency count: a not-yet-resident next tile is simply not
prefetched and takes the normal residency-stalled `S_LOAD_KV` path.

**Results (bit-exact vs the same golden `expected.hex`):**

| Target | No prefetch | Prefetch | Δ cycles |
|---|---|---|---|
| `core_banked_N64` (d=16, TB preload)     | 11929 | 11701 | −228 |
| `core_banked_N64_d64` (d=64, TB preload) | 45589 | 44785 | −804 |

DMA-fed top, `rd_latency` sweep (all rows bit-exact), `tb_cyc` end-to-end:

| rd_lat | `dma_banked_top_N64` | `dma_banked_prefetch_top_N64` | `dma_banked_top_N64_d64` | `dma_banked_prefetch_top_N64_d64` |
|---|---|---|---|---|
| 0   | 12140 | 11912 | 46412 | 45608 |
| 20  | 12380 | 12171 | 47372 | 46635 |
| 100 | 13340 | 13150 | 51212 | 50609 |

The prefetch variant is faster at every latency and shape. **The speedup is modest
(~2% at d=16, ~1.7% at d=64)** for the same Amdahl reason as everywhere else on this
path: the KV load is only a few percent of an inner iteration; the `output_buffer`
rescale/accumulate walks (256 cycles each, per tile) dominate and are unchanged. At
d=64 the non-prefetch top's `perf_core_busy` is a flat 45590 and the prefetch
variant brings it to 44786 (rd_lat 0) — the same −804 core-busy saving seen with TB
preload, now confirmed on the DMA path, because each d=64 KV load is 64 stripes
instead of 16.

**`core_busy` is not strictly latency-constant here (and that's expected).** The
non-prefetch top's `perf_core_busy_cycles` is a flat 11930 across latencies; the
prefetch variant's is 11702 / 11721 / 11740 for rd_lat 0 / 20 / 100. The prefetch
fires early (in `S_UPDATE_SOFTMAX`), so at high DRAM latency a few next tiles have
not streamed in yet — those prefetches gate off and become small in-core residency
waits instead of being hidden (graceful degradation toward the non-prefetch path).
Every run stays bit-exact, `perf_first_tile_wait_cycles` still absorbs the bulk of
the latency, and the prefetch variant is still faster than the non-prefetch top at
every latency. New targets: `core_banked_prefetch_N64`,
`core_banked_prefetch_N64_d64`, `dma_banked_prefetch_top_N64` (all in `regression`).

### Limitations (honest)
- The banked **DMA top is verified for N=64 at d=16 and d=64** — both the
  non-prefetch (`dma_banked_top_N64`, `dma_banked_top_N64_d64`) and the prefetch
  variant (`dma_banked_prefetch_top_N64`, `dma_banked_prefetch_top_N64_d64`), each
  bit-exact vs golden across the `rd_latency ∈ {0,20,100}` sweep. Only these two
  shapes are proven; other N (128/256) and other d are **not** claimed for the
  banked DMA path.
- `axi_mem_model.sv` / `axi_mem_model_rw.sv` are **simulation-only** — behavioral
  AXI slaves with programmable latency, not a real DDR/HBM controller.
- **Output write-back DMA is implemented and verified** on the banked path for
  N=64 at d=16 and d=64 (`dma_banked_wb_top_N64` / `_d64`): the output buffer is
  drained through `output_writeback_packer` and streamed back to DRAM over the
  `dma_write_engine` AXI4 write master, proving the full `DRAM → compute → DRAM`
  round-trip bit-exact vs golden. Honest caveats: it writes into the behavioral
  `axi_mem_model_rw` (not a real controller); the write-back path has **not** been
  run through Synopsys DC yet, so it has no memory-macro timing/area numbers; and
  it is **single-head** — 4-head / GQA banked write-back integration is future work.
- The **green** banked core removed the old byte-serial KV prefetch, so its
  end-to-end speedup is modest (the load phase is a fraction of total cycles, and
  the output-buffer rescale/accumulate walks dominate). KV double-buffering is
  restored in the separate prefetch variant above (`*_banked_prefetch`), but it
  only recovers ~2% because of that same Amdahl split — it does not change the
  conclusion that the output-buffer walks dominate.
- **Single-head only**; 4-head / GQA banked integration is future work.

New RTL: `rtl/interface/dma_engine.sv`, `rtl/interface/dma_engine_vec.sv`,
`rtl/interface/axi_mem_model.sv` (sim), `rtl/memory/banked_scratchpad.sv`,
`rtl/memory/stripe_reader.sv`, `rtl/memory/banked_tile_loader.sv`,
`rtl/ctrl/tile_controller_banked.sv`, `rtl/top/flash_attn_core_banked.sv`,
`rtl/top/flash_attn_top_dma.sv`, `rtl/top/flash_attn_top_dma_banked.sv`.
Prefetch variant (new, additive): `rtl/ctrl/tile_controller_banked_prefetch.sv`,
`rtl/top/flash_attn_core_banked_prefetch.sv`,
`rtl/top/flash_attn_top_dma_banked_prefetch.sv`.
Write-back path (new, additive): `rtl/interface/dma_write_engine.sv`,
`rtl/interface/output_writeback_packer.sv`, `rtl/interface/axi_mem_model_rw.sv`
(sim), `rtl/top/flash_attn_top_dma_banked_wb.sv`.

---

## Key Design Decisions

**16×16 INT8 Systolic Array — reused for QK^T and PV**
The same array handles both matrix multiplications via an input mux
(`is_pv_phase`), halving area vs two separate arrays. INT8×INT8→INT32
accumulation prevents overflow across 16 MAC operations. For HEAD_DIM=64,
the inner dimension is tiled into 4 chunks (NUM_CHUNKS=4); the PE
accumulators are not cleared between chunks (`array_no_clear`), so partial
sums accumulate correctly across passes.

**Online Softmax with exp LUT**
A 256-entry ROM covers exp(x) for x ∈ [−8, 0] in Q8.8 format — sufficient
because scores are always shifted to (score − running_max) ≤ 0 before
lookup, and exp(x < −8) ≈ 0. 16 softmax instances run in parallel (one
per Q-row). `running_max` and `running_sum` carry across KV tiles; the
output buffer applies a rescale correction (× exp(m_old − m_new)) whenever
a new tile lowers the running maximum, and a final normalize (÷ running_sum)
on the last tile.

**K Stored Row-Major, Transposed at Read Time**
QK^T requires K^T. K is stored row-major in a flat SRAM. The systolic array
data-slicing mux reads K with swapped row/col indices:
`K_reg[col * HEAD_DIM + chunk * TILE_SIZE + row]` — so K^T is obtained
without a transpose unit or extra cycles.

**KV Prefetch Double-Buffer**
While the systolic array processes the current KV tile, the next tile's K/V
data is prefetched from SRAM into shadow registers (`K_reg_nxt/V_reg_nxt`).
On tile completion, a single swap copies shadow → active in one cycle.
This overlaps SRAM reads with compute. Measured cycle reduction vs. a
serial-load baseline (both bit-exact, d=16) **grows with N** as load latency
amortizes:

| N | Prefetch | Serial | Reduction |
|-----|----------|--------|-----------|
| 64  | 13,585   | 16,669 | 18.5% |
| 128 | 48,161   | 62,553 | 23.0% |
| 256 | 180,289  | 241,969| **25.5%** |

**Dynamic SRAM Sizing**
`SRAM_DEPTH = SEQ_LEN × HEAD_DIM` is computed from parameters, so the
same RTL supports N=16 d=16 (256 entries) through N=256 d=64 (16384 entries)
without any structural changes.

**GQA (Grouped Query Attention)**
`GQA_RATIO` controls how many Q-heads share each KV-head. For `GQA_RATIO=2`
(LLaMA 2/3 / Mistral style): 4 Q-heads, 2 KV-heads — Q-heads 0,1 share
KV-head 0; Q-heads 2,3 share KV-head 1. The K/V AXI stream is
`NUM_KV_HEADS`-wide (half the bandwidth of MHA), and the AXI slave uses
separate address decompositions for Q vs K/V phases.

**KV Cache for Decode Mode**
Two SRAMs store K/V vectors indexed by token position (up to 256 tokens).
In decode mode (`mode=1`) a single query attends to all cached K/V; `kv_len`
sets the inner loop bound dynamically, enabling autoregressive generation
without recomputing past keys and values.

---

## Performance

Simulation cycles (Verilator, TILE_SIZE=16, with KV prefetch pipeline):

### Non-causal (prefill)

| N \ d | d=16    | d=64    |
|-------|---------|---------|
| 16    | 1,429   | —       |
| 64    | 13,585  | 52,429  |
| 128   | 48,161  | 185,081 |
| 256   | 180,289 | 691,057 |

### Causal (decoder self-attention)

| N \ d | d=16    | d=64    |
|-------|---------|---------|
| 64    | 9,655   | 37,393  |
| 256   | 101,689 | 390,337 |

Causal mode skips above-diagonal tiles (~50% fewer KV tiles at large N).

4-head AXI top (N=64): core done at cycle 25,881 (MHA d=16), 101,589 (MHA d=64), 21,785 (GQA d=16).

---

## Features

- **Tiled online FlashAttention** — running max + running sum carried across
  KV tiles; never materializes the full N×N score matrix
- **16×16 INT8 systolic array** — skewed input feeding, INT8×INT8→INT32 MAC,
  reused for QK^T and PV via input mux
- **HEAD_DIM=64** — inner-dimension tiling (4 chunks of 16); same RTL,
  backward-compatible with HEAD_DIM=16
- **Causal masking** — decoder self-attention; above-diagonal tiles skipped
  entirely (no wasted cycles)
- **KV prefetch pipeline** — double-buffer hides SRAM latency; measured cycle
  reduction scales 18→25% across N=64→256 (d=16) vs serial load
- **Dynamic SRAM depth** — `SEQ_LEN × HEAD_DIM`; tested N=16–256, d=16/64
- **4-head parallel attention** — four `flash_attn_core` instances share one
  AXI4-Stream interface
- **GQA (Grouped Query Attention)** — `GQA_RATIO` parameter; 4Q+2KV tested
  (LLaMA 2/3 / Mistral style); K/V stream bandwidth halved vs MHA
- **KV cache + decode mode** — append-only token storage; single-token
  decode attending to full prefill context; tested d=16 and d=64

---

## RTL Modules (19 files)

| Module | Path | Description |
|--------|------|-------------|
| `pe` | `rtl/systolic/pe.sv` | Single INT8×INT8→INT32 MAC PE |
| `systolic_array` | `rtl/systolic/systolic_array.sv` | 16×16 PE array, flat acc port |
| `array_controller` | `rtl/systolic/array_controller.sv` | Skewed-input FSM, no_clear mode |
| `quantizer` | `rtl/quantization/quantizer.sv` | Fixed-point → INT8 |
| `dequantizer` | `rtl/quantization/dequantizer.sv` | INT32×scale → Q8.8 (÷256) |
| `exp_lut` | `rtl/softmax/exp_lut.sv` | 256-entry ROM, Q8.8, 1-cycle |
| `online_softmax` | `rtl/softmax/online_softmax.sv` | Running max+sum, cross-tile rescale |
| `sram_1r1w` | `rtl/memory/sram_1r1w.sv` | 1R1W SRAM behavioral model |
| `kv_cache` | `rtl/memory/kv_cache.sv` | Append-only KV token store (max 256) |
| `q_tile_buffer` | `rtl/memory/q_tile_buffer.sv` | Q tile SRAM wrapper |
| `kv_tile_buffer` | `rtl/memory/kv_tile_buffer.sv` | K/V ping-pong buffer |
| `output_buffer` | `rtl/memory/output_buffer.sv` | INT32 accum + rescale + normalize |
| `addr_gen` | `rtl/ctrl/addr_gen.sv` | SRAM address generator, global offset |
| `tile_controller` | `rtl/ctrl/tile_controller.sv` | Two-level tile FSM (11 states) |
| `axi4_stream_slave` | `rtl/interface/axi4_stream_slave.sv` | Q/K/V byte stream → per-head SRAM, GQA-aware |
| `axi4_stream_master` | `rtl/interface/axi4_stream_master.sv` | INT32 output → AXI4-Stream |
| `flash_attn_top` | `rtl/top/flash_attn_top.sv` | Single-head top (no AXI) |
| `flash_attn_core` | `rtl/top/flash_attn_core.sv` | Single-head core + KV cache |
| `flash_attn_top_axi` | `rtl/top/flash_attn_top_axi.sv` | 4-head top with AXI4-Stream + GQA |

---

## Verification

| Test | Config | Mismatches |
|------|--------|------------|
| systolic_array | identity, all-ones, ramp, negative, back-to-back, random INT8 | 0 |
| quantizer | random Q8.8, scale sweep (100 cases) | 0 |
| exp_lut | full 256-entry sweep | 0 |
| online_softmax | single tile, dual tile, running max reset | 0 |
| sram_1r1w | write/read, sequential | 0 |
| addr_gen | overflow, counter stop | 0 |
| kv_tile_buffer | ping-pong full cycle | 0 |
| flash_attn_top | N=16/64/128/256 d=16 | 0 |
| flash_attn_top | N=64/128/256 d=64 | 0 |
| flash_attn_top | causal N=64/256 d=16 | 0 |
| flash_attn_top | causal N=64/256 d=64 | 0 |
| axi4_stream_slave | byte routing, mat_sel transition, second row | 0 |
| flash_attn_top_axi | MHA 4-head N=64 d=16 | 0 |
| flash_attn_top_axi | MHA 4-head N=64 d=64 | 0 |
| flash_attn_top_axi | GQA 4Q+2KV N=64 d=16 | 0 |
| kv_cache | write/read, fill, overflow guard, reset | 0 |
| flash_attn_core | prefill+decode N=32 d=16 | 0 |
| flash_attn_core | prefill+decode N=32 d=64 | 0 |
| **Total** | | **0** |

All tests run automatically via `make regression`.

---

## Synthesis

First-pass DC runs (R-2020.09-SP4, FreePDK45 `gscl45nm.db`, 10 ns / 100 MHz
target). **This is a first-pass, non-signoff DC flow.** SRAM/ROM are *logical
blackboxes*, so `Macro/Black Box Area` is reported as 0 in every run — real
memory-macro area and timing are **not** included. Areas are in FreePDK45
library area units.

**Pure-logic blocks — met the 10 ns target:**

| Design | Setup slack | Total cell area (lib units) |
|--------|-------------|-----------------------------|
| `systolic_array`     | +5.15 ns (MET) | 442,795.819439 |
| `dma_engine_vec`     | +6.62 ns (MET) | 4,181.462976   |
| `banked_tile_loader` | +8.00 ns (MET) | 970.043098     |

**Full core — `flash_attn_core_banked_prefetch` (macro-blackbox compile):**

The macro-blackbox compile completed mapping and produced a full DC output set —
`.ddc`, mapped netlist (`*_mapped.v`), SDC, and timing / area / power / QoR
reports. It did **not** meet the 10 ns target:

| Metric | Value |
|--------|-------|
| WNS (setup slack) | −35.45 ns |
| TNS | −75,662.33 |
| Violating paths | 4,096 |
| Critical path length | 45.42 ns |
| Leaf cell count | 1,815,578 |
| Sequential cell count | 43,592 |
| Total cell area | 7,709,924.248970 (FreePDK45 lib units) |

The critical path is inside the softmax:
`gen_softmax[*].u_softmax/running_sum_reg[...]` → `…/softmax_flat_reg[...]`.
This is a first-pass timing failure on the softmax / dequant / reduction
combinational paths. Note: **the memory macros are blackboxed here**, so unlike
an earlier direct full-core compile, this is *not* a flop-inference blowup — it
is real combinational-logic timing plus the fact that the run still lacks memory
macro `.db` area and timing. Both the failing paths and the missing macro models
must be addressed before these numbers mean anything.

See [`syn/README.md`](syn/README.md) for the DC scaffold, filelists, blackboxes,
and how to re-run. Local Synopsys bring-up notes are kept in
`synthesisprogress.md` (gitignored).

---

## How to Run

### Prerequisites

```bash
brew install verilator   # macOS
pip install numpy
```

### Full regression

```bash
cd sim/verilator
make regression
```

### Individual targets

```bash
# Single-head (no AXI)
make tb_top_N16                # N=16, d=16
make tb_top_N64                # N=64, d=16
make tb_top_N256               # N=256, d=16
make tb_top_N64_d64            # N=64, d=64
make tb_top_N256_d64           # N=256, d=64
make tb_top_causal_N64         # N=64, d=16, causal
make tb_top_causal_N256_d64    # N=256, d=64, causal

# 4-head AXI
make axi_top_N64               # MHA, N=64, d=16
make axi_top_N64_d64           # MHA, N=64, d=64
make axi_top_N64_gqa           # GQA 4Q+2KV, N=64, d=16

# KV cache / decode
make tb_kv_cache               # KV cache unit test
make tb_kv_decode              # prefill + decode, d=16
make tb_kv_decode_d64          # prefill + decode, d=64
```

### Regenerate test vectors

```bash
python golden/generate_test_vectors.py
python golden/generate_hw_expected.py
python golden/generate_multihead_data.py
python golden/generate_multihead_data.py --num_kv_heads 2 --out data/N64_gqa
python golden/generate_kv_cache_test.py
```

---

## File Structure

```
flashattn-accelerator/
├── rtl/
│   ├── systolic/       pe, systolic_array, array_controller
│   ├── quantization/   quantizer, dequantizer
│   ├── softmax/        exp_lut, online_softmax
│   ├── memory/         sram_1r1w, kv_cache, tile buffers, output_buffer
│   ├── ctrl/           addr_gen, tile_controller
│   ├── interface/      axi4_stream_slave (GQA-aware), axi4_stream_master
│   └── top/            flash_attn_top, flash_attn_core, flash_attn_top_axi
├── sim/verilator/
│   ├── Makefile        (regression + all individual targets)
│   └── tb_*.cpp
├── golden/
│   └── *.py            (HW-accurate Python reference models)
├── data/               (pre-generated test vectors, all configs)
└── roofline.png        (arithmetic intensity analysis)
```
