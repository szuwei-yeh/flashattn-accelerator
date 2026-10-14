# Full L17 DECODE RTL waveform

These PNGs are native **Surfer screenshots of a Verilator VCD**, captured locally
on macOS from production `rtl/v2/soc/soc.f`. Only viewer chrome/empty space was
cropped; title/footer bands were added separately. Waveform pixels were preserved
exactly. Raw screenshot hashes remain in [image_provenance.json](image_provenance.json);
the original captures are preserved in a local private review archive.
The root README now integrates the overview and collapsible zooms. No RTL,
checked-in testbench, accepted evidence, or historical measurement was changed.
No remote server, synthesis, P&R, commit, or push was used.

| Item | Captured configuration/result |
|---|---|
| Source commit | `0d05a6cd695ce7fec78c4af4722300756cf9585d` |
| Simulator | `Verilator 5.046 2026-02-28 rev vUNKNOWN-built20260228` |
| Viewer | `surfer 0.7.0 (git: 1cdbaa9)`, native Apple Silicon build |
| Existing regression | `focus`, population 0, canonical timing seed 0 |
| Geometry | HEAD_DIM=16, 4Q:2KV, DEQUANT_LANES=16, compact output enabled |
| Stateful preparation | PREFILL N16 → APPEND token 16 → DECODE L17, no intervening common reset |
| Capture | First L17 DECODE, host job ID 3; no KV reconstruction |
| AXI model | Latency 0, with the existing deterministic legal AR/R/AW/W backpressure; final-B hold 0 |
| Accepted host command → DONE | **2,072 clock cycles**, 86,755 ns → 107,475 ns |
| Child accepted → child completion | 1,947 cycles, including Q loading and per-head result copies |
| External reads | 4 Q bursts, 8 accepted 64-bit R beats, 64 bytes; zero K/V DMA payload |
| Cache | Both committed lengths remain 17; 68 checked internal KV read events; zero KV writes |
| Compute | Heads 0,1,2,3 in order; KV groups 0,0,1,1; 16 shared-array starts |
| Output | 64 exact INT32 words / 256 bytes; 3 AW, 32 W, 3 accepted OKAY B responses |
| Final B → DONE / success IRQ | 107,455 ns → 107,475 ns: **2 rising edges** |

The 10 ns VCD clock is an observation timebase added to the untimed C++ tick
loop. It does not establish operating frequency, timing closure, or PPA.
The host cycle count excludes CSR setup and ACK, and includes dispatch, shared
four-head execution, result copying, packing and external writeback. It is a
different workload/boundary from the published warm-L16 single-head traversal
benchmark and the matched four-head warm-L16 latency comparison. It is RTL
simulation evidence, separate from synthesis area/power, Qwen activation accuracy,
FPGA/silicon measurements or physical-memory latency. No speedup is claimed here.

## Verification

- Existing `tools/v2/verify.py` regression: **PASS**, 5 healthy commands,
  1,152 exact output words and 218 CSR checks. It continues through APPEND token 17
  and DECODE L18 after the captured transaction.
- Independent golden-reference suite: **3 tests PASS**. Every one of the captured
  job's 64 external output words also matches its generated bit-exact golden.
- Physical hierarchy: **PASS**, 16 dequantizers, 256 PEs, two cache groups,
  one shared core, 81 logical SRAMs and 16 LUTs.
- Negative controls: wrong final output word and missing fixture **rejected as
  expected**; their nonzero exits are intentional PASS conditions for the verifier.
- Instrumented and untraced `focus` logs are **byte-identical**. Original assertions,
  the real cache-read scoreboard and final-B/source-retirement checks remain active.
- Independent VCD edge audit: **PASS**. Handshakes use pre-rising-edge values;
  DONE/IRQ use post-edge values. See [events.json](events.json).

The original busy-descriptor/duplicate-command checks occur during this DECODE.
They produce a brief rejection IRQ that the bench clears. `IRQ_pending=4` means
rejection, `1` means completion. No earlier rejection pulse is labeled as DECODE
success. Final success follows the final OKAY response.

## Displayed signals

[signals.json](signals.json) gives every displayed label and full VCD path.
`TOP.obs_*` are the existing simulation wrapper's direct observations in
[`tb_soc.sv`](../../../sim/v2/soc/tb_soc.sv); no observational ports were added.

| Group | Actual observations |
|---|---|
| Host | `clk`, `rst_n`, `obs_command`, `u_job.owned`, `obs_phase`, `obs_irq_pending` |
| Q DMA / cache | engine `desc_valid` and `dma_done`; `m_araddr`; `obs_ar` = ARVALID∧ARREADY, `obs_r` = RVALID∧RREADY; both cache `cache_length` signals; `obs_kvhead`, `obs_cache_read`, `obs_cache_write` |
| Shared compute | `obs_qhead`, `obs_core_active`; core `u_fsm.state`, `tile_col`, shared-dequant `batch_busy`, row-0 softmax `exp_out_valid`, `fused_update_en`, `norm_en`; compute zoom adds array and dequant retire/publication observations |
| Result / output | `obs_compute_done`, `obs_pack_start`, AXI AW/W/B signals; completion zoom includes all VALID/READY pairs, AWADDR/AWLEN, WLAST/BRESP, `u_job.source_seen`, `obs_wb_done`, `obs_wb_busy`, `obs_output_committed`, `irq` |

`Q_DMA_req` is a descriptor request; the AR/R observations are accepted external
transfers. `KV_rd_accept` is internal resident-cache activity, not AXI traffic.
Accepted Q addresses are `0x1000/0x2000/0x3000/0x4000`; between requests the displayed
ARADDR can advance to the end address (`base+16`) while ARVALID is low.
The existing scoreboard independently checks returned committed K/V stripes.
`KV_base` is the token base **0 or 16**, not a 0/1 tile ordinal.

Viewer [JobPhase.txt](JobPhase.txt) and [TilePhase.txt](TilePhase.txt) map only real
RTL state encodings. In the overview, short tile states remain numeric for
readability. In the compute zoom, `SM` is state 4, `S_UPDATE_SOFTMAX`; this state
also waits for shared dequantization. `dq_busy` distinguishes that wait. The
displayed softmax pulse is the used unnormalized exponential-valid output;
normalized-softmax output is disabled in this datapath. `DONE` is the SoC's
`output_committed`, confirmed coincident with host phase 4 and completion IRQ.

## Reproduce locally

The exact executed build/binary argument vectors and all production-source hashes
are in [provenance.json](provenance.json). The original regression record is in
[regression_results.json](regression_results.json); the unmodified simulator
output is [simulation.log](simulation.log). Screenshot and viewer hashes are in
[image_provenance.json](image_provenance.json).

Actual top-level commands used from the repository root:

```sh
/tmp/flashattn-wave-env/bin/python tools/v2/verify.py --output-dir /tmp/flashattn-wave/regression --dim 16 --lanes 16 --compact 1 --scope focus --latencies 0
/tmp/flashattn-wave-env/bin/python tools/v2/waveform/capture.py --regression /tmp/flashattn-wave/regression --output /tmp/flashattn-wave/portfolio
/tmp/flashattn-wave-env/bin/python tools/v2/waveform/check_trace.py /tmp/flashattn-wave/portfolio/decode.vcd --output /tmp/flashattn-wave/portfolio/events.json
/tmp/flashattn-wave-env/bin/python tools/v2/waveform/layouts.py /tmp/flashattn-wave/portfolio/decode.vcd /tmp/flashattn-wave/portfolio/events.json --output docs/v2/waveform
/tmp/flashattn-wave-env/bin/python tools/v2/waveform/prepare_figures.py /tmp/flashattn-wave/portfolio --output docs/v2/figures
```

The exact simulation command was:

```sh
/private/tmp/flashattn-wave/portfolio/obj/sim_soc /private/tmp/flashattn-wave/regression/fixtures 0 focus '' 0 0
```

For a fresh run, use unique temporary paths; existing output directories/files
are deliberately refused by verification, capture and image preparation.

```sh
FLASHATTN_REPO="$PWD"
FLASHATTN_WAVE_TMP=$(mktemp -d /tmp/flashattn-decode-wave.XXXXXX)
python3 -m venv "$FLASHATTN_WAVE_TMP/venv"
"$FLASHATTN_WAVE_TMP/venv/bin/python" -m pip install numpy pillow
"$FLASHATTN_WAVE_TMP/venv/bin/python" tools/v2/verify.py --output-dir "$FLASHATTN_WAVE_TMP/regression" --dim 16 --lanes 16 --compact 1 --scope focus --latencies 0
"$FLASHATTN_WAVE_TMP/venv/bin/python" tools/v2/waveform/capture.py --regression "$FLASHATTN_WAVE_TMP/regression" --output "$FLASHATTN_WAVE_TMP/capture"
"$FLASHATTN_WAVE_TMP/venv/bin/python" tools/v2/waveform/check_trace.py "$FLASHATTN_WAVE_TMP/capture/decode.vcd" --output "$FLASHATTN_WAVE_TMP/capture/events.json"
"$FLASHATTN_WAVE_TMP/venv/bin/python" tools/v2/waveform/layouts.py "$FLASHATTN_WAVE_TMP/capture/decode.vcd" "$FLASHATTN_WAVE_TMP/capture/events.json" --output "$FLASHATTN_WAVE_TMP/layouts"
mkdir -p "$FLASHATTN_WAVE_TMP/viewer/.surfer/mappings"
cp docs/v2/waveform/surfer-config.toml "$FLASHATTN_WAVE_TMP/viewer/.surfer/config.toml"
cp docs/v2/waveform/JobPhase.txt docs/v2/waveform/TilePhase.txt "$FLASHATTN_WAVE_TMP/viewer/.surfer/mappings/"
cd "$FLASHATTN_WAVE_TMP/viewer"
surfer "$FLASHATTN_WAVE_TMP/capture/decode.vcd" --command-file "$FLASHATTN_WAVE_TMP/layouts/overview.sucl"
```

Use `compute.sucl` or `completion.sucl` on a fresh viewer instance for the zooms.
The actual capture used the same command with the task-local native Surfer binary
in `/tmp/flashattn-wave/FlashAttention Waveform.app/Contents/MacOS/FlashAttentionWaveform`
and working directory `/tmp/flashattn-wave/viewer-work`. The app bundle was a local
wrapper around the unmodified official executable to keep this task's windows
separate from other viewers; no global viewer configuration was changed.

To inspect the delivered trace without rebuilding:

```sh
gzip -dc "$FLASHATTN_REPO/docs/v2/waveform/decode.vcd.gz" > "$FLASHATTN_WAVE_TMP/decode.vcd"
surfer "$FLASHATTN_WAVE_TMP/decode.vcd" --command-file "$FLASHATTN_REPO/docs/v2/waveform/overview.sucl"
```

Capture each native viewer window as PNG, then run `prepare_figures.py` into a new
output directory. The script's crop bounds correspond to these 2000-pixel-wide
macOS captures; inspect new captures before applying them. [Surfer's official
distribution and documentation](https://surfer-project.org/) provide the viewer.

## Provenance and limitations

The bounded VCD is 2,730,265 bytes (gzip supplied), starting just before host command
acceptance and ending after 64 cycles of DONE retention. VCD was practical here
because the small trace permits direct, readable edge auditing; no FST conversion
or extra converter dependency was needed. Large arrays are excluded with
`--trace-max-array 1`; hierarchy depth is bounded at 5.

GTKWave was present but could not launch because its Homebrew `glib`/GTK libraries
were missing. The installed Python NumPy had an architecture mismatch; a temporary
virtual environment supplied native NumPy 2.5.4 and Pillow 12.3.0. Surfer was
downloaded from its official native macOS build into the temporary directory.
All final PNGs were visually inspected. Short FSM segments and one-cycle pulses
compress in the overview; the two zooms and actual trace provide detail.

## Publication contents

The three final PNGs, bounded compressed trace, reproduction utilities, viewer
configuration, signal mappings and compact evidence records are retained.
[files.json](files.json) classifies all 28 original artifacts and lists the
retained files with their hashes.

The superseded README insertion draft, raw screenshot ZIP, oracle console log and
two negative-control console logs are excluded from publication and preserved
locally pending cleanup approval. Their original hashes and reasons are recorded
in the manifest. The public regression record retains command exits and negative
control diagnostics; the simulator log retains all five healthy job results.
