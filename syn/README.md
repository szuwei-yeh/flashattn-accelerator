# Synthesis handoff (`syn/`)

Scaffold **and results** for running **Synopsys Design Compiler**. The scaffold
(filelists, run scripts, blackboxes) is committed, and a **first-pass DC flow has
now been run** against it (R-2020.09-SP4, FreePDK45 `gscl45nm.db`, 10 ns / 100 MHz
target). Its generated output — mapped netlists, `.ddc`/`.sdc`, and
timing/area/power/QoR reports plus `dc_shell` logs — now lives under `output/`,
`reports/`, and `logs/`. Functional verification is still Verilator-only
(cycle-accurate sim); the DC results are **first-pass, non-signoff** (SRAM/ROM are
logical blackboxes — see §5). The generated files are **git-ignored**; only the
scaffold is tracked — see §7.

```
syn/
├── README.md                      ← this file (committed)
├── filelists/                     ← TOP + RTL path lists (committed)
│   ├── systolic_array.f           ← TOP = systolic_array
│   ├── core_banked_prefetch.f     ← TOP = flash_attn_core_banked_prefetch
│   └── top_dma_banked_prefetch.f  ← TOP = flash_attn_top_dma_banked_prefetch
├── blackboxes/                    ← logical SRAM/ROM stubs for DC (committed)
│   ├── sram_1r1w_bb.sv
│   └── exp_lut_bb.sv
├── scripts/                       ← DC template + per-design run scripts (committed)
│   ├── dc_synth_template.tcl      ← template (placeholders — retarget per library)
│   └── run_*.tcl                  ← concrete scripts that produced the results below
├── output/   (.gitkeep only)      ← generated .ddc / .sdc / *_mapped.v   (git-ignored)
├── reports/  (.gitkeep only)      ← generated *.rpt timing/area/power/QoR (git-ignored)
└── logs/     (.gitkeep only)      ← dc_shell *.log                       (git-ignored)
```

---

## 1. Getting the repo onto the server

The Verilator build artifacts (`sim/verilator/obj_*/`, `obj_dir_ctrl/`) are already
`.gitignore`d, so a clean clone carries only source. Two options:

**A. via git (preferred — clean by construction):**
```bash
# locally: commit the design files (see "what must be committed" below), then
git push <remote> <branch>
# on the server:
git clone <remote-url> flashattn-accelerator
cd flashattn-accelerator
```
> Note: as of this handoff, the banked/DMA/prefetch RTL is **untracked** in git
> (added across sessions but never committed). It MUST be committed before a push
> will carry it — see §4. This turn did not commit or push anything.

**B. via rsync (if you want to skip git):** exclude build junk explicitly:
```bash
rsync -av --exclude='.git' \
          --exclude='obj_*' --exclude='obj_dir*' \
          --exclude='__pycache__' --exclude='*.pyc' \
          ./flashattn-accelerator/  user@server:~/flashattn-accelerator/
```

---

## 2. On the server: sanity-check the files exist

```bash
cd flashattn-accelerator
# every RTL path referenced by the filelists must exist:
for f in syn/filelists/*.f; do
  echo "== $f =="
  grep -vE '^\s*#|^\s*$' "$f" | while read p; do
    [ -f "$p" ] && echo "  OK  $p" || echo "  MISSING  $p"
  done
done
# the exp_lut ROM contents (read by exp_lut.sv via $readmemh):
ls -l data/exp_lut.hex
```
All lines should print `OK`. If any print `MISSING`, the design files were not
carried over (most likely the untracked RTL was never committed — see §4).

Then edit `syn/scripts/dc_synth_template.tcl` and fill the four placeholders:
`target_library`, `link_library`, `TOP_MODULE`, `CLK_PERIOD` (plus `REPO_ROOT` if
`[info script]` is empty in your dc_shell invocation).

---

## 3. Recommended synthesis order

Synthesize bottom-up in complexity so problems are isolated early:

| # | TOP_MODULE | Filelist | Why first / notes |
|---|---|---|---|
| 1 | `systolic_array` | `systolic_array.f` | Smallest, cleanest: pure INT8 MAC array + registered accumulators. No memories, no `$readmemh`. Fastest sanity that the toolchain + library are wired up. |
| 2 | `flash_attn_core_banked_prefetch` | `core_banked_prefetch.f` | The full single-head banked prefetch core (array + softmax + dequant + output buffer + banked scratchpads + tile loader + prefetch FSM). This is the interesting PPA target. |
| 3 | `flash_attn_top_dma_banked_prefetch` | `top_dma_banked_prefetch.f` | Adds the vector DMA (AXI4 read-master) around the core. Largest; DRAM is an external boundary (see below). |

Example invocation (see the template header for the exact command):
```bash
dc_shell -x "set TOP_MODULE flash_attn_core_banked_prefetch; \
             set FILELIST syn/filelists/core_banked_prefetch.f; \
             set CLK_PERIOD 10.0" \
         -f syn/scripts/dc_synth_template.tcl | tee syn/logs/core_banked_prefetch.log
```

---

## 4. What must go to the server, and what must NOT

### Must be committed / rsync'd (design + data the filelists reference)
Synthesizable RTL used by the three filelists:
- `rtl/systolic/pe.sv`, `systolic_array.sv`, `array_controller.sv`
- `rtl/quantization/dequantizer.sv`
- `rtl/softmax/exp_lut.sv`, `online_softmax.sv`
- `rtl/memory/sram_1r1w.sv`, `banked_scratchpad.sv`, `banked_tile_loader.sv`, `output_buffer.sv`
- `rtl/ctrl/addr_gen.sv`, `tile_controller_banked_prefetch.sv`
- `rtl/top/flash_attn_core_banked_prefetch.sv`
- `rtl/interface/dma_engine_vec.sv`, `rtl/top/flash_attn_top_dma_banked_prefetch.sv`
- `data/exp_lut.hex` — **required**: `exp_lut.sv` loads it via `$readmemh` (ROM init).
- `syn/**` (this scaffold).

> Most of the banked/DMA/prefetch RTL above is currently **untracked** in git.
> `git add` + commit them (or use the rsync path) before the handoff, or the
> server clone will be missing them.

### Must NOT go into synthesis (simulation-only)
These may still live in the repo, but they are **excluded from every filelist**:
- `rtl/interface/axi_mem_model.sv` — behavioral DRAM model. The DMA's AXI4 read
  ports are the synthesis boundary; the external DRAM is not part of the DUT.
- `sim/verilator/*` — all C++ testbenches (`tb_*.cpp`) and SystemVerilog harnesses
  (`tb_*_harness.sv`, `tb_dma_*.sv`, `tb_banked_tile_loader.sv`, …).
- `sim/verilator/Makefile`, `sim/verilator/obj_*/` (Verilator build output).

### Must NOT be transferred (build artifacts / junk — already `.gitignore`d)
- `sim/verilator/obj_*/`, `obj_dir_ctrl/` (Verilator generated C++/objects)
- `golden/__pycache__/`, `*.pyc`

Also not needed for synthesis (but harmless if carried): the `golden/*.py` reference
models, `data/N*/` test vectors — these are for functional verification, not DC.

---

## 5. Synthesis notes / things to watch (from the RTL as written)

These are heads-ups for the person running DC. None require RTL changes to *try*
synthesis, but expect warnings and decide how to map memories.

- **Large inferred flop arrays / SRAM-like memories.** Several structures are
  behavioral arrays that DC will infer as flip-flops unless mapped to compiled
  macros:
  - `sram_1r1w.sv` — `logic [W-1:0] mem [0:DEPTH-1]` with synchronous R/W. As-is it
    infers a register file / flop array (large). For real PPA, replace with a
    memory-compiler macro (e.g. OpenRAM / vendor SRAM) and black-box it, or accept
    the flop-array area for a first estimate.
  - `banked_scratchpad.sv` = 16 × `sram_1r1w` (48 total across Q/K/V in the top).
  - `output_buffer.sv` also instances `sram_1r1w` (32-bit wide).
  - Tile registers `Q_reg/K_reg/V_reg` **and the prefetch `K_shadow/V_shadow`** in
    `flash_attn_core_banked_prefetch.sv` are explicit flop arrays — the shadow set
    roughly doubles the KV register area vs the non-prefetch core. Expect this.
- **Copy-style shadow→active swap mux.** On `kv_swap_banks`, the core copies all
  `KV_FLAT` bytes `K_reg <= K_shadow` (and V) in one cycle — a wide parallel mux +
  load enable on every KV register bit. It will show up as a large mux/enable cone.
  (A lower-area ping-pong SELECT swap is noted as future PPA work; this variant uses
  the copy style to stay behaviorally identical to the proven baseline.)
- **`exp_lut.sv` ROM via `$readmemh`.** `initial $readmemh("../../data/exp_lut.hex", …)`.
  Two things: (a) DC ROM inference from `$readmemh` varies — you may prefer to
  synthesize it as combinational logic (a case/LUT) or a compiled ROM; (b) the path
  is relative (`../../data/...`, written for the sim cwd). When running DC from the
  repo root the relative path won't resolve — copy `data/exp_lut.hex` next to the
  run dir, pass an absolute path, or convert the ROM to explicit logic. `data/exp_lut.hex`
  must be present regardless.
- **`quantizer.sv` is intentionally absent** from the banked filelists — the banked
  cores only instantiate `dequantizer`. Do not add it back.
- **Verilator pragmas.** The RTL carries `/* verilator lint_off … */` and
  `// synthesis translate_off/on` comments and some `$error` in simulation-only
  blocks. `verilator lint_*` are comments DC ignores. `synthesis translate_off/on`
  IS honored by DC (used around the `$error` assertions in `output_buffer.sv` /
  `dma_engine_vec.sv`), so those assertions are correctly excluded from synthesis.
  Watch the elaborate log for any unexpected warnings anyway.
- **Async active-low reset.** Every sequential block is
  `always_ff @(posedge clk or negedge rst_n)` — `rst_n` is an **asynchronous** reset.
  Constrain it as such (the template has commented `set_ideal_network` /
  `set_false_path -from [get_ports rst_n]` — enable/tune on the server) so DC does
  not try to meet setup/hold on the reset as if it were data.
- **Wide combinational arithmetic.** `dequantizer` (32×16×16 → 64-bit multiply +
  shift), `online_softmax` normalization / LUT-index logic, and `output_buffer`
  (signed divide in `norm_en`) are likely timing hot spots. Start with a relaxed
  `CLK_PERIOD` and tighten to find f_max.

---

## 6. Status of this scaffold (honest)

- A **first-pass DC flow has now been run** against this scaffold: mapped netlists,
  `.ddc`/`.sdc`, timing/area/power/QoR reports, and `dc_shell` logs exist under
  `output/`, `reports/`, and `logs/`. Results are **non-signoff** — SRAM/ROM are
  logical blackboxes (so `Macro/Black Box Area` is 0 and real memory area/timing
  are absent; see §5). The pure-logic blocks (`systolic_array`, `dma_engine_vec`,
  `banked_tile_loader`) met the 10 ns target; the full
  `flash_attn_core_banked_prefetch` macro-blackbox compile completed mapping but
  did **not** meet 10 ns (WNS −35.45 ns, softmax reduction path). Numbers are
  summarized in the top-level `README.md`.
- The filelists were derived from the **actual instantiation hierarchy** (verified
  by grep), not from the Verilator Makefile source lists (which also carry a few
  modules the banked cores don't instantiate).
- `scripts/dc_synth_template.tcl` still hard-codes **no** vendor library path or
  clock period — those are placeholders. The per-design `scripts/run_*.tcl` are the
  concrete scripts that produced the committed results; they point at the FreePDK45
  `gscl45nm.db` used for that run. Retarget them for another library/node.
- RTL/testbench behavior was **not** changed to run synthesis; the DUT is the same
  design verified under Verilator.

---

## 7. What to commit vs. leave uncommitted

**Commit — the scaffold (reproducible inputs):**
- `syn/README.md`
- `syn/filelists/*.f`
- `syn/scripts/*.tcl` (template + `run_*.tcl`)
- `syn/blackboxes/*.sv`
- the `.gitkeep` files under `output/`, `reports/`, `logs/`

**Do NOT commit — generated DC output (reproducible, large/noisy):**
- `syn/output/` — `*.ddc`, `*.sdc`, `*_mapped.v`
- `syn/reports/*.rpt`
- `syn/logs/*.log`

These are already covered by the repo `.gitignore` (`syn/{output,reports,logs}/*`
with `!.../.gitkeep` exceptions), so `git add syn/` picks up only the scaffold, and
the empty `output/`/`reports/`/`logs/` directories survive a clean clone via their
`.gitkeep`.
