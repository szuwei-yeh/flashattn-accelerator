# Synthesis flow

這個目錄只保存可重現的 Synopsys Design Compiler 輸入與 server 執行工具。每次
synthesis 的 log、report、DDC、netlist 與 SDC 都放在獨立的 run-tag 目錄，不再共用
`syn/logs`、`syn/reports`、`syn/output`，避免新舊結果互相覆蓋或混在一起。

本機不需要安裝 Synopsys。預期工作流程是：

```text
本機修改 RTL／驗證
        ↓
git commit + push
        ↓
server git pull／clone
        ↓
server 執行 syn/scripts/run_server.sh
        ↓
rsync 指定的 syn/runs/<run-tag>/ 回本機
```

## 1. 目錄結構

```text
syn/
├── README.md
├── blackboxes/
│   ├── sram_1r1w_bb.sv
│   └── exp_lut_bb.sv
├── filelists/
│   ├── systolic_array.f
│   ├── dma_engine_vec.f
│   ├── banked_tile_loader.f
│   ├── core_banked_prefetch.f
│   ├── core_banked_prefetch_macro.f
│   ├── top_dma_banked_prefetch.f
│   └── top_dma_banked_prefetch_macro.f
├── scripts/
│   ├── dc_run.tcl
│   └── run_server.sh
└── runs/
    ├── .gitkeep
    └── <run-tag>/                 # generated，git-ignored
        └── <profile>/
            ├── manifest.txt
            ├── SUCCESS or FAILED
            ├── logs/dc_shell.log
            ├── reports/*.rpt
            ├── artifacts/*.{v,ddc,sdc}
            └── work/              # DC WORK library
```

### 會提交到 Git 的內容

- `syn/README.md`
- `syn/blackboxes/*.sv`
- `syn/filelists/*.f`
- `syn/scripts/*`
- `syn/runs/.gitkeep`

### 不提交的內容

`syn/runs/*` 是 server 產生物，已由 `.gitignore` 排除。結果以 run tag 為單位透過
`rsync`/`scp` 同步，避免把大型 DDC、mapped netlist 與 WORK database 推進 Git。

## 2. Profiles

`run_server.sh` 提供以下 profiles：

| Profile | Top module | Memory model | 建議用途 |
|---------|------------|--------------|----------|
| `systolic` | `systolic_array` | 無 memory | 最快的 DC/library sanity check |
| `dma` | `dma_engine_vec` | 無 memory | AXI read DMA 邏輯合成 |
| `loader` | `banked_tile_loader` | 無內部 SRAM | Banked tile loader 合成 |
| `core` | `flash_attn_core_banked_prefetch` | SRAM/ROM logical blackbox | **目前 full-core 首選** |
| `core_rtl` | 同上 | Behavioral SRAM/ROM | 只建議 elab/debug，不建議完整 compile |
| `top` | `flash_attn_top_dma_banked_prefetch` | SRAM/ROM logical blackbox | DMA + full core |
| `top_rtl` | 同上 | Behavioral SRAM/ROM | 只建議 elab/debug |

`core` 和 `top` 使用 `*_macro.f` filelist，以 synthesis-only stub 取代：

- `rtl/memory/sram_1r1w.sv`
- `rtl/softmax/exp_lut.sv`

這能避免 DC 把 behavioral memory 展開成大量 flops。注意 logical blackbox 沒有真實 memory
compiler `.db` 的 area、timing、power，因此 full-core 數字仍不是 signoff PPA。

## 3. 本機準備與 push

先完成 RTL regression，確認要送去合成的是已知正確的版本：

```bash
cd sim/verilator
make -j4 regression
cd ../..
```

確認 synthesis scaffold 和 RTL 都會被提交：

```bash
git status --short
git check-ignore -v syn/README.md
```

`git check-ignore` 對 `syn/README.md` 應該沒有輸出。接著由使用者自行 review、commit、push；
不要只 push synthesis scripts 而漏掉同一版 RTL。

## 4. Server 環境

在 server clone 或更新同一個 commit：

```bash
git clone <remote-url> flashattn-accelerator
cd flashattn-accelerator
```

若已經 clone：

```bash
git fetch
git checkout <branch>
git pull --ff-only
```

設定 server 上實際存在的 standard-cell `.db`：

```bash
export DC_TARGET_LIBRARY=/absolute/path/to/standard_cell.db
```

舊環境使用過的 FreePDK45 路徑是：

```text
/fs/ece/PDKs/bongjin/NCSU_FreePDK45/osu_soc/lib/files/gscl45nm.db
```

不要把 machine-specific library path 寫回 Tcl；`run_server.sh` 會從環境變數傳入。若
`dc_shell` 不在 PATH，可設定：

```bash
export DC_SHELL_BIN=/absolute/path/to/dc_shell
```

## 5. 建議執行順序

### 5.1 快速 tool/library sanity check

```bash
syn/scripts/run_server.sh systolic 2026-09-02_sanity compile_ultra
```

### 5.2 目前 fused RTL 的 full-core elaboration

先跑只需數分鐘等級的 front-end/elaboration，確認新 RTL、blackbox 與 hierarchy 可被 DC 接受：

```bash
syn/scripts/run_server.sh core 2026-09-02_fused_elab elab
```

應產生：

```text
syn/runs/2026-09-02_fused_elab/core/
├── SUCCESS
├── manifest.txt
├── logs/dc_shell.log
├── reports/check_design.rpt
├── reports/hierarchy.rpt
├── reports/reference.rpt
└── artifacts/elaborated.ddc
```

### 5.3 Full-core compile

Elaboration clean 後，再跑完整 compile：

```bash
syn/scripts/run_server.sh core 2026-09-02_fused_d16 compile
```

預設參數使用 RTL module default（目前是 d16 profile）。若要明確跑 d64：

```bash
ELAB_PARAMETERS='HEAD_DIM=64,SEQ_LEN=64' \
  syn/scripts/run_server.sh core 2026-09-02_fused_d64 compile
```

每個 tag 只能使用一次；若同名 run directory 已存在，wrapper 會拒絕覆蓋。重跑時請換 tag，
例如追加 `_retry1`。

### 5.4 DMA + core top

Full core 結果合理後，才建議跑更大的 top：

```bash
syn/scripts/run_server.sh top 2026-09-02_fused_dma_top compile
```

## 6. Run modes

| Mode | 行為 | 使用時機 |
|------|------|----------|
| `elab` | analyze → elaborate → link → check_design | 最快的語法/hierarchy/blackbox bring-up |
| `compile` | 上述流程 + `compile` + PPA reports | Full core 首輪，較容易 debug |
| `compile_ultra` | 上述流程 + `compile_ultra` | 純邏輯小 block 或後續較積極最佳化 |

共同 constraints：

- Clock port：`clk`
- Clock period：預設 10.0 ns，可用 `CLK_PERIOD` 覆寫
- IO delay：預設 1.0 ns，可用 `IO_DELAY` 覆寫
- `rst_n`：async reset，設為 ideal network 並從 input timing 排除

範例：

```bash
CLK_PERIOD=20.0 IO_DELAY=2.0 \
  syn/scripts/run_server.sh core 2026-09-02_fused_20ns compile
```

## 7. 每次 run 的可追溯資訊

`manifest.txt` 會記錄：

- run tag 與 profile
- top module 與 filelist
- `elab`/`compile`/`compile_ultra`
- clock 與 IO delay
- target library path
- Git commit SHA
- worktree 是否 dirty
- 開始與完成時間

建議只在 server 的 clean worktree 執行。若 worktree dirty，wrapper 會顯示 warning，並在
manifest 中記錄 `git_dirty=yes`；結果仍會跑，但不應當成正式可重現基準。

完成標記：

- `SUCCESS`：`dc_shell` exit status 為 0。
- `FAILED`：DC 或 flow 回傳非零；先看 `logs/dc_shell.log`。

即使有 `SUCCESS`，仍必須檢查 timing 與 constraints；SUCCESS 代表 flow 完成，不代表 timing met。

## 8. Report 與 artifact

Compile run 會產生：

```text
reports/
├── check_design.rpt
├── hierarchy.rpt
├── reference.rpt
├── timing_setup.rpt
├── constraint_violators.rpt
├── area.rpt
├── power.rpt
├── qor.rpt
├── resources.rpt
└── clocks.rpt

artifacts/
├── mapped.v
├── mapped.ddc
└── constraints.sdc
```

優先閱讀順序：

1. `logs/dc_shell.log`：搜尋 `Error:`、`Warning:`、unresolved reference。
2. `reports/check_design.rpt`：確認沒有結構性 error。
3. `reports/reference.rpt`：確認 SRAM/ROM 是預期的 blackbox，而不是大量 inferred flops。
4. `reports/qor.rpt`：WNS、TNS、cell count、area 摘要。
5. `reports/timing_setup.rpt`：critical path 起終點與組合邏輯。
6. `reports/constraint_violators.rpt`：setup/max-cap/max-transition 等違規。
7. `reports/area.rpt`、`power.rpt`：只在 library/macro/activity 假設清楚時解讀。

## 9. 從 server 同步回本機

在本機 repository root 執行，只抓指定 tag：

```bash
rsync -av \
  <user>@<server>:/absolute/path/to/flashattn-accelerator/syn/runs/<run-tag>/ \
  syn/runs/<run-tag>/
```

如果只想同步較小的 log/reports，不抓大型 DDC、WORK 與 mapped netlist：

```bash
mkdir -p syn/runs/<run-tag>
rsync -av \
  --exclude='artifacts/' \
  --exclude='work/' \
  <user>@<server>:/absolute/path/to/flashattn-accelerator/syn/runs/<run-tag>/ \
  syn/runs/<run-tag>/
```

同步完成後，先確認：

```bash
find syn/runs/<run-tag> -maxdepth 3 -type f | sort
```

由於 `syn/runs/*` 被 git-ignore，sync 回來不會污染 `git status`。

## 10. 本次最重要的 full-core 檢查

目前 RTL 新增 fused output update：

```text
SRAM read → rescale multiplier → 32-bit truncate → PV adder → SRAM write
```

相較舊版，多了一段 multiplier→adder 的單-cycle 組合串接。新的 synthesis run 應特別檢查：

1. `output_buffer` fused path 是否進入 top critical paths。
2. 舊版由 `online_softmax` normalization/division 主導的 critical path 是否仍然最差。
3. 10 ns 下的 WNS/TNS 與舊 run 比較。
4. `output_buffer` instance 的 area 增量。
5. SRAM/ROM 是否仍以 blackbox 存在。

Cycle simulation 已證明功能與週期數改善，但只有 DC/STA 能判斷 Fmax，因此 cycle speedup 不可直接
等同 wall-clock speedup。

## 11. 已知限制

- Logical SRAM/ROM blackbox 沒有真實 memory macro area、delay、power。
- 沒有 CTS、placement、routing、extracted parasitics；不是 signoff timing。
- `report_power` 沒有真實 switching activity 時，只能當 rough estimate。
- 完整 core 過去在 FreePDK45、10 ns、logical macro flow 下 timing 不收斂；舊 run 的主要 critical
  path 在 softmax normalization，不能假設新版本會自動改善。
- `core_rtl`/`top_rtl` 的 behavioral memory compile 可能極慢且產生不實際的 flop area，僅保留作
  front-end debug 對照。

## 12. 舊結果

2026-07-03 的舊 FreePDK45/10 ns 結果在本機整理至：

```text
syn/runs/2026-07-03_freepdk45_10ns/legacy/
```

這批資料使用舊 RTL 與舊的固定輸出結構，只供歷史比較，不應和新的 fused run 混用。因為
`syn/runs/*` 不進 Git，乾淨 clone 不會包含舊結果。
