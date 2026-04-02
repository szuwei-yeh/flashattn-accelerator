# FlashAttention Accelerator — Progress Log

---

## Week 1 — Completed ✅

### 完成項目

| 項目 | 檔案 | 狀態 |
|------|------|------|
| Golden model（FP32 tiled attention） | `golden/flash_attention.py` | ✅ PASS |
| Golden model（INT8 quantised attention） | `golden/flash_attention.py` | ⚠️ WARN（8.45% rel err，Week 2 再修） |
| Processing Element（INT8×INT8→INT32） | `rtl/systolic/pe.sv` | ✅ Lint clean |
| 16×16 Systolic Array（flat 1D port） | `rtl/systolic/systolic_array.sv` | ✅ Lint clean |
| Array Controller（skew FSM + shift reg） | `rtl/systolic/array_controller.sv` | ✅ Lint clean |
| Verilator C++ testbench（3 test cases） | `sim/verilator/sim_main.cpp` | ✅ ALL PASS |

### 測試結果

```
[golden/flash_attention.py]
✅  standard vs flash_tiled   — max error: 2.98e-07  (PASS)
✅  partial-tile test (N=70)  — max error: 4.17e-07  (PASS)
✅  INT8 quantised            — mean rel error: 8.45%  (WARN — > 1%)

[sim/verilator — SIZE=4]
Test1: I × all-2  => all-2    PASS  (16/16 elements)
Test2: all-1 × all-1 => all-4  PASS  (16/16 elements)
Test3: A[r][c]=r+1, B[r][c]=c+1 => SIZE*(r+1)*(c+1)  PASS  (16/16 elements)
```

---

## 遇到的坑與解法

### 1. Icarus Verilog 不支援多維 unpacked array port
**問題**：`array_controller.sv` 的 `acc` port 宣告為 `[SIZE-1:0][SIZE-1:0]`（2D），
Icarus 對此輸出全 `x`，且連接到 systolic_array 的 1D flat port 時 Verilator 也報錯：
```
Slices of arrays in assignments have different unpacked dimensions, 16 versus 256
```
**解法**：全部改成 1D flat port：`acc [SIZE*SIZE-1:0]`，
存取時用 `acc[r*SIZE + c]`。所有模組統一使用此慣例。

### 2. Verilator WIDTHEXPAND warnings
**問題**：`feed_col`（5-bit）和 `cycle_cnt`（6-bit）和 32-bit literal 做比較時，
Verilator 5.x 以 `-Wall` 會警告位元寬自動擴展。
**解法**：加明確 cast：`5'(SIZE-1)`、`6'(COMPUTE_CYCLES-1)`、`int'(feed_col)`。

### 3. INT8 golden model 誤差偏大（8.45% vs 目標 < 1%）
**問題**：`int8_attention()` 使用 per-tensor symmetric quantisation，
對 V 的 dequantise 路徑不完整（`P × V_i32 * v_scale` 缺乏 proper rescale）。
**狀態**：WARN 而非 FAIL，Week 2 在實作 RTL datapath 時一併修正。

### 4. Systolic array reset 前必須拉低 en
**問題**：testbench 在 reset 期間若 `en=1`，reset 拉起後第一個 tick 的 clear
可能被 PE 直接累加進去，造成非預期的初始值。
**解法**：reset 期間保持 `en=0`，`rst_n=1` 後才 assert `en=1` 並做 1-cycle clear。

---

## 檔案說明

```
flashattn-accelerator/
├── golden/
│   ├── flash_attention.py        FP32 tiled attention + INT8 quantised golden model
│   └── generate_test_vectors.py  產生 RTL 驗證用的測試向量（Week 2 用）
│
├── rtl/systolic/
│   ├── pe.sv                     單一 PE：INT8×INT8 MAC，sync reset，clear，en
│   ├── systolic_array.sv         SIZE×SIZE PE 陣列，flat 1D wire/port，generate 展開
│   └── array_controller.sv       輸入 skew FSM（IDLE/CLEAR/COMPUTE/FINISH）+
│                                 shift register，驅動 systolic_array
│
├── sim/verilator/
│   ├── sim_main.cpp              C++ testbench：3 test cases，手動 skewed input
│   └── Makefile                  verilator --sv -cc --exe --build -GSIZE=4
│
├── tb/                           （Week 2 擴充，現為空）
├── README.md                     架構總覽
└── PROGRESS.md                   本檔案
```

---

## Week 2 預計工作

### 主要目標：INT8 Quantization + Online Softmax

1. **修正 INT8 golden model**
   - 修正 `int8_attention()` dequantise 路徑，達到 < 1% rel error
   - 加入 per-row / per-tensor scale 比較實驗

2. **Online Softmax 模組（RTL）**
   - 實作 `online_softmax.sv`：running max（`m`）+ running sum（`l`）+ output rescale
   - 對應 FlashAttention Algorithm 1 的 tile-level 狀態更新
   - 介面：接收 systolic array 輸出的一行 INT32 scores

3. **Tile Controller FSM**
   - 協調 Q/K/V tile 的讀取順序
   - 銜接 array_controller → online_softmax → output accumulator

4. **測試向量整合**
   - `generate_test_vectors.py` 產生 Q, K, V 的 INT8 hex dump
   - Verilator testbench 讀入向量，對照 golden model 輸出驗證

5. **Verilator lint + regression**
   - 全部新模組都要先 lint clean 再寫 testbench
