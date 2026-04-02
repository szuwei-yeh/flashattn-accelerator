# FlashAttention Hardware Accelerator

> **Status: Week 1 in progress** — Golden Model + Systolic Array

---

## Architecture

```
AXI4-Stream IN (Q, K, V tiles)
         ↓
┌─────────────────────────────┐
│     Tile Controller FSM     │
└────────┬───────┬────────────┘
         ↓       ↓
┌───────────┐  ┌──────────────┐
│ SRAM Tile │  │  16×16 INT8  │
│  Buffers  │  │Systolic Array│
│  Q / K / V│  │  (MAC PEs)   │
└───────────┘  └──────┬───────┘
                       ↓
              ┌─────────────┐
              │Online Softmax│
              │(running max, │
              │ running sum) │
              └──────┬──────┘
                     ↓
              ┌─────────────┐
              │Output Accum │
              │  Buffer (O) │
              └──────┬──────┘
                     ↓
         AXI4-Stream OUT (attention output)
```

## Key Features (planned)

- Tiled online softmax — never stores full N×N matrix in SRAM
- 16×16 INT8 systolic array, 4 parallel attention heads
- KV Cache for autoregressive inference
- AXI4-Stream input/output interface

## Target Numbers

| Metric              | Target   |
|---------------------|----------|
| Systolic array size | 16×16    |
| Data type           | INT8 (INT32 accumulation) |
| Clock               | 200 MHz  |
| Attention heads     | 4 parallel |
| SRAM tile size      | 64×64    |
| UVM coverage        | >90%     |
| Accuracy vs PyTorch | <1%      |

## Results (fill in after synthesis)

| Metric            | Value |
|-------------------|-------|
| Clock             | — MHz |
| Latency (N=256)   | — cycles |
| LUT utilisation   | —% |
| BRAM utilisation  | —% |
| Accuracy          | —% |
| UVM coverage      | —% |

---

## How to Run

### Prerequisites

```bash
brew install icarus-verilog      # simulation
brew install --cask gtkwave      # waveform viewer (optional)
pip install torch numpy          # Python golden model
```

### Week 1: Golden Model

```bash
python golden/flash_attention.py
# Expected output:
#   ✅  standard vs flash_tiled   — max error: ...  (PASS)
#   ✅  partial-tile test (N=70)  — max error: ...  (PASS)
#   ✅  INT8 quantised            — mean rel error: ...%
```

### Week 1: RTL Simulation

```bash
cd sim/icarus

# PE unit test
make test_pe

# Systolic array test (16×16 matrix multiply)
make test_array

# Both at once
make test_all

# View waveforms
make wave_pe
make wave_array
```

### Synthesis (Vivado — school Linux server)

```bash
vivado -mode batch -source sim/vivado/run_synthesis.tcl
```

---

## File Structure

```
flashattn-accelerator/
├── rtl/
│   ├── systolic/
│   │   ├── pe.sv                ← Single MAC PE (INT8×INT8→INT32)
│   │   ├── systolic_array.sv    ← 16×16 PE array
│   │   └── array_controller.sv ← Skewed input feeding + FSM
│   └── ...
├── tb/
│   └── basic/
│       ├── tb_pe.sv             ← PE unit tests
│       └── tb_systolic_array.sv ← Matrix multiply verification
├── golden/
│   ├── flash_attention.py       ← Standard + tiled FlashAttention
│   └── generate_test_vectors.py ← Generate Q/K/V test data
├── sim/
│   └── icarus/
│       └── Makefile
└── README.md
```
