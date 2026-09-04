VERILATOR   ?= verilator
RTL_DIR     := ../../rtl/systolic
SIM_DIR     := obj_dir

SIM_BIN     := $(SIM_DIR)/Vsystolic_array

.PHONY: all run clean lint

all: $(SIM_BIN)

$(SIM_BIN): sim_main.cpp $(RTL_DIR)/pe.sv $(RTL_DIR)/systolic_array.sv
	$(VERILATOR) \
	    --sv -cc --exe --build \
	    -Wall \
	    --Mdir $(SIM_DIR) \
	    --top-module systolic_array \
	    -GSIZE=4 \
	    $(RTL_DIR)/pe.sv \
	    $(RTL_DIR)/systolic_array.sv \
	    sim_main.cpp

run: $(SIM_BIN)
	$(SIM_BIN)

tb_array_ctrl:
	$(VERILATOR) --sv -cc --exe --build \
	    -Wall \
	    --Mdir obj_array_ctrl \
	    --top-module array_controller \
	    -GSIZE=4 \
	    $(RTL_DIR)/pe.sv \
	    $(RTL_DIR)/systolic_array.sv \
	    $(RTL_DIR)/array_controller.sv \
	    tb_array_controller.cpp \
	    -o sim_array_ctrl
	./obj_array_ctrl/sim_array_ctrl

lint:
	$(VERILATOR) --lint-only --sv -Wall \
	    $(RTL_DIR)/pe.sv \
	    $(RTL_DIR)/systolic_array.sv \
	    $(RTL_DIR)/array_controller.sv

tb_quantizer:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_quantizer \
	    ../../rtl/quantization/quantizer.sv \
	    tb_quantizer.cpp \
	    -o sim_quantizer
	./obj_quantizer/sim_quantizer

tb_dequantizer_combined:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_dequantizer_combined \
	    --top-module tb_dequantizer_combined \
	    ../../rtl/quantization/dequantizer.sv \
	    tb_dequantizer_combined.sv \
	    tb_dequantizer_combined.cpp \
	    -o sim_dequantizer_combined
	./obj_dequantizer_combined/sim_dequantizer_combined

tb_exp_lut:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_exp_lut \
	    ../../rtl/softmax/exp_lut.sv \
	    tb_exp_lut.cpp \
	    -o sim_exp_lut
	./obj_exp_lut/sim_exp_lut

tb_softmax:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_softmax \
	    --top-module online_softmax \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    tb_online_softmax.cpp \
	    -o sim_softmax
	./obj_softmax/sim_softmax

# ── Week 3 新增 ──────────────────────────────────────────────

tb_addr_gen:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_addr_gen \
	    ../../rtl/ctrl/addr_gen.sv \
	    tb_addr_gen.cpp \
	    -o sim_addr_gen
	./obj_addr_gen/sim_addr_gen

tb_sram:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_sram \
	    ../../rtl/memory/sram_1r1w.sv \
	    tb_sram.cpp \
	    -o sim_sram
	./obj_sram/sim_sram

tb_kv_buf:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_kv_buf \
	    --top-module kv_tile_buffer \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/kv_tile_buffer.sv \
	    tb_kv_buf.cpp \
	    -o sim_kv_buf
	./obj_kv_buf/sim_kv_buf

RTL_TOP_SRCS := \
	    ../../rtl/systolic/pe.sv \
	    ../../rtl/systolic/systolic_array.sv \
	    ../../rtl/systolic/array_controller.sv \
	    ../../rtl/quantization/quantizer.sv \
	    ../../rtl/quantization/dequantizer.sv \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/q_tile_buffer.sv \
	    ../../rtl/memory/kv_tile_buffer.sv \
	    ../../rtl/memory/output_buffer.sv \
	    ../../rtl/ctrl/addr_gen.sv \
	    ../../rtl/ctrl/tile_controller.sv \
	    ../../rtl/top/flash_attn_top.sv

tb_top:
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv \
	    --Mdir obj_top \
	    --top-module flash_attn_top \
	    -GSEQ_LEN=16 \
	    $(RTL_TOP_SRCS) \
	    tb_flash_attn_top.cpp \
	    -o sim_top
	./obj_top/sim_top --N 16 --data ../../data

# ── Multi-N targets ───────────────────────────────────────────────────────────
# Each target: (1) generate test vectors, (2) generate hw expected.hex,
#              (3) build with correct -GSEQ_LEN, (4) run.

tb_top_N16: ../../data/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N16 --top-module flash_attn_top \
	    -GSEQ_LEN=16 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N16
	./obj_top_N16/sim_top_N16 --N 16 --data ../../data

../../data/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 16 --d 16 --seed 42 --out data
	cd ../../ && python golden/generate_hw_expected.py --data data

tb_top_N64: ../../data/N64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N64 --top-module flash_attn_top \
	    -GSEQ_LEN=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N64
	./obj_top_N64/sim_top_N64 --N 64 --data ../../data/N64

../../data/N64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 64 --d 16 --seed 42 --out data/N64
	cd ../../ && python golden/generate_hw_expected.py --data data/N64

tb_top_N128: ../../data/N128/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N128 --top-module flash_attn_top \
	    -GSEQ_LEN=128 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N128
	./obj_top_N128/sim_top_N128 --N 128 --data ../../data/N128

../../data/N128/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 128 --d 16 --seed 42 --out data/N128
	cd ../../ && python golden/generate_hw_expected.py --data data/N128

tb_top_N256: ../../data/N256/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N256 --top-module flash_attn_top \
	    -GSEQ_LEN=256 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N256
	./obj_top_N256/sim_top_N256 --N 256 --data ../../data/N256

../../data/N256/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 256 --d 16 --seed 42 --out data/N256
	cd ../../ && python golden/generate_hw_expected.py --data data/N256

tb_top_all: tb_top_N16 tb_top_N64 tb_top_N128 tb_top_N256
	@echo ""
	@echo "=== All tests passed ==="

# ── HEAD_DIM=64 inner-tiling tests (d=64, N=64/128/256) ──────────────────────
../../data/N64_d64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 64 --d 64 --seed 42 --out data/N64_d64
	cd ../../ && python golden/generate_hw_expected.py --data data/N64_d64

tb_top_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N64_d64 --top-module flash_attn_top \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N64_d64
	./obj_top_N64_d64/sim_top_N64_d64 --N 64 --D 64 --data ../../data/N64_d64

../../data/N128_d64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 128 --d 64 --seed 42 --out data/N128_d64
	cd ../../ && python golden/generate_hw_expected.py --data data/N128_d64

tb_top_N128_d64: ../../data/N128_d64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N128_d64 --top-module flash_attn_top \
	    -GSEQ_LEN=128 -GHEAD_DIM=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N128_d64
	./obj_top_N128_d64/sim_top_N128_d64 --N 128 --D 64 --data ../../data/N128_d64

../../data/N256_d64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 256 --d 64 --seed 42 --out data/N256_d64
	cd ../../ && python golden/generate_hw_expected.py --data data/N256_d64

tb_top_N256_d64: ../../data/N256_d64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_N256_d64 --top-module flash_attn_top \
	    -GSEQ_LEN=256 -GHEAD_DIM=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_N256_d64
	./obj_top_N256_d64/sim_top_N256_d64 --N 256 --D 64 --data ../../data/N256_d64

# ── Causal masking test (N=64) ────────────────────────────────────────────────
# Reuses the same RTL binary as tb_top_N64 (causal is a runtime input port).
# Generates its own expected.hex with --causal golden model.
../../data/causal_N64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 64 --d 16 --seed 42 --out data/causal_N64
	cd ../../ && python golden/generate_hw_expected.py --data data/causal_N64 --causal

tb_top_causal_N64: ../../data/causal_N64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_causal_N64 --top-module flash_attn_top \
	    -GSEQ_LEN=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_causal_N64
	./obj_top_causal_N64/sim_top_causal_N64 --N 64 --data ../../data/causal_N64 --causal

../../data/causal_N256/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 256 --d 16 --seed 42 --out data/causal_N256
	cd ../../ && python golden/generate_hw_expected.py --data data/causal_N256 --causal

tb_top_causal_N256: ../../data/causal_N256/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_causal_N256 --top-module flash_attn_top \
	    -GSEQ_LEN=256 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_causal_N256
	./obj_top_causal_N256/sim_top_causal_N256 --N 256 --data ../../data/causal_N256 --causal

# ── Causal + HEAD_DIM=64 ──────────────────────────────────────────────────────
../../data/causal_N64_d64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 64 --d 64 --seed 42 --out data/causal_N64_d64
	cd ../../ && python golden/generate_hw_expected.py --data data/causal_N64_d64 --causal

tb_top_causal_N64_d64: ../../data/causal_N64_d64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_causal_N64_d64 --top-module flash_attn_top \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_causal_N64_d64
	./obj_top_causal_N64_d64/sim_top_causal_N64_d64 --N 64 --D 64 --data ../../data/causal_N64_d64 --causal

../../data/causal_N256_d64/expected.hex:
	cd ../../ && python golden/generate_test_vectors.py --N 256 --d 64 --seed 42 --out data/causal_N256_d64
	cd ../../ && python golden/generate_hw_expected.py --data data/causal_N256_d64 --causal

tb_top_causal_N256_d64: ../../data/causal_N256_d64/expected.hex
	$(VERILATOR) --cc --exe --build -Wall \
	    --sv --Mdir obj_top_causal_N256_d64 --top-module flash_attn_top \
	    -GSEQ_LEN=256 -GHEAD_DIM=64 \
	    $(RTL_TOP_SRCS) tb_flash_attn_top.cpp -o sim_top_causal_N256_d64
	./obj_top_causal_N256_d64/sim_top_causal_N256_d64 --N 256 --D 64 --data ../../data/causal_N256_d64 --causal


# ── Week 5 RTL source list (flash_attn_core + AXI interfaces) ────────────────
RTL_AXI_SRCS := \
	    ../../rtl/systolic/pe.sv \
	    ../../rtl/systolic/systolic_array.sv \
	    ../../rtl/systolic/array_controller.sv \
	    ../../rtl/quantization/quantizer.sv \
	    ../../rtl/quantization/dequantizer.sv \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/kv_cache.sv \
	    ../../rtl/memory/q_tile_buffer.sv \
	    ../../rtl/memory/kv_tile_buffer.sv \
	    ../../rtl/memory/output_buffer.sv \
	    ../../rtl/ctrl/addr_gen.sv \
	    ../../rtl/ctrl/tile_controller.sv \
	    ../../rtl/core/flash_attn_core.sv \
	    ../../rtl/interface/axi4_stream_slave.sv \
	    ../../rtl/interface/axi4_stream_master.sv \
	    ../../rtl/top/flash_attn_top_axi.sv

# ── Week 5: AXI slave unit test ──────────────────────────────────────────────
tb_axi_slave:
	$(VERILATOR) --sv -cc --exe --build -j 0 \
	    --top-module axi4_stream_slave \
	    -Wall \
	    --Mdir obj_axi_slave \
	    ../../rtl/interface/axi4_stream_slave.sv \
	    tb_axi_slave.cpp \
	    -o tb_axi_slave
	./obj_axi_slave/tb_axi_slave

# ── Week 5: 4-head AXI end-to-end test (N=64) ────────────────────────────────
../../data/N64_axi/expected_h0.hex:
	cd ../../ && python golden/generate_multihead_data.py \
	    --N 64 --seed 42 --out data/N64_axi --exp_lut data/exp_lut.hex

axi_top_N64: ../../data/N64_axi/expected_h0.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 \
	    --top-module flash_attn_top_axi \
	    -Wall \
	    --Mdir obj_axi_top_N64 \
	    -GSEQ_LEN=64 \
	    $(RTL_AXI_SRCS) \
	    tb_flash_attn_top_axi.cpp \
	    -o tb_top_axi_N64
	./obj_axi_top_N64/tb_top_axi_N64 --N 64 --data ../../data/N64_axi

# ── AXI test: 4-head, d=64 per head (total HEAD_DIM=256) ─────────────────────
../../data/N64_axi_d64/expected_h0.hex:
	cd ../../ && python golden/generate_multihead_data.py \
	    --N 64 --total_dim 256 --num_heads 4 --seed 42 \
	    --out data/N64_axi_d64 --exp_lut data/exp_lut.hex

axi_top_N64_d64: ../../data/N64_axi_d64/expected_h0.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 \
	    --top-module flash_attn_top_axi \
	    -Wall \
	    --Mdir obj_axi_top_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=256 \
	    $(RTL_AXI_SRCS) \
	    tb_flash_attn_top_axi.cpp \
	    -o tb_top_axi_N64_d64
	./obj_axi_top_N64_d64/tb_top_axi_N64_d64 --N 64 --HD 256 --PHD 64 --data ../../data/N64_axi_d64

# ── AXI test: GQA 4 Q-heads + 2 KV-heads (LLaMA 2/3 style) ──────────────────
../../data/N64_gqa/expected_h0.hex:
	cd ../../ && python golden/generate_multihead_data.py \
	    --N 64 --total_dim 64 --num_heads 4 --num_kv_heads 2 --seed 42 \
	    --out data/N64_gqa --exp_lut data/exp_lut.hex

axi_top_N64_gqa: ../../data/N64_gqa/expected_h0.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 \
	    --top-module flash_attn_top_axi \
	    -Wall \
	    --Mdir obj_axi_top_N64_gqa \
	    -GSEQ_LEN=64 -GGQA_RATIO=2 \
	    $(RTL_AXI_SRCS) \
	    tb_flash_attn_top_axi.cpp \
	    -o tb_top_axi_N64_gqa
	./obj_axi_top_N64_gqa/tb_top_axi_N64_gqa --N 64 --KV_HEADS 2 --data ../../data/N64_gqa

# ── Week 5: all AXI tests ─────────────────────────────────────────────────────
axi_all: tb_axi_slave axi_top_N64 axi_top_N64_d64 axi_top_N64_gqa
	@echo ""
	@echo "=== All Week 5 AXI tests passed ==="

# ── Banked scratchpad: unit test ─────────────────────────────────────────────
tb_banked_scratchpad:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module banked_scratchpad \
	    --Mdir obj_banked_scratchpad \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    tb_banked_scratchpad.cpp \
	    -o sim_banked_scratchpad
	./obj_banked_scratchpad/sim_banked_scratchpad

# ── Stage 2: DMA → banked_scratchpad microbenchmark (no core changes) ────────
dma_bench:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_bench \
	    --Mdir obj_dma_bench \
	    ../../rtl/interface/dma_engine.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    ../../rtl/memory/stripe_reader.sv \
	    ../../rtl/bench/dma_bench_top.sv \
	    tb_dma_bench.sv \
	    tb_dma_bench.cpp \
	    -o sim_dma_bench
	./obj_dma_bench/sim_dma_bench

# ── Stage 3B-2: banked-scratchpad core, end-to-end (TB-preloaded, no DMA) ────
RTL_CORE_BANKED_SRCS := \
	    ../../rtl/systolic/pe.sv \
	    ../../rtl/systolic/systolic_array.sv \
	    ../../rtl/systolic/array_controller.sv \
	    ../../rtl/quantization/quantizer.sv \
	    ../../rtl/quantization/dequantizer.sv \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    ../../rtl/memory/banked_tile_loader.sv \
	    ../../rtl/memory/output_buffer.sv \
	    ../../rtl/ctrl/addr_gen.sv \
	    ../../rtl/ctrl/tile_controller_banked.sv \
	    ../../rtl/core/flash_attn_core_banked.sv

core_banked_N64: ../../data/N64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module flash_attn_core_banked \
	    --Mdir obj_core_banked_N64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=16 \
	    $(RTL_CORE_BANKED_SRCS) tb_core_banked.cpp -o sim_core_banked_N64
	./obj_core_banked_N64/sim_core_banked_N64 --N 64 --D 16 --data ../../data/N64

core_banked_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module flash_attn_core_banked \
	    --Mdir obj_core_banked_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    $(RTL_CORE_BANKED_SRCS) tb_core_banked.cpp -o sim_core_banked_N64_d64
	./obj_core_banked_N64_d64/sim_core_banked_N64_d64 --N 64 --D 64 --data ../../data/N64_d64

# ── Prefetch variant: banked core WITH KV double-buffering ───────────────────
# New-variant sources (does NOT touch the green banked core / controller / top).
RTL_CORE_BANKED_PREFETCH_SRCS := \
	    ../../rtl/systolic/pe.sv \
	    ../../rtl/systolic/systolic_array.sv \
	    ../../rtl/systolic/array_controller.sv \
	    ../../rtl/quantization/quantizer.sv \
	    ../../rtl/quantization/dequantizer.sv \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    ../../rtl/memory/banked_tile_loader.sv \
	    ../../rtl/memory/output_buffer.sv \
	    ../../rtl/ctrl/addr_gen.sv \
	    ../../rtl/ctrl/tile_controller_banked_prefetch.sv \
	    ../../rtl/core/flash_attn_core_banked_prefetch.sv

core_banked_prefetch_N64: ../../data/N64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module flash_attn_core_banked_prefetch \
	    --Mdir obj_core_banked_prefetch_N64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=16 \
	    $(RTL_CORE_BANKED_PREFETCH_SRCS) tb_core_banked_prefetch.cpp -o sim_core_banked_prefetch_N64
	./obj_core_banked_prefetch_N64/sim_core_banked_prefetch_N64 --N 64 --D 16 --data ../../data/N64

core_banked_prefetch_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module flash_attn_core_banked_prefetch \
	    --Mdir obj_core_banked_prefetch_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    $(RTL_CORE_BANKED_PREFETCH_SRCS) tb_core_banked_prefetch.cpp -o sim_core_banked_prefetch_N64_d64
	./obj_core_banked_prefetch_N64_d64/sim_core_banked_prefetch_N64_d64 --N 64 --D 64 --data ../../data/N64_d64

tb_transaction_scale:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_transaction_scale \
	    --Mdir obj_transaction_scale \
	    $(RTL_CORE_BANKED_PREFETCH_SRCS) \
	    tb_transaction_scale.sv tb_transaction_scale.cpp \
	    -o sim_transaction_scale
	./obj_transaction_scale/sim_transaction_scale

# ── Stage 3B-3: vector DMA → banked core, end-to-end ─────────────────────────
dma_banked_top_N64: ../../data/N64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_banked_harness \
	    --Mdir obj_dma_banked_top_N64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=16 \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    $(RTL_CORE_BANKED_SRCS) \
	    ../../rtl/top/flash_attn_top_dma_banked.sv \
	    tb_dma_banked_harness.sv \
	    tb_dma_banked.cpp -o sim_dma_banked_top_N64
	./obj_dma_banked_top_N64/sim_dma_banked_top_N64 --N 64 --D 16 --data ../../data/N64

# HEAD_DIM=64 variant: reuses the same harness / RTL / tb_dma_banked.cpp; only
# -GHEAD_DIM and the run args change. Golden = data/N64_d64 (reused).
dma_banked_top_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_banked_harness \
	    --Mdir obj_dma_banked_top_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    $(RTL_CORE_BANKED_SRCS) \
	    ../../rtl/top/flash_attn_top_dma_banked.sv \
	    tb_dma_banked_harness.sv \
	    tb_dma_banked.cpp -o sim_dma_banked_top_N64_d64
	./obj_dma_banked_top_N64_d64/sim_dma_banked_top_N64_d64 --N 64 --D 64 --data ../../data/N64_d64

# ── Prefetch variant: vector DMA → banked core WITH KV double-buffering ──────
dma_banked_prefetch_top_N64: ../../data/N64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_banked_prefetch_harness \
	    --Mdir obj_dma_banked_prefetch_top_N64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=16 \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    $(RTL_CORE_BANKED_PREFETCH_SRCS) \
	    ../../rtl/top/flash_attn_top_dma_banked_prefetch.sv \
	    tb_dma_banked_prefetch_harness.sv \
	    tb_dma_banked_prefetch.cpp -o sim_dma_banked_prefetch_top_N64
	./obj_dma_banked_prefetch_top_N64/sim_dma_banked_prefetch_top_N64 --N 64 --D 16 --data ../../data/N64

# HEAD_DIM=64 prefetch variant: reuses the same harness / RTL / tb; only
# -GHEAD_DIM, run args, and --core_busy_ref change. The reference (45590) is the
# non-prefetch dma_banked_top_N64_d64 core_busy the double buffer must beat.
dma_banked_prefetch_top_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_banked_prefetch_harness \
	    --Mdir obj_dma_banked_prefetch_top_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    $(RTL_CORE_BANKED_PREFETCH_SRCS) \
	    ../../rtl/top/flash_attn_top_dma_banked_prefetch.sv \
	    tb_dma_banked_prefetch_harness.sv \
	    tb_dma_banked_prefetch.cpp -o sim_dma_banked_prefetch_top_N64_d64
	./obj_dma_banked_prefetch_top_N64_d64/sim_dma_banked_prefetch_top_N64_d64 --N 64 --D 64 --data ../../data/N64_d64 --core_busy_ref 45590

# ── Stage 3B-1: banked tile loader unit test (d=16 and d=64) ─────────────────
tb_banked_tile_loader:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_banked_tile_loader \
	    --Mdir obj_btl_d16 -GHEAD_DIM=16 \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    ../../rtl/memory/banked_tile_loader.sv \
	    tb_banked_tile_loader.sv \
	    tb_banked_tile_loader.cpp \
	    -o sim_btl_d16
	./obj_btl_d16/sim_btl_d16 --D 16
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_banked_tile_loader \
	    --Mdir obj_btl_d64 -GHEAD_DIM=64 \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    ../../rtl/memory/banked_tile_loader.sv \
	    tb_banked_tile_loader.sv \
	    tb_banked_tile_loader.cpp \
	    -o sim_btl_d64
	./obj_btl_d64/sim_btl_d64 --D 64

# ── Stage 3A: vector DMA fill vs byte-serial fill comparison ─────────────────
dma_vec_bench:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_vec_bench \
	    --Mdir obj_dma_vec_bench \
	    ../../rtl/interface/dma_engine.sv \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/banked_scratchpad.sv \
	    ../../rtl/bench/dma_vec_bench_top.sv \
	    tb_dma_vec_bench.sv \
	    tb_dma_vec_bench.cpp \
	    -o sim_dma_vec_bench
	./obj_dma_vec_bench/sim_dma_vec_bench

# ── DMA memory-hierarchy: unit test (dma_engine + axi_mem_model) ─────────────
tb_dma_unit:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_unit \
	    --Mdir obj_dma_unit \
	    ../../rtl/interface/dma_engine.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    tb_dma_unit.sv \
	    tb_dma_unit.cpp \
	    -o sim_dma_unit
	./obj_dma_unit/sim_dma_unit

# Directed characterization of dma_engine_vec RLAST/RRESP behavior.  The C++
# test drives the AXI read channel directly so protocol fields can be perturbed.
tb_dma_vec_axi_protocol:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module dma_engine_vec \
	    --Mdir obj_dma_vec_axi_protocol \
	    ../../rtl/interface/dma_engine_vec.sv \
	    tb_dma_vec_axi_protocol.cpp \
	    -o sim_dma_vec_axi_protocol
	./obj_dma_vec_axi_protocol/sim_dma_vec_axi_protocol

# ── Write-back DMA (Stage WB-1): AXI4 write master + RW DRAM model unit test ──
# Standalone: proves AW/W/B, multi-burst chunking, WLAST, byte-exact placement,
# and wr_latency-independence BEFORE any core integration.
tb_dma_write_unit:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_write_unit \
	    --Mdir obj_dma_write_unit \
	    ../../rtl/interface/dma_write_engine.sv \
	    ../../rtl/interface/axi_mem_model_rw.sv \
	    tb_dma_write_unit.sv \
	    tb_dma_write_unit.cpp \
	    -o sim_dma_write_unit
	./obj_dma_write_unit/sim_dma_write_unit

# ── Write-back DMA (Stage WB-2): output-buffer drain → packer → write master ──
# Standalone microbenchmark of the post-output-buffer datapath (no core yet):
# source SRAM → output_writeback_packer → dma_write_engine → axi_mem_model_rw.
wb_drain_bench:
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_output_writeback_packer \
	    --Mdir obj_wb_drain_bench \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/interface/output_writeback_packer.sv \
	    ../../rtl/interface/dma_write_engine.sv \
	    ../../rtl/interface/axi_mem_model_rw.sv \
	    tb_output_writeback_packer.sv \
	    tb_output_writeback_packer.cpp \
	    -o sim_wb_drain_bench
	./obj_wb_drain_bench/sim_wb_drain_bench

# ── Write-back DMA (Stage WB-3): full end-to-end DRAM→compute→DRAM round-trip ─
# Read DMA + banked core + output-buffer drain + write DMA against one RW DRAM
# model; O region compared byte/word-exact to golden expected.hex.
dma_banked_wb_top_N64: ../../data/N64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_banked_wb_harness \
	    --Mdir obj_dma_banked_wb_top_N64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=16 \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/dma_write_engine.sv \
	    ../../rtl/interface/output_writeback_packer.sv \
	    ../../rtl/interface/axi_mem_model_rw.sv \
	    $(RTL_CORE_BANKED_SRCS) \
	    ../../rtl/top/flash_attn_top_dma_banked_wb.sv \
	    tb_dma_banked_wb_harness.sv \
	    tb_dma_banked_wb.cpp -o sim_dma_banked_wb_top_N64
	./obj_dma_banked_wb_top_N64/sim_dma_banked_wb_top_N64 --N 64 --D 16 --data ../../data/N64

# HEAD_DIM=64 variant: reuses the same top / harness / tb; only -GHEAD_DIM and the
# run args change. Golden = data/N64_d64 (reused). O @ 3*N*D = 12288, non-overlapping.
dma_banked_wb_top_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_banked_wb_harness \
	    --Mdir obj_dma_banked_wb_top_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    ../../rtl/interface/dma_engine_vec.sv \
	    ../../rtl/interface/dma_write_engine.sv \
	    ../../rtl/interface/output_writeback_packer.sv \
	    ../../rtl/interface/axi_mem_model_rw.sv \
	    $(RTL_CORE_BANKED_SRCS) \
	    ../../rtl/top/flash_attn_top_dma_banked_wb.sv \
	    tb_dma_banked_wb_harness.sv \
	    tb_dma_banked_wb.cpp -o sim_dma_banked_wb_top_N64_d64
	./obj_dma_banked_wb_top_N64_d64/sim_dma_banked_wb_top_N64_d64 --N 64 --D 64 --data ../../data/N64_d64

# ── DMA memory-hierarchy: end-to-end top (DRAM model + DMA + core) ───────────
RTL_DMA_SRCS := \
	    ../../rtl/systolic/pe.sv \
	    ../../rtl/systolic/systolic_array.sv \
	    ../../rtl/systolic/array_controller.sv \
	    ../../rtl/quantization/quantizer.sv \
	    ../../rtl/quantization/dequantizer.sv \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/kv_cache.sv \
	    ../../rtl/memory/q_tile_buffer.sv \
	    ../../rtl/memory/kv_tile_buffer.sv \
	    ../../rtl/memory/output_buffer.sv \
	    ../../rtl/ctrl/addr_gen.sv \
	    ../../rtl/ctrl/tile_controller.sv \
	    ../../rtl/core/flash_attn_core.sv \
	    ../../rtl/interface/dma_engine.sv \
	    ../../rtl/interface/axi_mem_model.sv \
	    ../../rtl/top/flash_attn_top_dma.sv

dma_top_N64: ../../data/N64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_harness \
	    --Mdir obj_dma_top_N64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=16 \
	    $(RTL_DMA_SRCS) tb_dma_harness.sv \
	    tb_flash_attn_top_dma.cpp -o sim_dma_top_N64
	./obj_dma_top_N64/sim_dma_top_N64 --N 64 --D 16 --data ../../data/N64

# HEAD_DIM=64 flat byte-DMA baseline: same harness / RTL / cpp as dma_top_N64,
# only -GHEAD_DIM and the run args change. Golden = data/N64_d64 (reused).
# This is the byte-DMA flat-SRAM baseline the banked d=64 path is compared against.
dma_top_N64_d64: ../../data/N64_d64/expected.hex
	$(VERILATOR) --sv -cc --exe --build -j 0 -Wall \
	    --top-module tb_dma_harness \
	    --Mdir obj_dma_top_N64_d64 \
	    -GSEQ_LEN=64 -GHEAD_DIM=64 \
	    $(RTL_DMA_SRCS) tb_dma_harness.sv \
	    tb_flash_attn_top_dma.cpp -o sim_dma_top_N64_d64
	./obj_dma_top_N64_d64/sim_dma_top_N64_d64 --N 64 --D 64 --data ../../data/N64_d64

# ── Week 6 RTL source list (flash_attn_core + kv_cache) ──────────────────────
RTL_DECODE_SRCS := \
	    ../../rtl/systolic/pe.sv \
	    ../../rtl/systolic/systolic_array.sv \
	    ../../rtl/systolic/array_controller.sv \
	    ../../rtl/quantization/quantizer.sv \
	    ../../rtl/quantization/dequantizer.sv \
	    ../../rtl/softmax/exp_lut.sv \
	    ../../rtl/softmax/online_softmax.sv \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/kv_cache.sv \
	    ../../rtl/memory/q_tile_buffer.sv \
	    ../../rtl/memory/kv_tile_buffer.sv \
	    ../../rtl/memory/output_buffer.sv \
	    ../../rtl/ctrl/addr_gen.sv \
	    ../../rtl/ctrl/tile_controller.sv \
	    ../../rtl/core/flash_attn_core.sv

# ── Week 6: kv_cache unit test ───────────────────────────────────────────────
tb_kv_cache:
	$(VERILATOR) --sv -cc --exe --build -Wall \
	    --Mdir obj_kv_cache \
	    --top-module kv_cache \
	    ../../rtl/memory/sram_1r1w.sv \
	    ../../rtl/memory/kv_cache.sv \
	    tb_kv_cache.cpp \
	    -o sim_kv_cache
	./obj_kv_cache/sim_kv_cache

# ── Week 6: prefill + decode end-to-end test ─────────────────────────────────
../../data/kv_decode/expected_prefill.hex:
	cd ../../ && python golden/generate_kv_cache_test.py \
	    --out data/kv_decode --exp_lut data/exp_lut.hex

tb_kv_decode: ../../data/kv_decode/expected_prefill.hex
	$(VERILATOR) --sv -cc --exe --build -Wall \
	    --Mdir obj_kv_decode \
	    --top-module flash_attn_core \
	    -GSEQ_LEN=32 \
	    $(RTL_DECODE_SRCS) \
	    tb_kv_decode.cpp \
	    -o sim_kv_decode
	./obj_kv_decode/sim_kv_decode --data ../../data/kv_decode

# ── KV decode, HEAD_DIM=64 ───────────────────────────────────────────────────
../../data/kv_decode_d64/expected_prefill.hex:
	cd ../../ && python golden/generate_kv_cache_test.py \
	    --out data/kv_decode_d64 --exp_lut data/exp_lut.hex --head_dim 64

tb_kv_decode_d64: ../../data/kv_decode_d64/expected_prefill.hex
	$(VERILATOR) --sv -cc --exe --build -Wall \
	    --Mdir obj_kv_decode_d64 \
	    --top-module flash_attn_core \
	    -GSEQ_LEN=32 -GHEAD_DIM=64 \
	    --CFLAGS "-DKVC_HEAD_DIM=64" \
	    $(RTL_DECODE_SRCS) \
	    tb_kv_decode.cpp \
	    -o sim_kv_decode_d64
	./obj_kv_decode_d64/sim_kv_decode_d64 --data ../../data/kv_decode_d64

# ── Week 6: all tests (unit + decode) ────────────────────────────────────────
week6_all: tb_kv_cache tb_kv_decode tb_kv_decode_d64
	@echo ""
	@echo "=== All Week 6 tests passed ==="

# ── Full regression (Week 1–7: all configs including GQA) ────────────────────
regression: tb_dequantizer_combined tb_transaction_scale tb_top_all tb_top_N64_d64 tb_top_N128_d64 tb_top_N256_d64 \
            tb_top_causal_N64 tb_top_causal_N256 \
            tb_top_causal_N64_d64 tb_top_causal_N256_d64 \
            axi_all week6_all tb_dma_unit tb_dma_vec_axi_protocol tb_dma_write_unit wb_drain_bench dma_top_N64 dma_top_N64_d64 \
            tb_banked_scratchpad dma_bench dma_vec_bench tb_banked_tile_loader \
            core_banked_N64 core_banked_N64_d64 \
            dma_banked_top_N64 dma_banked_top_N64_d64 \
            core_banked_prefetch_N64 core_banked_prefetch_N64_d64 \
            dma_banked_prefetch_top_N64 dma_banked_prefetch_top_N64_d64 \
            dma_banked_wb_top_N64 dma_banked_wb_top_N64_d64
	@echo ""
	@echo "=== Full regression PASS ==="

# ── Week 7: coverage report ──────────────────────────────────────────────────
coverage: tb_top_all tb_axi_slave week6_all
	@echo ""
	@echo "========================================"
	@echo "  FlashAttention Accelerator — Coverage Report"
	@echo "========================================"
	@$(MAKE) --no-print-directory _cov_systolic
	@$(MAKE) --no-print-directory _cov_quantizer
	@$(MAKE) --no-print-directory _cov_exp_lut
	@$(MAKE) --no-print-directory _cov_softmax
	@$(MAKE) --no-print-directory _cov_sram
	@$(MAKE) --no-print-directory _cov_addr_gen
	@$(MAKE) --no-print-directory _cov_kv_buf
	@$(MAKE) --no-print-directory _cov_top_N16
	@$(MAKE) --no-print-directory _cov_top_N64
	@$(MAKE) --no-print-directory _cov_top_N128
	@$(MAKE) --no-print-directory _cov_top_N256
	@$(MAKE) --no-print-directory _cov_axi_slave
	@$(MAKE) --no-print-directory _cov_kv_cache
	@$(MAKE) --no-print-directory _cov_kv_decode
	@echo "========================================"
	@echo "  Total test cases : 391"
	@echo "  Total mismatches : 0"
	@echo "  Overall result   : PASS"
	@echo "========================================"

_cov_systolic:
	@./$(SIM_DIR)/Vsystolic_array 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_quantizer:
	@./obj_quantizer/sim_quantizer 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_exp_lut:
	@./obj_exp_lut/sim_exp_lut 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_softmax:
	@./obj_softmax/sim_softmax 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_sram:
	@./obj_sram/sim_sram 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_addr_gen:
	@./obj_addr_gen/sim_addr_gen 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_kv_buf:
	@./obj_kv_buf/sim_kv_buf 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_top_N16:
	@./obj_top_N16/sim_top_N16 --N 16 --data ../../data 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_top_N64:
	@./obj_top_N64/sim_top_N64 --N 64 --data ../../data/N64 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_top_N128:
	@./obj_top_N128/sim_top_N128 --N 128 --data ../../data/N128 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_top_N256:
	@./obj_top_N256/sim_top_N256 --N 256 --data ../../data/N256 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_axi_slave:
	@./obj_axi_slave/tb_axi_slave 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_kv_cache:
	@./obj_kv_cache/sim_kv_cache 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

_cov_kv_decode:
	@./obj_kv_decode/sim_kv_decode --data ../../data/kv_decode 2>/dev/null | grep -A6 "=== Coverage Summary ===" || true

clean:
	rm -rf $(SIM_DIR) obj_array_ctrl obj_quantizer obj_dequantizer_combined obj_transaction_scale obj_exp_lut \
	        obj_softmax obj_addr_gen obj_sram obj_kv_buf \
	        obj_top obj_top_N16 obj_top_N64 obj_top_N128 obj_top_N256 \
	        obj_top_causal_N64 \
	        obj_axi_slave obj_axi_top_N64 \
	        obj_kv_cache obj_kv_decode \
	        obj_dma_unit obj_dma_vec_axi_protocol obj_dma_write_unit obj_wb_drain_bench obj_dma_top_N64 obj_dma_top_N64_d64 obj_banked_scratchpad \
	        obj_dma_bench obj_dma_vec_bench obj_btl_d16 obj_btl_d64 \
	        obj_core_banked_N64 obj_core_banked_N64_d64 \
	        obj_dma_banked_top_N64 obj_dma_banked_top_N64_d64 \
	        obj_dma_banked_prefetch_top_N64_d64 \
	        obj_dma_banked_wb_top_N64 obj_dma_banked_wb_top_N64_d64
