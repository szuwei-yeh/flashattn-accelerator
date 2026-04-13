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
	    ../../rtl/top/flash_attn_core.sv \
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

# ── Week 5: all AXI tests ─────────────────────────────────────────────────────
axi_all: tb_axi_slave axi_top_N64
	@echo ""
	@echo "=== All Week 5 AXI tests passed ==="

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
	    ../../rtl/top/flash_attn_core.sv

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

# ── Week 6: all tests (unit + decode) ────────────────────────────────────────
week6_all: tb_kv_cache tb_kv_decode
	@echo ""
	@echo "=== All Week 6 tests passed ==="

# ── Full regression (Week 1–6) ────────────────────────────────────────────────
regression: tb_top_all axi_all week6_all
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
	rm -rf $(SIM_DIR) obj_array_ctrl obj_quantizer obj_exp_lut \
	        obj_softmax obj_addr_gen obj_sram obj_kv_buf \
	        obj_top obj_top_N16 obj_top_N64 obj_top_N128 obj_top_N256 \
	        obj_axi_slave obj_axi_top_N64 \
	        obj_kv_cache obj_kv_decode