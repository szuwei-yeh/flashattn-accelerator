# ============================================================
#  sim/icarus/Makefile
#  Week 1: PE + Systolic Array tests
#
#  Usage:
#    make test_pe        — compile & run PE unit test
#    make test_array     — compile & run systolic array test
#    make test_all       — run both
#    make wave_pe        — open PE waveform in GTKWave
#    make wave_array     — open array waveform in GTKWave
#    make clean          — remove build artifacts
# ============================================================

IVERILOG = iverilog
IVFLAGS  = -g2012 -Wall

ROOT     = ../..
RTL      = $(ROOT)/rtl
TB       = $(ROOT)/tb/basic

# ── RTL source lists ─────────────────────────────────────────
SYSTOLIC_RTL = \
    $(RTL)/systolic/pe.sv \
    $(RTL)/systolic/systolic_array.sv \
    $(RTL)/systolic/array_controller.sv

# ── Targets ──────────────────────────────────────────────────
.PHONY: test_pe test_array test_all wave_pe wave_array clean golden

test_all: test_pe test_array

## ── PE test ─────────────────────────────────────────────────
test_pe: tb_pe.out
	@echo "──────────────────────────────────────"
	@echo " Running PE unit test..."
	@echo "──────────────────────────────────────"
	./tb_pe.out

tb_pe.out: $(TB)/tb_pe.sv $(RTL)/systolic/pe.sv
	$(IVERILOG) $(IVFLAGS) -o $@ $^

## ── Systolic array test ─────────────────────────────────────
test_array: tb_array.out
	@echo "──────────────────────────────────────"
	@echo " Running Systolic Array test..."
	@echo "──────────────────────────────────────"
	./tb_array.out

tb_array.out: $(TB)/tb_systolic_array.sv $(SYSTOLIC_RTL)
	$(IVERILOG) $(IVFLAGS) -o $@ $^

## ── Waveform viewers ────────────────────────────────────────
wave_pe: dump_pe.vcd
	gtkwave dump_pe.vcd &

wave_array: dump_systolic.vcd
	gtkwave dump_systolic.vcd &

dump_pe.vcd: test_pe
dump_systolic.vcd: test_array

## ── Golden model (run from project root) ───────────────────
golden:
	cd $(ROOT) && python golden/flash_attention.py

## ── Clean ───────────────────────────────────────────────────
clean:
	rm -f *.out *.vcd
