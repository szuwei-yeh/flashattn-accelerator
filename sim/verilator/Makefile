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

lint:
	$(VERILATOR) --lint-only --sv -Wall \
	    $(RTL_DIR)/pe.sv \
	    $(RTL_DIR)/systolic_array.sv \
	    $(RTL_DIR)/array_controller.sv

clean:
	rm -rf $(SIM_DIR)
