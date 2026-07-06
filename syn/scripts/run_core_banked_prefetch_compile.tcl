set_app_var search_path [list . ./rtl ./rtl/memory ./rtl/systolic ./rtl/softmax ./rtl/quantization ./rtl/ctrl ./rtl/top]
set_app_var target_library [list /fs/ece/PDKs/bongjin/NCSU_FreePDK45/osu_soc/lib/files/gscl45nm.db]
set_app_var link_library   [concat {*} $target_library]

set TOP_MODULE flash_attn_core_banked_prefetch
set CLK_PERIOD 10.0

file mkdir syn/reports
file mkdir syn/logs
file mkdir syn/output

analyze -format sverilog [list \
  rtl/memory/sram_1r1w.sv \
  rtl/memory/output_buffer.sv \
  rtl/memory/banked_scratchpad.sv \
  rtl/memory/banked_tile_loader.sv \
  rtl/systolic/pe.sv \
  rtl/systolic/systolic_array.sv \
  rtl/systolic/array_controller.sv \
  rtl/softmax/exp_lut.sv \
  rtl/softmax/online_softmax.sv \
  rtl/quantization/dequantizer.sv \
  rtl/ctrl/addr_gen.sv \
  rtl/ctrl/tile_controller_banked_prefetch.sv \
  rtl/top/flash_attn_core_banked_prefetch.sv \
]

elaborate $TOP_MODULE
current_design $TOP_MODULE
link

check_design > syn/reports/${TOP_MODULE}_compile_check_design.rpt

create_clock -name clk -period $CLK_PERIOD [get_ports clk]

# Async active-low reset. Keep it out of data timing for this first-pass compile.
if {[sizeof_collection [get_ports rst_n]] > 0} {
  set_ideal_network [get_ports rst_n]
  set_false_path -from [get_ports rst_n]
}

# Conservative IO assumptions for first-pass block synthesis.
set_input_delay  1.0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 1.0 -clock clk [all_outputs]

# First pass: use compile, not compile_ultra. Faster and easier to debug.
compile

report_timing      -max_paths 20 -nworst 5 > syn/reports/${TOP_MODULE}_compile_timing.rpt
report_area        -hierarchy            > syn/reports/${TOP_MODULE}_compile_area.rpt
report_power                             > syn/reports/${TOP_MODULE}_compile_power.rpt
report_qor                               > syn/reports/${TOP_MODULE}_compile_qor.rpt
report_hierarchy                         > syn/reports/${TOP_MODULE}_compile_hierarchy.rpt
report_reference   -hierarchy            > syn/reports/${TOP_MODULE}_compile_reference.rpt

write -format verilog -hierarchy -output syn/output/${TOP_MODULE}_compile_mapped.v
write -format ddc     -hierarchy -output syn/output/${TOP_MODULE}_compile.ddc
write_sdc                                syn/output/${TOP_MODULE}_compile.sdc

puts "=== COMPILE DONE: $TOP_MODULE ==="
exit
