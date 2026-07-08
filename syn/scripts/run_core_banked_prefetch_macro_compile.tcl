set_app_var search_path [list . ./rtl ./rtl/memory ./rtl/systolic ./rtl/softmax ./rtl/quantization ./rtl/ctrl ./rtl/top ./syn/blackboxes]
set_app_var target_library [list /fs/ece/PDKs/bongjin/NCSU_FreePDK45/osu_soc/lib/files/gscl45nm.db]
set_app_var link_library   [concat {*} $target_library]

set TOP_MODULE flash_attn_core_banked_prefetch
set CLK_PERIOD 10.0

file mkdir syn/reports
file mkdir syn/logs
file mkdir syn/output

# Fresh read. Use synthesis-only macro stubs for SRAM/ROM.
analyze -format sverilog [list \
  syn/blackboxes/sram_1r1w_bb.sv \
  syn/blackboxes/exp_lut_bb.sv \
  rtl/memory/output_buffer.sv \
  rtl/memory/banked_scratchpad.sv \
  rtl/memory/banked_tile_loader.sv \
  rtl/systolic/pe.sv \
  rtl/systolic/systolic_array.sv \
  rtl/systolic/array_controller.sv \
  rtl/softmax/online_softmax.sv \
  rtl/quantization/dequantizer.sv \
  rtl/ctrl/addr_gen.sv \
  rtl/ctrl/tile_controller_banked_prefetch.sv \
  rtl/core/flash_attn_core_banked_prefetch.sv \
]

elaborate $TOP_MODULE
current_design $TOP_MODULE
link

check_design > syn/reports/${TOP_MODULE}_macro_compile_check_design.rpt

create_clock -name clk -period $CLK_PERIOD [get_ports clk]

if {[sizeof_collection [get_ports rst_n]] > 0} {
  set_ideal_network [get_ports rst_n]
  set_false_path -from [get_ports rst_n]
}

set_input_delay  1.0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 1.0 -clock clk [all_outputs]

# First macro-based full-core pass: plain compile for runtime/debuggability.
compile

report_timing      -max_paths 20 -nworst 5 > syn/reports/${TOP_MODULE}_macro_compile_timing.rpt
report_area        -hierarchy            > syn/reports/${TOP_MODULE}_macro_compile_area.rpt
report_power                             > syn/reports/${TOP_MODULE}_macro_compile_power.rpt
report_qor                               > syn/reports/${TOP_MODULE}_macro_compile_qor.rpt
report_hierarchy                         > syn/reports/${TOP_MODULE}_macro_compile_hierarchy.rpt
report_reference   -hierarchy            > syn/reports/${TOP_MODULE}_macro_compile_reference.rpt

write -format verilog -hierarchy -output syn/output/${TOP_MODULE}_macro_compile_mapped.v
write -format ddc     -hierarchy -output syn/output/${TOP_MODULE}_macro_compile.ddc
write_sdc                                syn/output/${TOP_MODULE}_macro_compile.sdc

puts "=== MACRO COMPILE DONE: $TOP_MODULE ==="
exit
