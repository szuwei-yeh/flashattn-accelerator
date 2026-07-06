set_app_var search_path [list . ./rtl ./rtl/memory]
set_app_var target_library [list /fs/ece/PDKs/bongjin/NCSU_FreePDK45/osu_soc/lib/files/gscl45nm.db]
set_app_var link_library   [concat {*} $target_library]

set TOP_MODULE banked_tile_loader
set CLK_PERIOD 10.0

file mkdir syn/reports
file mkdir syn/logs
file mkdir syn/output

analyze -format sverilog [list \
  rtl/memory/banked_tile_loader.sv \
]

elaborate $TOP_MODULE
current_design $TOP_MODULE
link

check_design > syn/reports/${TOP_MODULE}_check_design.rpt

create_clock -name clk -period $CLK_PERIOD [get_ports clk]

if {[sizeof_collection [get_ports rst_n]] > 0} {
  set_ideal_network [get_ports rst_n]
  set_false_path -from [get_ports rst_n]
}

set_input_delay  1.0 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 1.0 -clock clk [all_outputs]

compile_ultra

report_timing      -max_paths 10 > syn/reports/${TOP_MODULE}_timing.rpt
report_area        -hierarchy    > syn/reports/${TOP_MODULE}_area.rpt
report_power                     > syn/reports/${TOP_MODULE}_power.rpt
report_qor                       > syn/reports/${TOP_MODULE}_qor.rpt
report_hierarchy                 > syn/reports/${TOP_MODULE}_hierarchy.rpt
report_reference   -hierarchy    > syn/reports/${TOP_MODULE}_reference.rpt

write -format verilog -hierarchy -output syn/output/${TOP_MODULE}_mapped.v
write -format ddc     -hierarchy -output syn/output/${TOP_MODULE}.ddc
write_sdc syn/output/${TOP_MODULE}.sdc

puts "=== DONE: $TOP_MODULE ==="
exit
