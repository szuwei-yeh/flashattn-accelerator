set_app_var search_path [list . ./rtl ./rtl/memory ./rtl/systolic ./rtl/softmax ./rtl/quantization ./rtl/ctrl ./rtl/top]
set_app_var target_library [list /fs/ece/PDKs/bongjin/NCSU_FreePDK45/osu_soc/lib/files/gscl45nm.db]
set_app_var link_library   [concat {*} $target_library]

set TOP_MODULE flash_attn_core_banked_prefetch

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
  rtl/core/flash_attn_core_banked_prefetch.sv \
]

elaborate $TOP_MODULE
current_design $TOP_MODULE
link

check_design > syn/reports/${TOP_MODULE}_elab_check_design.rpt

report_hierarchy > syn/reports/${TOP_MODULE}_elab_hierarchy.rpt
report_reference -hierarchy > syn/reports/${TOP_MODULE}_elab_reference.rpt

write -format ddc -hierarchy -output syn/output/${TOP_MODULE}_elab.ddc

puts "=== ELAB DONE: $TOP_MODULE ==="
exit
