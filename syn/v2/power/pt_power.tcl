# Averaged RTL activity on the unchanged M6B2 mapped standard-cell endpoint.
set_app_var power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var target_library [list $env(DC_TARGET_LIBRARY)]
set_app_var link_path [concat * $target_library]
read_verilog $env(PT_MAPPED_V)
current_design $env(PT_TOP)
if {![link]} {error "M6C mapped link failed"}
read_sdc $env(PT_SDC)
update_timing
report_clock > clocks.rpt
report_units > units.rpt
check_timing -verbose -include no_clock > no_clock.rpt
set clocks_count 0
set bad 0
set fh [open clock_scope.rpt w]
foreach_in_collection pin [all_registers -clock_pins] {
    incr clocks_count
    set clocks [get_attribute $pin clocks]
    if {[sizeof_collection $clocks]!=1 || [get_object_name $clocks] ne "clk"} {
        incr bad
        puts $fh "BAD [get_object_name $pin]"
    }
}
puts $fh "actual_register_clock_pins=$clocks_count"
puts $fh "actual_register_clock_pins_outside_clk=$bad"
close $fh
if {$clocks_count==0 || $bad!=0} {error "M6C actual CLK scope failed"}
read_vcd -rtl -strip_path tb_m6b1/h/u_dut -time [list $env(ACTIVITY_BEGIN_NS) $env(ACTIVITY_END_NS)] $env(PT_ACTIVITY)
report_switching_activity > activity_before.rpt
update_power
report_switching_activity > activity_after.rpt
report_switching_activity -hierarchy > activity_hierarchy.rpt
report_switching_activity -list_by_source default -show_pin > residual_default.rpt
report_switching_activity -include_only sequential -list_by_source propagated -show_pin > residual_sequential.rpt
report_power -verbose -significant_digits 8 -nosplit > power.rpt
report_power -hierarchy -levels 3 -significant_digits 8 -nosplit > power_hierarchy.rpt
puts "M6C_POWER_COMPLETE"
exit
