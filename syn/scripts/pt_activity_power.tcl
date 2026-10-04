# Activity-based, pre-layout standard-cell power on an accepted mapped netlist.
# SRAM/ROM remain logical blackboxes; this is not physical/system power signoff.
proc need {key} {
    if {![info exists ::env($key)] || [string trim $::env($key)] eq ""} {
        error "Required environment variable: $key"
    }
    return $::env($key)
}
set run_dir [file normalize [need PT_RUN_DIR]]
file mkdir $run_dir
set mapped [file normalize [need PT_MAPPED_V]]
set sdc [file normalize [need PT_SDC]]
set library [file normalize [need DC_TARGET_LIBRARY]]
set activity [file normalize [need PT_ACTIVITY]]
set top [need PT_TOP]
set begin_ns [need PT_ACTIVITY_BEGIN_NS]
set end_ns [need PT_ACTIVITY_END_NS]
foreach path [list $mapped $sdc $library $activity] {
    if {![file isfile $path]} { error "Missing analysis input: $path" }
}
if {$end_ns <= $begin_ns} { error "Empty activity window" }
set_app_var power_enable_analysis true
set_app_var power_analysis_mode averaged
set_app_var target_library [list $library]
set_app_var link_path [concat * $target_library]
set_app_var search_path [concat [list [file dirname $library]] $search_path]
read_verilog $mapped
current_design $top
if {![link]} { error "Mapped netlist link failed" }
read_sdc $sdc
report_clock > $run_dir/clocks.rpt
check_timing > $run_dir/check_timing.rpt
report_units > $run_dir/units.rpt
read_vcd -rtl -strip_path tb_attention/dut/u_dut -time [list $begin_ns $end_ns] $activity
report_switching_activity > $run_dir/activity_before.rpt
update_power
report_switching_activity > $run_dir/activity_after.rpt
report_switching_activity -list_by_source default -show_pin > $run_dir/residual_default.rpt
report_switching_activity -include_only sequential -list_by_source propagated -show_pin > $run_dir/residual_sequential.rpt
report_switching_activity -hierarchy > $run_dir/activity_hierarchy.rpt
report_power -verbose -significant_digits 8 -nosplit > $run_dir/power.rpt
report_power -hierarchy -levels 3 -significant_digits 8 -nosplit > $run_dir/power_hierarchy.rpt
set fh [open $run_dir/analysis_identity.txt w]
puts $fh "top=$top"
puts $fh "mode=averaged"
puts $fh "activity_begin_ns=$begin_ns"
puts $fh "activity_end_ns=$end_ns"
puts $fh "activity_duration_ns=[expr {$end_ns-$begin_ns}]"
puts $fh "library=[file tail $library]"
puts $fh "mapped_standard_cells=[sizeof_collection [get_cells -hierarchical -filter {is_hierarchical == false}]]"
close $fh
puts "RESULT: PASS PrimeTime activity power"
exit
