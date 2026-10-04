# Audit actual register CLK domains independently of async-control warnings.
# Use the same PT_MAPPED_V / PT_SDC / PT_TOP / DC_TARGET_LIBRARY inputs as power.
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
foreach input [list $mapped $sdc $library] {
    if {![file isfile $input]} {error "Missing analysis input: $input"}
}
set_app_var target_library [list $library]
set_app_var link_path [concat * $target_library]
read_verilog $mapped
current_design [need PT_TOP]
if {![link]} {error "Mapped link failed"}
read_sdc $sdc
update_timing
check_timing -verbose -include no_clock > $run_dir/no_clock.rpt
report_clock > $run_dir/clocks.rpt
report_disable_timing > $run_dir/disabled.rpt
set pins [all_registers -clock_pins]
set bad 0
set counted 0
set f [open $run_dir/clock_scope.txt w]
foreach_in_collection pin $pins {
    incr counted
    set clocks [get_attribute $pin clocks]
    set names [get_object_name $clocks]
    if {[sizeof_collection $clocks] != 1 || $names ne "clk"} {
        incr bad
        puts $f "BAD [get_object_name $pin] clocks=$names"
    }
    if {$counted <= 10} {puts $f "SAMPLE [get_object_name $pin] clocks=$names"}
}
puts $f "actual_register_clock_pins=$counted"
puts $f "actual_register_clock_pins_outside_clk=$bad"
close $f
if {$counted == 0 || $bad != 0} {error "Actual register CLK domain audit failed"}
puts "RESULT: PASS actual register clock scope"
exit
