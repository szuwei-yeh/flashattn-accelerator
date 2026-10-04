# Recheck an accepted mapped run without recompiling or changing its netlist.
# DC_TARGET_LIBRARY and MAPPED_RUN_DIR are supplied by the caller.
foreach key {DC_TARGET_LIBRARY MAPPED_RUN_DIR} {
    if {![info exists ::env($key)] || [string trim $::env($key)] eq ""} {
        error "Required environment variable: $key"
    }
}
set base [file normalize $::env(MAPPED_RUN_DIR)]
if {![file exists $base/SUCCESS]} { error "Require an accepted synthesis run" }
set_app_var target_library [list $::env(DC_TARGET_LIBRARY)]
set_app_var link_library [concat * $target_library]
read_ddc $base/artifacts/mapped.ddc
if {![link]} { error "Mapped design link failed" }
report_constraint -all_violators -significant_digits 9 > $base/reports/postcheck_constraints.rpt
report_qor > $base/reports/postcheck_qor.rpt
report_timing -delay_type min -max_paths 20 -nworst 5 > $base/reports/timing_min.rpt
check_timing > $base/reports/check_timing.rpt
check_design > $base/reports/postcheck_design.rpt
report_clock > $base/reports/postcheck_clocks.rpt

set summary [open $base/reports/postcheck_structure.rpt w]
set latch_count [sizeof_collection [all_registers -level_sensitive]]
puts $summary "MAPPED_LATCH_COUNT $latch_count"
if {$latch_count != 0} { error "Mapped latches require review" }
set pin_count 0
set live_count 0
set cells [get_cells -hierarchical -filter {ref_name =~ dequantizer*DW02_mult* || ref_name =~ online_softmax*DW02_mult*}]
foreach_in_collection cell $cells {
    foreach_in_collection pin [get_pins -of_objects $cell] {
        set pname [get_object_name $pin]
        if {![regexp {/PRODUCT\[1\]$} $pname]} { continue }
        incr pin_count
        set endpoints [all_fanout -flat -endpoints_only -from $pin]
        if {[sizeof_collection $endpoints] != 0} { incr live_count }
        puts $summary "UNUSED_LOW_PRODUCT_PIN $pname endpoints=[sizeof_collection $endpoints]"
    }
}
puts $summary "UNUSED_LOW_PRODUCT_PINS $pin_count"
puts $summary "UNUSED_LOW_PRODUCT_LIVE_ENDPOINTS $live_count"
if {$pin_count == 0 || $live_count != 0} { error "Unused low-product pins require review" }
foreach ref {NAND2X1 NAND3X1 NOR2X1 AOI21X1 AOI22X1 DFFSR TBUFX1} {
    foreach_in_collection pin [get_lib_pins -quiet */$ref/* -filter {direction == out}] {
        puts $summary "LIB_PIN_CAP [get_object_name $pin] [get_attribute $pin max_capacitance]"
    }
}
puts $summary "MAPPED_POSTCHECK_DONE"
close $summary
puts "MAPPED_POSTCHECK_DONE"
exit
