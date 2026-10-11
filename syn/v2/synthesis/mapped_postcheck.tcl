# Read-only saved-DDC audit. No compile, optimization, or netlist write.
if {[catch {
set_host_options -max_cores 1
set_app_var target_library [list $::env(DC_TARGET_LIBRARY)]
set_app_var link_library [concat * $target_library]
read_ddc $::env(MAPPED_DDC)
if {![link]} {error "Saved DDC link failed"}
set out $::env(POSTCHECK_DIR)
set f [open $out/audit.rpt w]
puts $f "TOP [get_object_name [current_design]]"
set result_bits [get_cells -hierarchical -filter {full_name =~ u_engine/results_reg* && is_hierarchical == false}]
puts $f "RESULT_BITS [sizeof_collection $result_bits]"
if {[sizeof_collection $result_bits]!=32768} {error "Result register population changed"}
puts $f "LATCHES [sizeof_collection [all_registers -level_sensitive]]"
set n 0
set live 0
set cells [get_cells -hierarchical -filter {ref_name =~ dequantizer*DW02_mult* || ref_name =~ online_softmax*DW02_mult*}]
foreach_in_collection c $cells {
 foreach_in_collection p [get_pins -of_objects $c] {
  if {![regexp {/PRODUCT\[1\]$} [get_object_name $p]]} {continue}
  incr n
  set endpoints [sizeof_collection [all_fanout -flat -endpoints_only -from $p]]
  if {$endpoints!=0} {incr live}
  puts $f "LOW_PRODUCT_PIN [get_object_name $p] ENDPOINTS $endpoints"
 }
}
puts $f "LOW_PRODUCT_PINS $n LIVE $live"
if {$n!=($::env(SYN_DEQUANT_LANES)+64) || $live!=0 || [sizeof_collection [all_registers -level_sensitive]]!=0} {error "Unexpected low-product fanout/latches"}
# Bulk attributes avoid a million separate Tcl API calls; count every occurrence.
array set counts {}
foreach r [get_attribute [get_cells -hierarchical] ref_name] {
 if {![info exists counts($r)]} {set counts($r) 0}
 incr counts($r)
}
set sum 0.0
set total 0
foreach r [lsort [array names counts]] {
 set libs [get_lib_cells -quiet */$r]
 if {[sizeof_collection $libs]!=1} {continue}
 set a [get_attribute $libs area]
 puts $f "LIB_AREA_TOKEN $r COUNT $counts($r) AREA $a"
 incr total $counts($r)
 set sum [expr {$sum+$counts($r)*$a}]
}
puts $f "TYPE_COUNT_CELL_TOTAL $total"
puts $f "TYPE_COUNT_AREA_SUM [format %.12f $sum]"
close $f
report_area > $out/area.rpt
report_clock > $out/clocks.rpt
report_constraint -all_violators -significant_digits 9 > $out/constraints.rpt
check_timing > $out/check_timing.rpt
puts "M6B2_READ_ONLY_POSTCHECK_PASS"
} message]} {puts "Error: $message";exit 1}
exit 0
