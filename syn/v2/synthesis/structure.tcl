# Read-only structural acceptance in the currently linked top. No logic removal.
set f [open [file join $REPORT_DIR ${M6B_STAGE}_structure.rpt] w]
puts $f "TOP [get_object_name [current_design]]"
puts $f "TOP_PARAMS [get_attribute [current_design] hdl_canonical_params]"
set engine_ref [get_attribute [get_cells u_engine] ref_name]
set core_ref [get_attribute [get_cells u_engine/u_core] ref_name]
puts $f "ENGINE_PARAMS [get_attribute [get_designs $engine_ref] hdl_canonical_params]"
puts $f "ENGINE_DEFAULTS [get_attribute [get_designs $engine_ref] hdl_canonical_default_params]"
puts $f "CORE_PARAMS [get_attribute [get_designs $core_ref] hdl_canonical_params]"
puts $f "CORE_DEFAULTS [get_attribute [get_designs $core_ref] hdl_canonical_default_params]"
foreach {label pattern count} [list SRAM sram_1r1w* 81 ROM exp_lut* 16 DEQUANT dequantizer* $LANES SOFTMAX online_softmax_masked_m2c3* 16 CORE flash_attn_core_decode_m3b* 1 ARRAY systolic_array* 1 PE pe* 256 OUTPUT_BUFFER output_buffer* 1 READ_DMA dma_engine_vec* 1 CACHE kv_cache_banked_decode_m2c3* 2 BANKED banked_scratchpad* 5 CSR axi_lite_csr_m6b1* 1 JOB soc_job_adapter_m5* 1 PACK gqa_output_packer_m5* 1 WRITE_DMA dma_write_engine_m5* 1 ENGINE flash_attn_serial_m6b1* 1] {
 set cells [get_cells -hierarchical -filter "ref_name =~ $pattern && ref_name !~ *_DW* && ref_name !~ dequantizer_tile*"]
 puts $f "$label [sizeof_collection $cells]"
 foreach_in_collection c $cells {puts $f "$label [get_object_name $c] [get_attribute $c ref_name]"}
 if {[sizeof_collection $cells]!=$count} {error "Unexpected $M6B_STAGE $label count"}
}
set staging [get_cells -hierarchical -filter {ref_name =~ dequantizer_tile* && ref_name !~ *_DW*}]
puts $f "DEQUANT_TILE [sizeof_collection $staging]"
if {[sizeof_collection $staging]!=($LANES==16?1:0)} {error "Unexpected score staging count"}
set csr_ref [get_attribute [get_cells u_csr] ref_name]
puts $f "CSR_PARAMS [get_attribute [get_designs $csr_ref] hdl_canonical_params]"
set expected_lane_token [format "DEQUANT_LANES=32'h%08x" $LANES]
foreach token [list [get_attribute [current_design] hdl_canonical_params] [get_attribute [get_designs $engine_ref] hdl_canonical_params] [get_attribute [get_designs $core_ref] hdl_canonical_params] [get_attribute [get_designs $csr_ref] hdl_canonical_params]] {
 if {[string first $expected_lane_token $token]<0} {error "Incorrect lane selection in top/engine/core/CSR"}
}
set core_defaults [get_attribute [get_designs $core_ref] hdl_canonical_default_params]
foreach token {HEAD_DIM=32'h00000010 SEQ_LEN=32'h00000040 SRAM_DEPTH=32'h00001000 TILE_SIZE=32'h00000010} {
 if {[string first $token $core_defaults]<0} {error "Incorrect core geometry"}
}
set latches [all_registers -level_sensitive]
set outside 0
set live 0
foreach_in_collection c $latches {
 if {![string match *active_row.u_softmax/softmax_flat_reg* [get_object_name $c]]} {incr outside}
 foreach_in_collection pin [get_pins -of_objects $c -filter {direction == out}] {
  if {[sizeof_collection [all_fanout -flat -endpoints_only -from $pin]]>0} {incr live}
 }
}
puts $f "LATCHES [sizeof_collection $latches] OUTSIDE_COMPAT_OUTPUT $outside LIVE_LATCH_OUTPUT_PINS $live"
if {$outside!=0 || $live!=0 || ($M6B_STAGE eq "mapped" && [sizeof_collection $latches]!=0)} {error "Unexpected/live latch"}
set unresolved 0
set intrinsic 0
set constants 0
set standard_cells 0
set standard_area 0.0
set gtech 0
array unset m6b_refs
array set m6b_refs {}
foreach_in_collection c [get_cells -hierarchical] {
 set r [get_attribute $c ref_name]
 if {$r eq "**logic_0**" || $r eq "**logic_1**"} {incr constants;continue}
 if {[string index $r 0] eq "*"} {incr intrinsic;continue}
 if {[string match GTECH_* $r]} {incr gtech}
 if {![info exists m6b_refs($r)]} {
  set m6b_refs($r) [list [expr {[sizeof_collection [get_designs -quiet $r]]>0}] [expr {[sizeof_collection [get_lib_cells -quiet */$r]]>0}]]
 }
 if {[lindex $m6b_refs($r) 1]} {incr standard_cells;set standard_area [expr {$standard_area+[get_attribute $c area]}]}
 if {![lindex $m6b_refs($r) 0] && ![lindex $m6b_refs($r) 1]} {incr unresolved;puts $f "UNRESOLVED [get_object_name $c] $r"}
}
puts $f "UNRESOLVED_MODULES $unresolved"
puts $f "INTRINSIC_OPERATORS $intrinsic"
puts $f "GTECH_CELLS $gtech"
puts $f "CONSTANT_NODES $constants"
puts $f "LIBRARY_LEAF_CELL_COUNT $standard_cells"
puts $f "LIBRARY_LEAF_AREA_SUM [format %.9f $standard_area]"
puts $f "REGISTERS [sizeof_collection [all_registers]]"
close $f
if {$unresolved!=0 || ($M6B_STAGE eq "mapped" && ($intrinsic!=0 || $gtech!=0))} {error "Unresolved or unmapped reference"}
puts "M6B2_STRUCTURE_PASS $M6B_STAGE"
