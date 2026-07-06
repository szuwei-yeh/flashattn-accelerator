# ============================================================================
#  dc_synth_template.tcl — Synopsys Design Compiler synthesis TEMPLATE
#
#  *** THIS IS A TEMPLATE. It does NOT run as-is. ***
#  The real Synopsys run happens on a machine that has a DC license and a
#  standard-cell library.  Before running, fill in the four placeholders below
#  (target_library / link_library / TOP_MODULE / CLK_PERIOD) with values that
#  exist ON THAT SERVER — do not commit real vendor library paths into this repo.
#
#  Nothing here was executed locally; this file is scaffold only.
#
#  Usage on the license server (from the repo root or anywhere):
#     dc_shell -f syn/scripts/dc_synth_template.tcl \
#              | tee syn/logs/<top>_<date>.log
#  or, overriding the filelist/top from the command line:
#     dc_shell -x "set TOP_MODULE flash_attn_core_banked_prefetch; \
#                  set FILELIST syn/filelists/core_banked_prefetch.f" \
#              -f syn/scripts/dc_synth_template.tcl
#
#  Recommended synth order (see syn/README.md):
#     1) systolic_array                       filelist: systolic_array.f
#     2) flash_attn_core_banked_prefetch      filelist: core_banked_prefetch.f
#     3) flash_attn_top_dma_banked_prefetch   filelist: top_dma_banked_prefetch.f
# ============================================================================

# ---------------------------------------------------------------------------
# 0. Repo root resolution
#    This script lives in <repo>/syn/scripts/. Resolve the repo root so the
#    filelists (which hold repo-root-relative paths) work regardless of cwd.
#    If [info script] is unavailable in your dc_shell invocation, set REPO_ROOT
#    manually below.
# ---------------------------------------------------------------------------
if {![info exists REPO_ROOT]} {
    if {[info script] ne ""} {
        set REPO_ROOT [file normalize [file join [file dirname [info script]] .. ..]]
    } else {
        # <<< FILL IN on the server if [info script] is empty >>>
        set REPO_ROOT "/path/to/flashattn-accelerator"
    }
}
puts "REPO_ROOT = $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Placeholders — FILL IN ON THE SERVER
# ---------------------------------------------------------------------------
# Standard-cell + IO libraries available on the license server. Examples only;
# replace with the real .db files for your target process/library.
set target_library "REPLACE_WITH_STDCELL.db"       ;# <<< FILL IN >>>
set link_library   "* $target_library"             ;# add IO/mem .db as needed  <<< FILL IN >>>

# Which design to compile. Override on the command line per run.
if {![info exists TOP_MODULE]} {
    set TOP_MODULE "flash_attn_core_banked_prefetch" ;# <<< FILL IN / OVERRIDE >>>
}

# Target clock period in ns (technology-dependent). The local Verilator flow is
# cycle-accurate only and says NOTHING about achievable frequency — pick this on
# the server based on the PDK. Example starting points: 10 ns (100 MHz) sign-off,
# or sweep to find f_max.
if {![info exists CLK_PERIOD]} {
    set CLK_PERIOD "10.0"                            ;# <<< FILL IN >>> (ns)
}

# Clock port name. All tops in this repo use `clk`.
set CLK_PORT "clk"

# Filelist to read. Defaults to the core; override per TOP_MODULE.
if {![info exists FILELIST]} {
    set FILELIST "syn/filelists/core_banked_prefetch.f" ;# <<< match TOP_MODULE >>>
}

# ---------------------------------------------------------------------------
# 2. Work directories (kept out of the repo except reports/logs)
# ---------------------------------------------------------------------------
set REPORT_DIR "$REPO_ROOT/syn/reports"
set OUT_DIR    "$REPO_ROOT/syn/output"
file mkdir $REPORT_DIR
file mkdir $OUT_DIR
# define_design_lib WORK -path ./WORK   ;# uncomment if you want an explicit WORK

# ---------------------------------------------------------------------------
# 3. Read the filelist (repo-root-relative paths → absolute) and analyze
#    Using analyze/elaborate (SystemVerilog). Order in the .f is leaf-first.
# ---------------------------------------------------------------------------
set fl_path [file join $REPO_ROOT $FILELIST]
puts "Reading filelist: $fl_path"
set fh [open $fl_path r]
set rtl_files {}
foreach line [split [read $fh] "\n"] {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} { continue }
    lappend rtl_files [file join $REPO_ROOT $line]
}
close $fh
puts "RTL files to analyze:"
foreach f $rtl_files { puts "    $f" }

analyze -format sverilog $rtl_files

# ---------------------------------------------------------------------------
# 4. Elaborate + link
# ---------------------------------------------------------------------------
elaborate $TOP_MODULE
current_design $TOP_MODULE
link

# ---------------------------------------------------------------------------
# 5. Sanity check the elaborated netlist BEFORE compile
#    Expect warnings (not errors) for the known items documented in syn/README.md:
#    large inferred flop arrays (tile registers / shadow), SRAM-like memories
#    (sram_1r1w, banked_scratchpad), exp_lut ROM ($readmemh), copy-style swap mux.
# ---------------------------------------------------------------------------
check_design > "$REPORT_DIR/${TOP_MODULE}_check_design.rpt"

# ---------------------------------------------------------------------------
# 6. Constraints — single clock, async active-low reset (rst_n)
#    NOTE: all sequential logic here is `posedge clk or negedge rst_n`, so rst_n
#    is an async reset. On the server, mark it so DC does not try to time it as
#    data (see set_ideal_network / set_false_path below).
# ---------------------------------------------------------------------------
create_clock -name $CLK_PORT -period $CLK_PERIOD [get_ports $CLK_PORT]

# Reset handling — uncomment/adjust on the server:
# set_ideal_network            [get_ports rst_n]
# set_false_path -from         [get_ports rst_n]
# set_dont_touch_network       [get_ports $CLK_PORT]

# Conservative default IO timing (tune on the server):
# set_input_delay  [expr {0.2 * $CLK_PERIOD}] -clock $CLK_PORT [remove_from_collection [all_inputs] [get_ports $CLK_PORT]]
# set_output_delay [expr {0.2 * $CLK_PERIOD}] -clock $CLK_PORT [all_outputs]

# ---------------------------------------------------------------------------
# 7. Compile
#    compile_ultra is the modern flow. For the very first bring-up you may prefer
#    a plain `compile` to get a quick, less-optimized result and shorter runtime.
# ---------------------------------------------------------------------------
compile_ultra
# compile    ;# alternative: faster first-pass bring-up

# ---------------------------------------------------------------------------
# 8. Reports
# ---------------------------------------------------------------------------
report_timing      -max_paths 20 -nworst 5 > "$REPORT_DIR/${TOP_MODULE}_timing.rpt"
report_area        -hierarchy            > "$REPORT_DIR/${TOP_MODULE}_area.rpt"
report_power                             > "$REPORT_DIR/${TOP_MODULE}_power.rpt"
report_qor                               > "$REPORT_DIR/${TOP_MODULE}_qor.rpt"
report_hierarchy                         > "$REPORT_DIR/${TOP_MODULE}_hierarchy.rpt"
report_reference   -hierarchy            > "$REPORT_DIR/${TOP_MODULE}_reference.rpt"

# ---------------------------------------------------------------------------
# 9. Write out netlist / ddc / sdc
# ---------------------------------------------------------------------------
write -format verilog -hierarchy -output "$OUT_DIR/${TOP_MODULE}_netlist.v"
write -format ddc     -hierarchy -output "$OUT_DIR/${TOP_MODULE}.ddc"
write_sdc                                "$OUT_DIR/${TOP_MODULE}.sdc"

puts "=== DONE: $TOP_MODULE — reports in $REPORT_DIR, netlist in $OUT_DIR ==="
# exit   ;# uncomment for batch/non-interactive runs
