# =============================================================================
# Generic Synopsys Design Compiler flow.
#
# Do not invoke this file directly. syn/scripts/run_server.sh validates a design
# profile, creates an isolated run directory, exports the required SYN_* values,
# and captures the complete dc_shell transcript.
# =============================================================================

proc require_env {name} {
    if {![info exists ::env($name)] || [string trim $::env($name)] eq ""} {
        error "Required environment variable $name is not set"
    }
    return $::env($name)
}

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set REPO_ROOT  [file normalize [file join $SCRIPT_DIR .. ..]]

set PROFILE        [require_env SYN_PROFILE]
set RUN_TAG        [require_env SYN_RUN_TAG]
set TOP_MODULE     [require_env SYN_TOP_MODULE]
set FILELIST_REL   [require_env SYN_FILELIST]
set RUN_MODE       [require_env SYN_RUN_MODE]
set CLK_PERIOD     [require_env SYN_CLK_PERIOD]
set IO_DELAY       [require_env SYN_IO_DELAY]
set TARGET_DB      [require_env DC_TARGET_LIBRARY]
set GIT_COMMIT     [require_env SYN_GIT_COMMIT]
set GIT_DIRTY      [require_env SYN_GIT_DIRTY]

set RUN_DIR        [file normalize [file join $REPO_ROOT syn runs $RUN_TAG $PROFILE]]
set REPORT_DIR     [file join $RUN_DIR reports]
set ARTIFACT_DIR   [file join $RUN_DIR artifacts]
set WORK_DIR       [file join $RUN_DIR work]
set FILELIST_PATH  [file normalize [file join $REPO_ROOT $FILELIST_REL]]

foreach dir [list $REPORT_DIR $ARTIFACT_DIR $WORK_DIR] {
    file mkdir $dir
}

puts "============================================================"
puts "DC synthesis run"
puts "  repository : $REPO_ROOT"
puts "  git commit : $GIT_COMMIT"
puts "  git dirty  : $GIT_DIRTY"
puts "  run tag    : $RUN_TAG"
puts "  profile    : $PROFILE"
puts "  top        : $TOP_MODULE"
puts "  filelist   : $FILELIST_PATH"
puts "  mode       : $RUN_MODE"
puts "  clock      : $CLK_PERIOD ns"
puts "  target db  : $TARGET_DB"
puts "  run dir    : $RUN_DIR"
puts "============================================================"

set manifest [open [file join $RUN_DIR manifest.txt] w]
puts $manifest "run_tag=$RUN_TAG"
puts $manifest "profile=$PROFILE"
puts $manifest "top_module=$TOP_MODULE"
puts $manifest "filelist=$FILELIST_REL"
puts $manifest "run_mode=$RUN_MODE"
puts $manifest "clock_period_ns=$CLK_PERIOD"
puts $manifest "io_delay_ns=$IO_DELAY"
puts $manifest "target_library=$TARGET_DB"
puts $manifest "git_commit=$GIT_COMMIT"
puts $manifest "git_dirty=$GIT_DIRTY"
puts $manifest "elaboration_parameters=$::env(SYN_ELAB_PARAMETERS)"
puts $manifest "source_hashes=source_sha256.json"
puts $manifest "started_at=[clock format [clock seconds] -format {%Y-%m-%dT%H:%M:%S%z}]"
close $manifest

# The target library path is supplied by the server environment. Keep machine-
# specific PDK paths out of the repository.
set_app_var search_path [list \
    $REPO_ROOT \
    [file join $REPO_ROOT rtl] \
    [file join $REPO_ROOT rtl systolic] \
    [file join $REPO_ROOT rtl softmax] \
    [file join $REPO_ROOT rtl quantization] \
    [file join $REPO_ROOT rtl memory] \
    [file join $REPO_ROOT rtl ctrl] \
    [file join $REPO_ROOT rtl core] \
    [file join $REPO_ROOT rtl interface] \
    [file join $REPO_ROOT rtl top] \
    [file join $REPO_ROOT syn blackboxes] \
    [file dirname $TARGET_DB] \
]
set_app_var target_library [list $TARGET_DB]
set_app_var link_library   [concat "*" $target_library]

if {![file exists $TARGET_DB]} {
    error "Target library does not exist on this server: $TARGET_DB"
}
if {![file exists $FILELIST_PATH]} {
    error "Synthesis filelist does not exist: $FILELIST_PATH"
}

# Filelists contain repository-root-relative paths, comments, and blank lines.
set fh [open $FILELIST_PATH r]
set rtl_files {}
foreach raw_line [split [read $fh] "\n"] {
    set line [string trim $raw_line]
    if {$line eq "" || [string match "#*" $line]} {
        continue
    }
    set rtl_path [file normalize [file join $REPO_ROOT $line]]
    if {![file exists $rtl_path]} {
        error "Filelist entry does not exist: $line ($rtl_path)"
    }
    lappend rtl_files $rtl_path
}
close $fh

if {[llength $rtl_files] == 0} {
    error "Filelist is empty: $FILELIST_PATH"
}

define_design_lib WORK -path $WORK_DIR
analyze -work WORK -format sverilog $rtl_files

if {[info exists ::env(SYN_ELAB_PARAMETERS)] &&
    [string trim $::env(SYN_ELAB_PARAMETERS)] ne ""} {
    set ELAB_PARAMETERS $::env(SYN_ELAB_PARAMETERS)
    puts "Elaboration parameters: $ELAB_PARAMETERS"
    elaborate $TOP_MODULE -work WORK -parameters $ELAB_PARAMETERS
} else {
    elaborate $TOP_MODULE -work WORK
}

# elaborate already selected the effective parameterized design. Re-selecting
# the unparameterized module name can fail or synthesize the wrong geometry.
set EFFECTIVE_TOP [get_object_name [current_design]]
if {![string match "${TOP_MODULE}*" $EFFECTIVE_TOP]} {
    error "Unexpected elaborated top: $EFFECTIVE_TOP"
}
set manifest [open [file join $RUN_DIR manifest.txt] a]
puts $manifest "effective_top=$EFFECTIVE_TOP"
close $manifest
if {![link]} { error "Link failed for $EFFECTIVE_TOP" }

check_design > [file join $REPORT_DIR check_design.rpt]
report_hierarchy > [file join $REPORT_DIR hierarchy.rpt]
report_reference -hierarchy > [file join $REPORT_DIR reference.rpt]

if {$RUN_MODE eq "elab"} {
    write -format ddc -hierarchy -output [file join $ARTIFACT_DIR elaborated.ddc]
    puts "=== ELAB DONE: $TOP_MODULE ==="
} elseif {$RUN_MODE eq "compile" || $RUN_MODE eq "compile_ultra"} {
    if {[sizeof_collection [get_ports -quiet clk]] == 0} {
        error "Top module $TOP_MODULE has no clk port"
    }

    create_clock -name clk -period $CLK_PERIOD [get_ports clk]

    if {[sizeof_collection [get_ports -quiet rst_n]] > 0} {
        set_ideal_network [get_ports rst_n]
        set_false_path -from [get_ports rst_n]
    }

    set non_clock_inputs [remove_from_collection [all_inputs] [get_ports clk]]
    if {[sizeof_collection $non_clock_inputs] > 0} {
        set_input_delay $IO_DELAY -clock clk $non_clock_inputs
    }
    if {[sizeof_collection [all_outputs]] > 0} {
        set_output_delay $IO_DELAY -clock clk [all_outputs]
    }

    if {$RUN_MODE eq "compile_ultra"} {
        compile_ultra
    } else {
        compile
    }

    report_timing -delay_type max -max_paths 20 -nworst 5 > [file join $REPORT_DIR timing_setup.rpt]
    report_constraint -all_violators > [file join $REPORT_DIR constraint_violators.rpt]
    report_area -hierarchy > [file join $REPORT_DIR area.rpt]
    report_power > [file join $REPORT_DIR power.rpt]
    report_qor > [file join $REPORT_DIR qor.rpt]
    report_resources -hierarchy > [file join $REPORT_DIR resources.rpt]
    report_clock > [file join $REPORT_DIR clocks.rpt]

    # Normalize identifiers before emitting Verilog. Without this step DC may
    # invent SYNOPSYS_UNCONNECTED_* nets and report VO-11 warnings, which makes
    # the mapped netlist unnecessarily noisy for downstream tools.
    change_names -rules verilog -hierarchy
    write -format verilog -hierarchy -output [file join $ARTIFACT_DIR mapped.v]
    write -format ddc -hierarchy -output [file join $ARTIFACT_DIR mapped.ddc]
    write_sdc [file join $ARTIFACT_DIR constraints.sdc]
    puts "=== SYNTHESIS DONE: $TOP_MODULE ==="
} else {
    error "Unsupported SYN_RUN_MODE: $RUN_MODE"
}

set manifest [open [file join $RUN_DIR manifest.txt] a]
puts $manifest "finished_at=[clock format [clock seconds] -format {%Y-%m-%dT%H:%M:%S%z}]"
close $manifest

puts "Reports   : $REPORT_DIR"
puts "Artifacts : $ARTIFACT_DIR"
exit
