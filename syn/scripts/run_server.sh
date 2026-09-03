#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  syn/scripts/run_server.sh <profile> <run-tag> [elab|compile|compile_ultra]

Profiles:
  systolic   Pure-logic 16x16 systolic array
  dma        Pure-logic vector read DMA
  loader     Pure-logic banked tile loader
  core       Banked-prefetch core with logical SRAM/ROM blackboxes (recommended)
  core_rtl   Banked-prefetch core with behavioral memories (elab/debug only)
  top        DMA + banked-prefetch core with logical SRAM/ROM blackboxes
  top_rtl    DMA + banked-prefetch core with behavioral memories (elab/debug only)

Required server environment:
  DC_TARGET_LIBRARY=/absolute/path/to/standard_cell.db

Optional environment:
  DC_SHELL_BIN=dc_shell       # executable name/path
  CLK_PERIOD=10.0             # ns
  IO_DELAY=1.0                # ns
  ELAB_PARAMETERS='HEAD_DIM=64,SEQ_LEN=64'

Example:
  DC_TARGET_LIBRARY=/path/to/gscl45nm.db \
    syn/scripts/run_server.sh core 2026-09-02_fused compile
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

profile=$1
run_tag=$2
run_mode=${3:-compile}

if [[ ! $run_tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: run-tag may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi

case $run_mode in
    elab|compile|compile_ultra) ;;
    *)
        echo "ERROR: unsupported mode '$run_mode'." >&2
        usage >&2
        exit 2
        ;;
esac

case $profile in
    systolic)
        top_module=systolic_array
        filelist=syn/filelists/systolic_array.f
        ;;
    dma)
        top_module=dma_engine_vec
        filelist=syn/filelists/dma_engine_vec.f
        ;;
    loader)
        top_module=banked_tile_loader
        filelist=syn/filelists/banked_tile_loader.f
        ;;
    core)
        top_module=flash_attn_core_banked_prefetch
        filelist=syn/filelists/core_banked_prefetch_macro.f
        ;;
    core_rtl)
        top_module=flash_attn_core_banked_prefetch
        filelist=syn/filelists/core_banked_prefetch.f
        ;;
    top)
        top_module=flash_attn_top_dma_banked_prefetch
        filelist=syn/filelists/top_dma_banked_prefetch_macro.f
        ;;
    top_rtl)
        top_module=flash_attn_top_dma_banked_prefetch
        filelist=syn/filelists/top_dma_banked_prefetch.f
        ;;
    *)
        echo "ERROR: unknown profile '$profile'." >&2
        usage >&2
        exit 2
        ;;
esac

if [[ -z ${DC_TARGET_LIBRARY:-} ]]; then
    echo "ERROR: DC_TARGET_LIBRARY must point to a server-side .db library." >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
run_dir="$repo_root/syn/runs/$run_tag/$profile"
dc_shell_bin=${DC_SHELL_BIN:-dc_shell}

if ! dc_shell_path=$(command -v "$dc_shell_bin"); then
    echo "ERROR: cannot find '$dc_shell_bin' in PATH." >&2
    exit 127
fi
if [[ $dc_shell_path == */* && $dc_shell_path != /* ]]; then
    dc_shell_dir=$(cd -- "$(dirname -- "$dc_shell_path")" && pwd -P)
    dc_shell_path="$dc_shell_dir/$(basename -- "$dc_shell_path")"
fi
if [[ ! -f $DC_TARGET_LIBRARY ]]; then
    echo "ERROR: target library does not exist: $DC_TARGET_LIBRARY" >&2
    exit 2
fi
if [[ -e $run_dir ]]; then
    echo "ERROR: run directory already exists; choose a new tag:" >&2
    echo "  $run_dir" >&2
    exit 2
fi

mkdir -p "$run_dir/logs"

export SYN_PROFILE=$profile
export SYN_RUN_TAG=$run_tag
export SYN_RUN_MODE=$run_mode
export SYN_TOP_MODULE=$top_module
export SYN_FILELIST=$filelist
export SYN_CLK_PERIOD=${CLK_PERIOD:-10.0}
export SYN_IO_DELAY=${IO_DELAY:-1.0}
export SYN_ELAB_PARAMETERS=${ELAB_PARAMETERS:-}
export SYN_GIT_COMMIT
export SYN_GIT_DIRTY

SYN_GIT_COMMIT=$(git -C "$repo_root" rev-parse HEAD)
if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
    SYN_GIT_DIRTY=yes
    echo "WARNING: synthesis is running from a dirty worktree." >&2
else
    SYN_GIT_DIRTY=no
fi

log_file="$run_dir/logs/dc_shell.log"
echo "Starting $profile ($run_mode), run tag $run_tag"
echo "Log: $log_file"

set +e
(
    cd -- "$run_dir" || exit 1
    "$dc_shell_path" -f "$script_dir/dc_run.tcl"
) 2>&1 | tee "$log_file"
dc_status=${PIPESTATUS[0]}
set -e

if [[ $dc_status -eq 0 ]]; then
    touch "$run_dir/SUCCESS"
    echo "Synthesis completed successfully: $run_dir"
else
    touch "$run_dir/FAILED"
    echo "Synthesis failed with status $dc_status: $run_dir" >&2
fi

exit "$dc_status"
