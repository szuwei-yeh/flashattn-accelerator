#!/usr/bin/env bash
# Verilator wrapper: appends the scoped lint waiver file (lint_waivers.vlt) to
# every invocation so `make` / `make regression` build clean while the design
# (including the dequant fold) still gets full -Wall linting.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec verilator "$@" "$D/lint_waivers.vlt"
