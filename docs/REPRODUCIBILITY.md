# Reproduction and source identity

[Home](../README.md) · [Results](RESULTS.md) · [Verification](v2/VERIFICATION.md) · [Limits](LIMITATIONS.md)

## Fresh V2 RTL checks

Use Verilator 5.046, a C++ compiler/make and Python 3 with compatible NumPy.
The production filelist, real host/cache/AXI bench, scalar reference, imports and
LUT/data are bundled. The runner stages sources in a temporary path without spaces,
then freshly lints, checks hierarchy and compiles with `-j2`; it reuses no builds.

```sh
python3 -B tools/v2/verify.py --dim 16 --lanes 256 --scopes prefill,focus --latencies 0 --seeds 0,17 --final-b-hold 37 --output-dir /tmp/v2-d16-l256
python3 -B tools/v2/verify.py --dim 64 --lanes 16 --scopes prefill,focus --latencies 0 --output-dir /tmp/v2-d64-l16
python3 -B tools/v2/validate_release.py --output /tmp/v2-release-validation.json
python3 -B tools/v2/test_validate_release.py
```

Use new output directories for every invocation. Both dimensions support 256/16
dequantizer lanes. `focus` runs prefill→append→decode→append→decode without healthy
common resets; `warm` checks changed Q/K/V/scale populations and warm L16 jobs.
`stress` and `append_fault` add bounded fault/reset/capacity/publication checks.
Scope is fixed 4Q:2KV, N16 prefill, host decode L16/17/18, capacity64.

Every positive invocation must also reject wrong-final-word and missing-fixture
controls with the specific detector, nonzero exit and no simulator PASS.
`--test-control expected` or `missing_fixture` deliberately fails the normal
verification path. Timeouts, failed lint/build, missing/duplicate PASS, incorrect
hierarchy or outputs all fail the runner and CI.

On the validated Mac, existing NumPy is Intel-only: replace `python3 -B` with
`arch -x86_64 /usr/local/bin/python3 -B`. The oracle defaults to the caller
interpreter; `--python-command` or `V2_PYTHON` can select another existing launcher.
The fixture generator executes in the caller interpreter, which also needs NumPy.
The runner installs nothing.

## Canonical warm-cycle conditions

```sh
python3 -B tools/v2/verify.py --dim 16 --lanes 256 --compact 1 --scope warm --latencies 0 --seeds 0 --final-b-hold 0 --output-dir /tmp/v2-warm-d16-l256
```

Repeat with d16/d64 and 256/16 lanes, each into a new directory.
Accepted host command-to-DONE cycles are d16 1228/1292 and d64 4210/4272.
The window includes packing/writeback and excludes cache population, CSR setup,
readout and ACK. Timing seeds and extra final-B hold define different experiments;
they are not expected to preserve canonical cycles.
[Canonical historical receipts](v2/evidence/verification/canonical_cycles.json)

## V1 and historical evidence

[V1 verification](VERIFICATION.md#reproducing-checks) uses the unchanged original
runners from a checkout without spaces. For a representative core/top causal and
noncausal sweep:

```sh
python3 -B sim/verilator/run_dequant_sweep.py --dim 16 --lanes 256 --smoke --jobs 2
```

V1 CI history certifies its recorded source/scope, not new V2 execution.
Historical [curated-source quick-start receipts](v2/evidence/results/quickstart_results.json) and
[bounded SoC qualification campaign](v2/evidence/verification/verification.json) retain their original sources,
counts and tool identities. The original curation regressions are separately recorded in
[verification-v1](../release/verification-v1.json); current layout checks are summarized
in [release scope](RELEASE.md#current-functional-layout-checks).
[Source-to-public ledger](v2/evidence/results/source-map.json) keeps original and public
SHA256s; the [original path mapping](../release/path-map-v1.json) plus
[layout-v2](../release/layout-v2.json) identify current locations. The default release
validator checks [manifest-v2](../release/manifest-v2.json) and the immutable v1 evidence lock.

## Figure reproduction

The current README online-softmax and stateful-flow figures use Python's standard
library. Render copies outside the source tree:

```sh
v2_figure_dir="$(mktemp -d)"
python3 -B tools/v2/generate_online_softmax.py --output "$v2_figure_dir/online_softmax.svg"
python3 -B tools/v2/generate_stateful_dataflow.py --output "$v2_figure_dir/stateful_dataflow.svg"
```

The online-softmax diagram distinguishes ideal recurrences from fixed-point RTL,
including masked tails and the raw-denominator/capped-PV-weight distinction.
Its generator checks source reference anchors; it does not prove RTL correctness.

With matplotlib 3.7.1 for accepted byte reproduction:

```sh
python3 -B tools/v2/generate_figures.py --output-dir /tmp/v2-figures-new
```

All eight plots/diagrams use bundled accepted data. Other matplotlib versions may
render different bytes. Outputs and the generated figure manifest go only to the
new directory; accepted SVGs and historical figure manifests are never overwritten.
The complete-archive evidence collector is locally archived, not a public quick start.

## Licensed and formal rerun boundaries

Matched d16 PPA uses the earlier frozen d16 PPA baseline source snapshot, not the current SoC
d16/d64 host integration. Exact frozen source/flow/library/artifact hashes are
preserved in [PPA provenance](v2/evidence/results/ppa_provenance.json).
[Macro filelist](../syn/v2/synthesis/soc_macro.f), [DC flow](../syn/v2/synthesis/dc_soc.tcl),
[PT activity script](../syn/v2/power/pt_power.tcl) and
[activity normalizer](../tools/v2/power/normalize_activity.py) are retained.
Independent PPA/power reruns need licensed DC/PT/PrimePower, the original library,
mapped DDC/netlist/SDC and waveform hydration. The PT watchdog is Linux-specific
and requires an external input descriptor. These are outside the RTL quick start.

V2 formal property/abstraction files are retained. Original qualified harnesses,
Yosys scripts and full proof inputs are not all bundled; historical bounded
results are reviewable evidence, not a promised one-command public proof rerun.
Some earlier single-head/serial GQA comparison benchmark sources likewise remain in the full local
archive. Current SoC RTL is publicly rerunnable; that does not reproduce every
historical experiment or licensed measurement.

Linux/new V2 Actions execution remains UNVERIFIED. The Ubuntu matrix is configured
with Python3.12, NumPy2.2.6 and pinned Verilator5.046; a configuration is not a
completed Actions run. No fresh VCS, formal, synthesis or power run is implied.
