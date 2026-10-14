# V2 verification and reporting tools

[Reproduction guide](../../docs/REPRODUCIBILITY.md) ·
[Verification contract](../../docs/v2/VERIFICATION.md)

## Verification and integrity

| Tool | Purpose |
|---|---|
| [verify.py](verify.py) | Fresh production SoC lint/build, independent exact outputs, hierarchy and mandatory failure controls |
| [validate_release.py](validate_release.py) | Current manifest-v2 inventory, immutable v1 hashes, exact approved compressed trace, provenance, filelists, links and privacy patterns |
| [test_validate_release.py](test_validate_release.py) | Independent evidence, source/fixture/figure drift, trace authorization, link and inventory controls |
| [soc/fixtures.py](soc/fixtures.py) | Deterministic d16/d64 fixture generator used by verify.py |

Verification needs Verilator 5.046, a C++ compiler/make and Python 3 with NumPy.
The runner uses the caller interpreter by default and never installs dependencies.
Use `verify.py` for public quick starts.
Path-only adaptations to fixtures and mutation controls are recorded in
[layout-v2](../../release/layout-v2.json); golden values and checks are preserved.

## Frozen Qwen activation replay

| Tool | Purpose |
|---|---|
| [qwen/run.py](qwen/run.py) | Integrity-checked snapshot and bounded production SoC replay with exact outputs and failure controls |
| [qwen/fixtures.py](qwen/fixtures.py) | Validate hash-pinned activation data and prepare replay streams using the independent oracle |
| [qwen/test_fixtures.py](qwen/test_fixtures.py) | Frozen-fixture, oracle and corruption controls |

Fixture checks need NumPy; RTL replay additionally needs the verification toolchain
above. No model weights are downloaded. See the
[Qwen guide](../../docs/v2/qwen/README.md) for scope and commands.

## RTL waveform capture and publication

| Tool | Purpose |
|---|---|
| [waveform/capture.py](waveform/capture.py) | Trace an existing verified focus regression in an isolated, instrumented testbench copy |
| [waveform/check_trace.py](waveform/check_trace.py) | Check accepted edges, cache activity, head order, writeback and completion in the bounded VCD |
| [waveform/layouts.py](waveform/layouts.py) | Generate Surfer overview and zoom layouts from the trace and checked events |
| [waveform/prepare_figures.py](waveform/prepare_figures.py) | Crop native screenshots and add title/footer bands while asserting unchanged waveform pixels |

Trace checking and layouts use the Python standard library. Capture needs a prior
verification run and its toolchain; native viewing uses Surfer, and screenshot
processing needs Pillow. See the [waveform guide](../../docs/v2/waveform/README.md)
for the reviewed macOS capture procedure and reproduction boundaries.

## Conceptual and historical figures

| Tool | Purpose |
|---|---|
| [generate_online_softmax.py](generate_online_softmax.py) | Reproduce the README shared QK/PV and online-softmax SVG with RTL reference checks |
| [generate_stateful_dataflow.py](generate_stateful_dataflow.py) | Reproduce the current README stateful-flow SVG with the Python standard library |
| [generate_figures.py](generate_figures.py) | Reproduce eight historical figures into a fresh directory; legacy SoC output is `historical_soc_architecture.svg` |

The conceptual SVGs are explanatory diagrams, separate from captured RTL waveforms.
The online-softmax and stateful-flow generators need only the Python standard
library. Use `--output` to write review copies to a temporary directory.
The eight historical figures additionally need matplotlib; the accepted renderer
was matplotlib 3.7.1. Other versions may render different bytes. That generator
requires a new output directory and cannot overwrite accepted figures.

The current `soc_architecture.svg`, `memory_banking.svg` and
`shared_compute_timeline.svg` under `docs/v2/figures/` are canonical SVG
assets maintained directly. This historical generator does not reproduce them;
its SoC output corresponds to the preserved original artifact. Do not copy
historical outputs over the current assets.

## Historical flow helpers

| Tool | Purpose |
|---|---|
| [soc/run_mutants.py](soc/run_mutants.py) | Targeted RTL mutation controls; retains original Mac-specific requirements; use verify.py for portable checks |
| [power/normalize_activity.py](power/normalize_activity.py) | Preserved activity-header/window helper for the licensed power flow |

The activity helper retains its accepted bytes and original input requirements.
Temporary build, run and review outputs belong outside the public source tree or
in ignored build directories. Checked-in figures and evidence are intentionally
retained, verified publication artifacts; a fresh run does not replace them.
No tool substitutes a new run for historical licensed evidence.
