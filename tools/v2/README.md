# V2 verification and reporting tools

[Reproduction guide](../../docs/REPRODUCIBILITY.md) ·
[Verification contract](../../docs/v2/VERIFICATION.md)

| Tool | Purpose |
|---|---|
| [verify.py](verify.py) | Fresh production SoC lint/build, independent exact outputs, hierarchy and mandatory failure controls |
| [validate_release.py](validate_release.py) | Current manifest-v2 inventory plus immutable v1 protected hashes, provenance, filelists, links and privacy patterns |
| [test_validate_release.py](test_validate_release.py) | Independent missing-evidence, source/fixture/figure drift, link and inventory controls |
| [generate_online_softmax.py](generate_online_softmax.py) | Reproduce the README shared QK/PV and online-softmax SVG with RTL reference checks |
| [generate_stateful_dataflow.py](generate_stateful_dataflow.py) | Reproduce the current README stateful-flow SVG with the Python standard library |
| [generate_figures.py](generate_figures.py) | Reproduce eight accepted figures into a fresh output directory |
| [soc/fixtures.py](soc/fixtures.py) | Deterministic d16/d64 fixture generator used by verify.py |
| [soc/run_mutants.py](soc/run_mutants.py) | Targeted RTL mutation controls; retains original Mac-specific requirements; use verify.py for portable checks |
| [power/normalize_activity.py](power/normalize_activity.py) | Preserved activity-header/window helper for the licensed power flow |

Verification needs Verilator 5.046, a C++ compiler/make and Python 3 with NumPy.
The runner uses the caller interpreter by default and never installs dependencies.
Use `verify.py` for public quick starts. The activity helper retains its accepted bytes and original input requirements.
Path-only adaptations to fixtures and mutation controls are recorded in
[layout-v2](../../release/layout-v2.json); golden values and checks are preserved.

The online-softmax and stateful-flow generators need only the Python standard
library. Use `--output` to write review copies to a temporary directory.
The eight historical figures additionally need matplotlib; the accepted renderer
was matplotlib 3.7.1. Other versions may render different bytes. That generator
requires a new output directory and cannot overwrite accepted figures.

All generated outputs belong outside the public source tree or in ignored build
directories. No tool substitutes a new run for historical licensed evidence.
