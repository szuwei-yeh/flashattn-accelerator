# Main-path four-state coverage

This complements the Verilator regression with licensed VCS/URG code coverage,
covergroups and true unknown initialization. It uses the same DMA/banked/prefetch
RTL and fixed-point reference; it does not change the single-shot interface.
See [recorded results and coverage gaps](../../docs/COVERAGE_POWER.md).

From a checkout without spaces in its path:

```sh
# Python + NumPy; generated cases never overwrite data/.
python3 sim/vcs/prepare_cases.py /tmp/attention-extra-fixtures

# VCS and URG must be installed, licensed and on PATH.
python3 sim/vcs/run_coverage.py \
  --run-dir "$PWD/sim/vcs/runs/coverage_new" \
  --fixtures /tmp/attention-extra-fixtures --reset-sweep
python3 sim/vcs/summarize_coverage.py \
  sim/vcs/runs/coverage_new /tmp/attention-coverage-export
```

`VCS` and `URG` can override executable commands. The runner is Python 3.6+;
fixture preparation uses the NumPy-based reference and can run on another machine.
Use a **new** run directory. Compiler output, VDBs and HTML reports stay ignored
under `sim/vcs/runs/`. Six profiles are run by default: 256/32/16 lanes × d16/d64.
For a smaller check, append `--lanes 16 --dims 16`.
The export directory must also be new; accepted evidence is never overwritten.

Each profile has canonical causal/noncausal cases at read latencies 0/20/100,
plus optional random, uniform, saturation and signed-scale cases. Each regular
invocation checks the complete output, performs a common reset, changes V to
zero and checks the complete output again. A contract invocation checks rejected
configurations, accepted-input locking, busy/post-done starts, DMA errors,
common-reset recovery and abort during a partial output update.

`--reset-sweep` adds two invocations per profile, causal and noncausal at read
latency 100. Each runs 32 checkpoints for the parallel design or 34 for shared
dequantization: every non-idle controller, scheduler, DMA, array, loader and
softmax state, plus dequantized-score publication and the shared issue/retire
pipeline. The DMA receive checkpoint waits for a retained half stripe; array
compute and loader issue checkpoints wait until work has progressed.

Each checkpoint first completes and checks an old-data job, launches another
job, and waits for the named stage. Reset is asserted between clock edges, and
the bench checks idle states, cleared counters/errors/validity, and absence of
writes or AXI activity both during reset and after release. It then replaces
Q/K/V and accepted scales with the prepared random fixture and checks every
output on the same DUT **without a second reset**. The recovery golden must be
nonzero and differ from the old output; all four fixture files must differ.
The regular cases separately retain changed-V-zero recovery. SRAM is not
cleared by the bench. A common reset also resets
the behavioral AXI slave; independent master-only recovery is outside this test.

Reset covergroup bins are sampled only after complete recovery-output and
accounting checks. The runner requires every named checkpoint exactly once,
ordered prime/hit/recovery markers, matching job identities, and a 100% reset
bin summary. Omitting `--reset-sweep` retains the original 15-invocation profile
when all prepared fixtures are supplied. With the flag, the full experiment is
102 invocations / 986 exact jobs, including 400 targeted reset/recovery cases.

Configuration/data stimulus changes on falling edges; targeted reset assertion
is deliberately between edges. Checks use four-state comparisons and reject
unknown or incomplete fixtures. A nonzero process exit, error/fatal diagnostic,
missing PASS marker or wrong completion identity rejects the run—even if VCS
returns zero after `$fatal`. Functional completion bins are sampled only after
successful full-output and accounting checks. Controller bins observe all 12
states and shadow promotion; they are observations, not a formal proof.

Code coverage is scoped to `tb_attention.dut.u_dut`. Each profile has a separate
VDB/URG report; different elaborations are not merged into a misleading score.
No coverage waivers are applied. Functional group coverage describes only the
bins declared in `tb_attention.sv`; it is not exhaustive specification coverage.

`test_eda_reporting.py` checks failure handling and report parsing without a
license. GitHub Actions runs these Python checks, while licensed coverage stays
a separate recorded experiment. The old `make coverage` target under
`sim/verilator` is a scenario test target, not this measured coverage flow.
