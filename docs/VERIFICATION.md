# Verification and operating contracts

[Home](../README.md) · [Design](DESIGN.md) · [Results and evidence](RESULTS_EVIDENCE.md)

The preserved mapped baseline is tied to RTL `dfbd28e`. See the
[verification record](evidence/2026-09-30/verification.txt) for its scope and
[provenance](evidence/2026-09-30/provenance.json) for hashes. The commands below
are reproduction entry points. The newer shared-dequantizer implementation has a
[separate completed behavioral record](analysis/2026-10-03/verification.json):
312 exact invocations / 624 transactions across 256/32/16 lanes and d16/d64,
plus a full default regression, six core corner cases, shared top contract tests,
unit-level four-state tests and bounded controller formal. Its measured source
identity is recorded separately from the earlier mapped PPA.

## Functional and protocol checks

Canonical exact regression covers N64 d16/d64. Core counts are 7,589 / 28,337;
DMA-prefetch counts are 7,800 / 29,160 at modeled latency zero. The DMA tests also
exercise latency 20 and 100. Directed checks cover configuration locking,
invalid configuration rejection, busy starts, causal operation, and reset/restart.
Optimized causal core/top regression includes both d16 and d64; the top runs
each at modeled read latencies 0/20/100. The N64/d16 contract test also resets
mid-output update after eight SRAM writes, changes V, then checks the restarted
transaction exactly. This is one directed abort point, not an all-phase reset sweep.

The Python reference models signed scales and INT16 dequantizer saturation.
Additional extreme-value full-core cases exercise saturation and negative scales
for both dimensions. Existing checked-in canonical expected files are preserved.

**Output memory:** first-K/V-tile updates explicitly overwrite old output state.
SRAM itself has no reset. A four-state test, checked under Icarus and VCS, covers unknown initial contents,
stale-data overwrite, reset/restart, 32-bit wrap, and signed normalization.
The optimized interface remains one transaction per reset.

**AXI:** the vector DMA splits bursts at 4 KiB boundaries and checks each accepted
RRESP/RLAST. It rejects descriptors whose source span exceeds the address space
or destination span exceeds the 4096-byte scratchpad. The optimized top rejects
wrapping Q/K/V matrix spans through `cfg_error` before accepting start, including
K/V spans split into separate descriptors. A span ending exactly at the final
legal byte is accepted. Invalid descriptors or bad responses latch `error`; the optimized
top exposes `dma_error`, suppresses successful `done`, and does not promote a
failed tile as resident. The offending stripe is not written. Partial results
are invalid. Recovery requires **common reset of master and slave**; there is
no timeout, outstanding-burst draining, or independent slave recovery. Legacy
non-prefetch wrappers do not expose the new host error status.

A directed scoreboard checks AR payload stability during backpressure, RVALID
gaps, multi-burst data conservation, a stripe crossing two bursts, error handling,
and reset recovery. Real integration monitors check loader stripe count/order,
active-bank ownership, and complete matching shadow promotion.

**Formal:** controller safety uses Yosys/SMT/Z3 BMC to depth 48 and reachability
covers. The harness abstracts datapath/loader completions. It checks resident
matching active tiles, valid shadow promotion, index bounds and loader request
exclusion. It is not an unbounded proof or a proof of SRAM/arithmetic contents.
The integration monitors above are simulation checks, not additional datapath
formal proofs.

## Reproducing checks

[GitHub Actions](../.github/workflows/ci.yml) runs Python reference/accuracy,
synthesis-parameter and runner checks; canonical noncausal/causal core/top smoke
tests for all six d16/d64 × 256/32/16-lane configurations; default-path AXI/control
and four-state output checks; and shared-dequantizer arithmetic, four-state
initialization and top configuration/error contract checks. Top smoke tests
include read latencies 0/20/100. Logs and a fresh accuracy report are artifacts.
CI is a subset of the full commands below; it does not run licensed synthesis,
VCS or bounded formal. Its badge is live, while historical tables are recorded.
CI builds and caches Verilator 5.046 from a pinned source commit; Ubuntu 24.04's
packaged 5.020 cannot elaborate the existing nonblocking array loops.

The optional variants have a [separate sweep](DEQUANT_EXPERIMENT.md) that checks
all generated numerical-analysis cases against the RTL. The [accuracy guide](NUMERICAL_ACCURACY.md)
defines the floating-point comparison and its limits.

Install Verilator, a C++ compiler, Python 3 with NumPy, and Icarus Verilog for the
four-state check. Formal additionally needs Yosys, Z3 and SymbiYosys.
The current RTL is tested with Verilator 5.046. Use a checkout path without spaces.

```bash
make -C sim/verilator regression
make -C sim/verilator audit_fixes
python3 golden/test_hw_reference.py
# Uses the d16/d64 core binaries built by regression; generates only temp files.
python3 golden/check_rtl_corners.py
python3 syn/scripts/test_run_metadata.py
make -C sim/verilator test_runners
make -C formal all
```

Selected targets:

```bash
make -C sim/verilator dma_vec_bench dma_bench
make -C sim/verilator core_banked_prefetch_N64 core_banked_prefetch_N64_d64
make -C sim/verilator dma_banked_prefetch_top_N64 dma_banked_prefetch_top_N64_d64
make -C sim/verilator core_banked_prefetch_causal_N64_d64 dma_banked_prefetch_causal_top_N64_d64
make -C sim/verilator tb_dma_vec_axi_protocol tb_dma_banked_prefetch_contract
make -C sim/verilator tb_oracle_checks
make -C sim/verilator numerical_checks
```

`regression` also runs the standalone 16×16 array-controller test, LUT sweep,
runner failure-propagation checks, and `tb_oracle_checks`. The latter exercises
the real optimized core/top binaries with mismatched geometry, missing/short
vectors, overflowing hex values, extra/partial tokens, invalid scales, and
one-bit errors in the first/last expected output.
It also corrupts LUT entries and builds a temporary loader with `done` suppressed
to verify that bad results and missing completion fail the test. All mutations
stay in temporary directories. Optimized E2E test geometry is tied to the RTL
build parameters, and all three scale values are required fixture inputs.

`numerical_checks` is also part of regression. It checks 40 temporary fixtures
across d16/d64 core and DMA top (160 transactions including latency 0/20/100),
without importing the hardware golden model. Closed-form cases cover equal
scores, saturation, signed/extreme V scales and causal prefix sums. Exact
permutation checks cover Q rows, paired K/V rows within tiles, and feature
columns across d64 chunk boundaries. Cross-tile K/V permutation is not asserted
bit-identical because fixed-point online rescaling can depend on tile order.

The licensed VCS four-state check is optional:
`make -C sim/verilator vcs_output_buffer_init VCS=/path/to/vcs`. Its runner
requires a fresh explicit PASS, a zero exit status, and no failure diagnostics;
VCS process status alone is insufficient after `$fatal`.

`make coverage` is a compatibility entry point for scenario tests. It builds and
runs its listed test targets, propagates failures, and prints a success summary
only when all prerequisites succeed; it does not report measured code/functional
coverage or a fabricated aggregate case count. Four-state, Python checks and
formal remain separate commands above. Full raw reports stay outside the repository;
public tables state the run/configuration boundaries needed
to interpret claims.
