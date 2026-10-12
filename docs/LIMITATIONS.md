# Verification scope and remaining limitations

[Home](../README.md) · [Results](RESULTS.md) · [Reproduction](REPRODUCIBILITY.md)

## Functional verification is strong; closure is partial

Independent scalar fixed-point references, actual DMA-populated SRAM/cache, full valid external-memory output comparisons and meaningful simulator mutants support the implemented bounded geometry. Recorded current SoC d64 campaigns include Verilator2172 successful commands/2,586,624 exact words and genuine VCS1098 commands/1,317,888 words,126 fault cases and 78 reset checkpoints. d16 integrated VCS and previous component and V1 regression results remain separate campaigns. The accepted curated-source receipts record fresh local two-state quick starts; licensed VCS, formal, synthesis and power are not rerun. [Verification identity](v2/evidence/results/soc_provenance.json)

Coverage is per DUT/profile, not the bench and not a universal closure score. Current SoC d64 stress line96.88%/95.09%, FSM54.77%/54.46% for 16/256 lanes; stateful/profile results differ. [Actual profile metrics](v2/evidence/results/soc_coverage.json). Historical component qualification d16 coverage and uncovered-state triage are a separate source/campaign; do not combine denominators or grant waivers. Procedural monitors/functional groups are not SVA proof coverage. Reachable/error/reset/unresolved FSM gaps remain explicit.

Cache bounded formal PASS24 normalized steps assumes legal accepted commands, noncolliding ownership and commit only after complete K/V; arbitrary SRAM contents, not payload correctness. Prefill commit was not reached in that proof. Host-adapter PASS12 steps is the earlier d16 job adapter with arbitrary child/DMA inputs, first-step reset and three reached covers; no AXI byte stream/cache/CSR/IRQ proof.24-step and integrated attempts timed out INCONCLUSIVE. These are bounded checks, not induction or unbounded integrated proof. [Cache assumptions](v2/evidence/results/cache_formal.json),[host contract navigation supplement](v2/evidence/navigation/host_formal_contract.md),[gap review navigation supplement](v2/evidence/navigation/formal_gap_review.md).

Directed tests/monitors establish final output retirement and final successful B before DONE/IRQ, atomic append publication, common-reset boundary safety and changed-data recovery. They do not prove all timings/fault interleavings. Fatal read/write AXI error needs common reset with endpoint; no timeout, independent outstanding-drain recovery, cache physical rollback or external-memory write rollback.

## Architecture and numerical limits

Host4Q:2KV, serial heads, d16/d64, N16 prefill, decodeL16/17/18 latest-position query, capacity 64. Other single-head lengths are earlier separate-core scope. No arbitrary batch/head/sequence shape, general host variable-length decode, FP16/BF16, tapeout or production deployment. Fixed-point softmax saturation/LUT/rounding,8-bit exponent weights and 32-bit output wrap can lose precision; exact oracle equality is not floating-point equivalence. [Preserved numerical study](NUMERICAL_ACCURACY.md)

## PPA and public-package limits

d16 PPA is DC pre-layout standard-cell only, logical SRAM/ROM blackboxes, idealized timing constraints, retained nonzero-limit capacitance violations. No P&R, physical reset/electrical signoff, mapped formal equivalence, achieved Fmax or memory-inclusive die area. d64 matched PPA unavailable due resource limit. Activity-based power excludes physical memory, routed parasitics, clock tree and system energy; default/propagated annotation categories are disclosed. One warm workload is not a workload-general energy claim.

The bounded qualification supplies a portable interpreter-selected runner and a V2 CI matrix. [Ubuntu Verification run 38274178267](https://github.com/szuwei-yeh/flashattn-accelerator/actions/runs/38274178267) passed all 12 jobs for commit `b0ee70d6565f85698d196f6f1c3abe9b498f8196`; later documentation-only revisions require their own CI result. The broader local campaign used macOS with Verilator5.046 and Intel Python3/NumPy. Historical receipts retain their capture-time CI statuses. The targeted mutation runner retains its original Mac-specific requirements. See [bounded SoC qualification scope and actual-interface checks](v2/VERIFICATION.md). Licensed tools/libraries, large raw artifacts and full experimental history remain preserved locally. The curated package includes compact reports/identities, not all coverage databases, netlists or private original activity. No claim of full private-archive reconstruction from the small release alone.

The public package retains V2 properties and recorded proof contracts, but original qualified formal harnesses and scripts are not all bundled. These receipts do not promise a complete public proof rerun. See [reproduction boundaries](REPRODUCIBILITY.md#licensed-and-formal-rerun-boundaries).
