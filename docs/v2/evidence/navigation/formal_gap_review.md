# Historical navigation supplement

This is a navigation-only derivative of the accepted [bounded SoC qualification text](../verification/formal_gap_review.txt). The [earlier original](../results/formal_gap_review.txt), original hashes and proof scope remain unchanged. This supplement records no new test or proof.

> Navigation supplement for the bounded SoC qualification. The original accepted extract and SHA256 remain in the immutable source map. Links below resolve to included sources, historical logs/receipts or explicitly identified public summaries. No proof depth, result or measurement value was changed.

# Targeted safety evidence and remaining gaps

| Property | Established evidence | Remaining scope |
|---|---|---|
| Host output ownership until ACK | Fresh two-state CSR/actual d16 adapter tests; early-owner-release simulator mutant rejected; 12-step adapter assertions/cover | Arbitrary illegal internal owner withdrawal in real SoC remains untested; no waiver |
| Writeback conservation | Fresh adapter tests inject wrong words/bytes/AW/B/OKAY counts; bounded control assertions and conservation-fault cover | Actual AXI payload/channel conservation is not proved by abstract counter inputs |
| Last successful B before DONE/IRQ | Fresh genuine four-state d16 warm/prefill exact capture monitors; premature-DONE Verilator mutant rejected; historical component qualification/frozen d16 PPA baseline/current SoC stalls/fault/reset monitors | CSR IRQ is outside new formal; no mapped equivalence, gate simulation or physical reset signoff |
| Atomic K/V append publication | Historical genuine VCS K-only/K-V gap/V/pre-/post-publication reset and changed-data recovery; 24-step cache metadata proof | Proof assumes complete pairs and legal host inputs; prefill commit/capacity64 not reached at proof bound |
| Reset/fault recovery | Historical integrated VCS reset/fault campaigns and fresh CSR reset-abort tests | Fatal read/write AXI errors still require common reset; no physical DRAM rollback or independent AXI recovery |

## Activity capture and bounded control checks

Both compact0/1 original CSR/adapter suites passed lint, elaboration/build and real Verilator simulation: **21,682 checks /20 jobs each**. Mock child and writeback completions are explicit; these are not SoC inference tests. Logs/results: [baseline](../verification/csr_c0_01.json), [optimized](../verification/csr_c1_01.json). Two copied-DUT mutants fail for [premature DONE](../verification/csr_control_done_01.json) and [early owner release](../verification/csr_control_owner_01.json). Original functional RTL remains unchanged.

Accepted activity capture04: genuine VCS four-state, actual frozen d16 PPA baseline d16/Q4:KV2 SoC, lanes16/256, compact1, modeled AXI latency0, population0 noncausal, initial internal SRAM X. Each runs N16 prefill followed by warm L16 decode, all valid external-memory words independently exact checked: **four successful commands /2,176 exact words total**. Host final-B/store/ownership, real cache/DMA and score assembly monitors remain enabled. [Measured cycles/traffic](../verification/rtl_measurements.csv), [identity/accounting](../verification/rtl_measurements.json). This is a narrow fresh sample, not a fresh full regression or append/reset campaign. Earlier captures and failed power-quality attempts remain preserved.

New bounded host-control scope: [12-step proof/assumptions](../verification/host_formal_contract.txt). Twenty-four-step attempt timeout and full integrated formal remain INCONCLUSIVE. No dependencies installed.

## Historical source-identical evidence retained

[component qualification](../verification/component_verification.json) (historical verification summary): 1,072 successful commands /318,720 exact words,126 fault cases,66 reset checkpoints, selected append/final-store/B-response boundaries and five real-VCS negative-control types. [frozen d16 PPA baseline](../verification/d16_verification_provenance.json) (historical verification summary) verifies both d16 lane variants; [current SoC](../results/soc_provenance.json) (historical verification summary) verifies d64 and d16 wrappers. The current SoC d64 VCS campaign:1,098 commands /1,317,888 words,126 faults,78 resets. The current SoC qualification also ran full component and V1 regressions; the later activity-power experiment preserves their source hashes and results, and does **not** relabel it as a new run.

## Coverage is PARTIAL, not waived

The actual [component qualification URG reports](../results/controller_coverage.json) (historical verification summary) merge six positive runs per traversal: DUT line92.82%, FSM53.85%, branch85.67/85.68%, condition80.15/80.13%, toggle82.41%. Defined functional-group aggregate95.83% is the average of three groups; CSR zero-response-stall bin is reachable but untested. Procedural monitors are not instrumented SVA coverage; the integrated SVA statistics scope is empty.

[Gap inventory](../verification/coverage_gap_inventory.json) retains all312 missed FSM report rows,156 per variant:97 reset/error-recovery-untested,44 structurally unreachable **by inspection**,7 unsupported-configuration,8 unresolved. Repeated instance views are not unique transitions. No denominator changes or waivers. The frozen d16 PPA baseline and current SoC report their own per-profile coverage in [frozen d16 PPA baseline coverage](../verification/d16_coverage.json) and [current SoC coverage](../results/soc_coverage.json); these are not merged with component qualification. The activity-power experiment gathers no new code/functional/SVA coverage database and claims no percentage improvement.

Priority gaps remain integrated ownership/conservation guards, wider reset/fault phase coverage and legal random host protocols. The fresh module proof and negative controls strengthen the first two guards, but do not waive unhit integrated FSM arcs. Broad 100% coverage or another unbounded integrated proof is not justified for this bounded closure. Before asserting complete verification, use targeted integrated ownership/error guard evidence and an explicit boundary/assumption review.
