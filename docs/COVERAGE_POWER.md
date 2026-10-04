# Four-state coverage and workload activity

The October 4 experiment adds verification and power-analysis infrastructure
without changing RTL. The mapped inputs remain the October 3 N64/d16 matched
netlists from `d5f9123` (equivalent source snapshot `dd68964`). Original evidence
and the earlier Verilator sweep remain intact.

## Main-path VCS/URG coverage

VCS/URG **V-2023.12-SP2** completed **90 invocations / 186 exact transactions**
across 256/32/16 lanes and d16/d64. All outputs, DMA byte/tile accounting and
cycle-counter checks passed. The 72 canonical transactions in this experiment
have exactly the same cycles as the recorded October 3 sweep. These are additional
checks of the same design, not an expansion to a multiple-jobs-per-reset interface.

| Lanes | Dimension | Line | Condition | Toggle | FSM transitions | Branch | Defined functional groups |
|---|---|---:|---:|---:|---:|---:|---:|
| 256 | d16 | 96.97% | 88.04% | 88.64% | 69.55% | 96.17% | 100% |
| 256 | d64 | 97.11% | 91.77% | 89.88% | 64.09% | 96.24% | 100% |
| 32 | d16 | 97.36% | 90.26% | 82.40% | 69.55% | 97.58% | 100% |
| 32 | d64 | 97.58% | 90.30% | 83.06% | 64.09% | 97.66% | 100% |
| 16 | d16 | 97.41% | 90.26% | 81.41% | 69.55% | 97.71% | 100% |
| 16 | d64 | 97.64% | 90.30% | 81.97% | 64.09% | 97.79% | 100% |

Each profile runs canonical causal/noncausal fixtures at read latencies
0/20/100 and four additional fixture families in both causal modes: random
seed 7, uniform bias, distinct saturated logits and signed scales. All regular
runs complete one original-data job and one changed-V-zero job after a common
reset. Contract runs add invalid lengths/alignment/range, hostile input changes
and starts while busy, ignored post-done starts, RRESP/RLAST faults, a failed
later resident tile, reset recovery and reset during a partial output update.

Fresh SRAM is checked to contain unknown values. Missing/unknown fixture words
are rejected. Every successful job reads and compares all 1,024 or 4,096 output
words with four-state comparisons. Two intentional negative controls—corrupting
the first expected word and truncating the expected file—were rejected by real
VCS execution. Python failure-handling checks also reject zero-exit failures and
missing completion markers.

### Remaining coverage gaps

Coverage includes only the integrated DUT hierarchy; there are no waivers. The
100% functional number means all explicitly declared bins were hit: successful
latency/causal combinations, nine contract events, 12 controller states and
shadow promotion. It does not mean every legal input or fault interleaving was
exercised. Procedural error checks contribute code coverage; no new complete
SVA assertion-coverage claim is made.

URG's FSM score counts transitions; state coverage is reported separately and
is not included in that score. All enumerated controller, scheduler, DMA,
array-controller, loader and softmax states were reached. For example, in the
16-lane/d64 report the tile controller reaches all 12 states but 18/26 transitions,
and softmax reaches all six states but 6/10 transitions. The remaining softmax
transitions return to idle through reset from intermediate states. Reset during
partial output is exercised, but reset at every state is not. The DMA's
`S_IDLE → S_ERROR` guard is not driven by the valid integrated scheduler.

Other gaps include the output buffer's standalone rescale branch (the main path
uses fused rescale/accumulate), forbidden-mode error statements and signal bits
that stay constant under these dimensions/fixtures. These are retained in the
reported percentages. They do not establish unreachable-code proofs or replace
unit tests, protocol checks and the separately recorded bounded formal analysis.

[Commands and bench contracts](../sim/vcs/README.md) ·
[Coverage metrics and report hashes](analysis/2026-10-04/coverage.json) ·
[Transactions and source hashes](analysis/2026-10-04/coverage_verification.json) ·
[Unwaived FSM detail](analysis/2026-10-04/fsm_review.txt) ·
[Intentional failing checks](analysis/2026-10-04/negative_controls.json)
