# Documentation index

Start with [RESULTS_EVIDENCE.md](RESULTS_EVIDENCE.md) for current results and
verification. It is the maintained explanation of the measurements; the linked
provenance and report excerpts are the underlying evidence.

## October 4 four-state coverage and activity analysis

The added VCS bench checks the unchanged RTL: 90 invocations / 186 exact
transactions across all six lane/dimension profiles. [Coverage scope and gaps](COVERAGE_POWER.md),
[metrics](analysis/2026-10-04/coverage.json) and
[source/transaction hashes](analysis/2026-10-04/coverage_verification.json)
are separate from the earlier Verilator sweep and synthesis provenance.
Licensed run databases and raw waveforms stay outside the public repository.

## Latest shared-dequantizer experiment

| Item | Authoritative location / identity |
|---|---|
| Original clean measured RTL | `d5f9123fbbf73154073b28d683ea4bb88b9e3571` |
| Equivalent source after history consolidation | [dd68964](https://github.com/szuwei-yeh/flashattn-accelerator/commit/dd6896461fe19c974b7df1bd366a03a2a0d0a11a); identical measured snapshot |
| Matched N64/d16 integrated tops | 256 / 32 / 16 lanes; same source, library, 10 ns clock and compile flow |
| Area/timing/cycle tradeoff and commands | [DEQUANT_EXPERIMENT.md](DEQUANT_EXPERIMENT.md) |
| New mapped provenance and hashes | [2026-10-03/provenance.json](evidence/2026-10-03/provenance.json) |
| Exact behavioral sweep and other checks | [verification.json](analysis/2026-10-03/verification.json); 312 invocations / 624 transactions |
| Canonical causal/noncausal latency counts | [cycles.csv](analysis/2026-10-03/cycles.csv) |
| Same-quantized-input float64 error analysis | [NUMERICAL_ACCURACY.md](NUMERICAL_ACCURACY.md); 52 cases |

The default remains 256 lanes. The new PPA is **top-only, N64/d16**; d64 is
functionally verified. It does not supply a newly mapped standalone core, and
must not be paired with the earlier core area to calculate DMA overhead.
Dates name run series; source and report hashes identify measured bytes.

## October 3 revision identities

The October 3 commits were consolidated into architecture documentation,
shared-dequantizer implementation, verification, and results publication.
The snapshots below retain exactly the same tracked files and Git tree hashes;
only their commit identities and parent histories changed.

| Original snapshot | Equivalent current commit | Exact Git tree |
|---|---|---|
| Architecture `fbfa0c8` | [9165cbc](https://github.com/szuwei-yeh/flashattn-accelerator/commit/9165cbc52d97e7c4b02b1336df33387fd7ac07e4) | `418105ea45359d1d7849b67af1b71d2996a2948e` |
| Measured implementation `d5f9123` | [dd68964](https://github.com/szuwei-yeh/flashattn-accelerator/commit/dd6896461fe19c974b7df1bd366a03a2a0d0a11a) | `48bc3a68bf2061f4127e9fc3fe63aed126659d64` |
| Recorded CI `068a33f` / consolidated verification `7a50b78` | [6f5d45e](https://github.com/szuwei-yeh/flashattn-accelerator/commit/6f5d45e740a8aa67127c7de3a6605fcb1f92ae1d) | `c3ca2c41bc1ba11a5ee1322ef8b0f5ad66d6d34a` |

Measurement manifests and recorded CI identities retain their original values.
Use `dd68964` to check out the equivalent measured source; use the published
source hashes to verify its contents. Reports, fixtures, RTL and test inputs
were not changed by this consolidation. Earlier commits through `9a66742`
retain their original identities. The final publication also adds this index
and revision links to the guides; its implementation files are unchanged.

## Preserved 256-lane standalone-core/top pair

| Item | Authoritative location / identity |
|---|---|
| Measured and verified RTL | `dfbd28e6262f9c40b5ff840557a4c9d1957b7606` |
| Initial publication of this evidence | `cd99a09`; subsequent documentation commits may have a newer HEAD |
| Preserved results and limitations | [RESULTS_EVIDENCE.md](RESULTS_EVIDENCE.md) |
| Source, library, report, netlist and verification hashes | [2026-09-30/provenance.json](evidence/2026-09-30/provenance.json) |
| Accepted mapped reports | [Core](evidence/2026-09-30/core.txt), [DMA top](evidence/2026-09-30/top.txt) |
| Verification scope | [Verification summary](evidence/2026-09-30/verification.txt) |
| Project overview and quick start | [Repository README](../README.md) |
| Architecture, arithmetic and configuration | [Design guide](DESIGN.md) |
| Operating contracts and simulation/formal commands | [Verification guide](VERIFICATION.md) |
| Synthesis settings and reproduction | [Synthesis guide](../syn/README.md) |
| Fixed-point vs float64 arithmetic error | [Numerical accuracy](NUMERICAL_ACCURACY.md) |
| Optional 32/16-lane score hardware | [Shared dequantizer experiment](DEQUANT_EXPERIMENT.md) |

The preserved September commit IDs were translated after repository history cleanup.
The measured source, library and report hashes are unchanged.

The October shared-dequantizer measurements above are a separate comparison.
The September hashes do not describe every subsequent RTL checkout.

`2026-09-30` names the final run series. It includes the DMA address-range guards.
Accepted runs are `final_20260930_n64d16/core` and
`final_20260930_n64d16_retry/top`. The first top attempt was OOM-killed; its output
is not an accepted result. Directory dates and publication commits are not
substitutes for the measured source hashes.

## Historical evidence

| Location | Purpose |
|---|---|
| [2026-09-24/provenance.json](evidence/2026-09-24/provenance.json) and adjacent core/top reports | Earlier mapped baseline; superseded for current guarded RTL timing/area |
| [historical_shared_scale.txt](evidence/2026-09-24/historical_shared_scale.txt) | Original matched N64/d16 comparison supporting the 7.35% shared-scale area reduction |
| [verification.txt](evidence/2026-09-24/verification.txt), [verification_followup.txt](evidence/2026-09-24/verification_followup.txt) | Earlier verification milestones, preserved in their original scope |

Historical evidence remains useful for its named experiment. Do not merge its
numbers with the final baseline into a new optimization comparison.

## Future updates

Keep new raw logs and netlists in run bundles; publish portable excerpts and
hashes. For a new accepted baseline, add its evidence directory and update this
index plus `RESULTS_EVIDENCE.md` and the main README. Retain prior evidence intact.
