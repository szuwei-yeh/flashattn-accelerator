# Documentation index

Start with [RESULTS_EVIDENCE.md](RESULTS_EVIDENCE.md) for current results and
verification. It is the maintained explanation of the measurements; the linked
provenance and report excerpts are the underlying evidence.

## Current baseline

| Item | Authoritative location / identity |
|---|---|
| Measured and verified RTL | `d2bb2044d7a70ef53ac2aa64fe95e732a2abf153` |
| Initial publication of this evidence | `c1902ba`; subsequent documentation commits may have a newer HEAD |
| Current results and limitations | [RESULTS_EVIDENCE.md](RESULTS_EVIDENCE.md) |
| Source, library, report, netlist and verification hashes | [2026-09-30/provenance.json](evidence/2026-09-30/provenance.json) |
| Accepted mapped reports | [Core](evidence/2026-09-30/core.txt), [DMA top](evidence/2026-09-30/top.txt) |
| Verification scope | [Verification summary](evidence/2026-09-30/verification.txt) |
| Project overview and quick start | [Repository README](../README.md) |
| Architecture, arithmetic and configuration | [Design guide](DESIGN.md) |
| Operating contracts and simulation/formal commands | [Verification guide](VERIFICATION.md) |
| Synthesis settings and reproduction | [Synthesis guide](../syn/README.md) |

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
