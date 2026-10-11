# Portfolio release scope

[Overview](../README.md) · [Reproduction](REPRODUCIBILITY.md) · [Limits](LIMITATIONS.md) · [History](HISTORY.md)

The V2 package is a bounded RTL/ASIC portfolio, with V1 preserved as its foundation.
Public source identity is Git HEAD plus any reported working-tree changes.
Measured implementation identity comes from accepted source SHA256s, independently
of later documentation or tooling commits. The primary branch is `main`;
local commits and history consolidation do not imply GitHub publication.

The historical release receipt records **CONDITIONAL PASS for the bounded
Mac-validated portfolio scope**. The current layout is locally validated and
registered in manifest-v2. Linux/V2 Actions validation of this revision remains
pending; open closure and physical limitations remain part of the release claim.

The current [inventory](../release/manifest-v2.json), functional [layout mapping](../release/layout-v2.json),
original [path map](../release/path-map-v1.json) and [evidence lock](../release/evidence-lock-v1.json)
distinguish the current payload from immutable historical manifests.
The original [manifest-v1](../release/manifest-v1.json) is preserved byte-for-byte.
The current inventory excludes only itself to avoid a
self-referential hash; every other public file is hashed. Strict validation checks
the complete payload, modes, functional inputs, accepted evidence/provenance,
filelists, figures, historical link mappings and privacy patterns.
[Original curation verification](../release/verification-v1.json) is a separate historical receipt.

## Current functional-layout checks

On macOS, fresh d16/16 and d64/16 SoC builds each passed eight executions at
modeled latency0, seeds0/17 and final-B hold37 across prefill/focus/warm/append_fault.
The d16 result is86 successful commands/42,496 exact words; d64 is86 commands/
169,984 words. Each rejected wrong-final-word and missing-fixture controls with
nonzero exit and the specific detector. These are bounded two-state observations,
not a new full four-state, coverage or formal campaign.

All272 implementation/verification inputs preserve their original contents or
recorded reversible path-only adaptations; all56 production RTL source files are
byte-identical. Renamed filenames use narrowly scoped DECLFILENAME lint rules
while original module identifiers remain intact; assertions and numerical checks
are unchanged. Fresh V1 N64/d16 core/top causal/noncausal smoke passed four
invocations, with top latency models0/20/100. Eight accepted SVGs reproduced
byte-for-byte; the new stateful-flow SVG also reproduced deterministically.
Strict release validation and17 corruption/default-inventory tests pass locally.
No licensed EDA, mapping or physical validation was rerun.

## Recorded representative checks for the published layout

All four profiles freshly rebuilt unchanged production RTL and functional benches.
At modeled latency0, seeds0/17 and final-B hold37, each ran
prefill/focus/warm/stress/append_fault. Faults and reset checkpoints are separate
from successful command and exact-word counts.

| Dimension / lanes | Successful commands | Exact words | Fault cases | Reset checkpoints |
|---|---:|---:|---:|---:|
| d16 / 256 | 340 | 103168 | 42 | 18 |
| d16 / 16 | 364 | 111872 | 42 | 26 |
| d64 / 256 | 340 | 412672 | 42 | 18 |
| d64 / 16 | 364 | 447488 | 42 | 26 |

The 40 positive executions total 1408 commands / 1075200 exact words, plus
168 fault cases and 88 reset checkpoints. Eight mandatory simulator controls
were rejected. A separate wrong-expected-word invocation exercised the normal
runner failure path and exited nonzero.

Four additional canonical warm executions reused the freshly built profile
binaries with seed0/final-B hold0: d16 cycles1228/1292, d64 cycles4210/4272.
These add 20 commands / 21760 words and are not folded into the table above.
V1 N64/d16 core/top causal/noncausal smoke, representative reference/runner
checks, two Icarus four-state units and four corrupted/missing-fixture controls
also passed. Fourteen release-integrity tests passed; all eight accepted SVGs
reproduced byte-for-byte with matplotlib3.7.1. No licensed EDA or formal rerun
was performed.

## Evidence boundaries

- Current SoC supports d16/d64, fixed 4Q:2KV and bounded host operations.
- Accepted d16 synthesis compares two lane configurations of the same frozen
  d16 PPA baseline source. It does not establish mapped equivalence for the current integration.
- Historical bounded SoC qualification actual-SoC tests passed 120 invocations / 4224 successful commands /
  3225600 exact words, with 504 fault cases, 264 reset checkpoints and eight
  rejected controls counted separately. [Recorded campaign](v2/evidence/verification/verification.json)
- Historical four-state and bounded formal receipts remain distinct from new
  two-state representative curation tests.
- Linux/V2 Actions validation of this revised layout remains NOT RUN. Historical
  receipts keep their capture-time CI status. Full integrated formal/coverage
  closure and mapped equivalence remain open.
- d64/16 mapping is resource-limited; d64/256 mapping was not run; matched d64 PPA
  is unavailable. No physical/electrical/reset signoff or achieved Fmax is claimed.

Successful host DONE/success IRQ requires final successful AXI B and output
retirement. Fault/reject IRQ is distinct. Fatal AXI recovery uses common reset
with the endpoint; external writes are not rolled back.
[Operating contract](v2/VERIFICATION.md)

## Historical navigation and formatting

Historical receipts retain capture-time statuses and original path strings.
[Evidence navigation](v2/evidence/README.md) supplies current entrypoints and
labeled navigation supplements. Strict validation resolves moved text using its
immutable original context and exact path mapping; it cannot excuse a changed
receipt or missing replacement.

Six accepted formatting exceptions are preserved byte-for-byte: CRLF in
`docs/v2/evidence/verification/rtl_measurements.csv` and
`docs/v2/evidence/verification/verification_matrix.csv`, and historical Yosys formatting in
`host_proof.log`, `host_cover_success.log`, `host_cover_owner_fault.log` and
`host_cover_conservation_fault.log` under `docs/v2/evidence/verification/`.
New edits must be whitespace-clean; these historical exceptions are not generalized.

Distribution remains [portfolio review; reuse licensing pending](../DISTRIBUTION.md).
No project-level open-source reuse license is granted. Existing attribution,
history and third-party notices retain their scope.
