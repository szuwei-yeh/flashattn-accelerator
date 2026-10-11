# Historical navigation supplement

This is a navigation-only derivative of the accepted [bounded SoC qualification text](../verification/closure_attempts.txt). The [earlier original](../results/closure_attempts.txt), original hashes and proof scope remain unchanged. This supplement records no new test or proof.

> Navigation supplement for the bounded SoC qualification. The original accepted extract and SHA256 remain in the immutable source map. Links below resolve to included sources, historical logs/receipts or explicitly identified public summaries. No proof depth, result or measurement value was changed.

# Attempts and excluded evidence

All failed attempts remain intact; none changes production RTL or accepted mapped bytes.

- PT qualification01: toy ANSI-port Verilog rejected by the legacy parser (SVR-4); qualification02 uses equivalent non-ANSI toy syntax and verifies actual PT/PrimePower license/library. This is tool qualification only.
- Activity01 both lanes: exact outputs PASS, but independent waveform gate found7ns cycles with zero-width high pulses from a new driver replacement error. Excluded from power. Original drivers/logs are retained in each `executed/` snapshot.
- Activity02 both lanes: corrected10ns/50% clock, exact PASS; compile-time `+vcs+dumparrays` is reported ignored. Activity03 lane16 moved the option to runtime but still lacked complete debug access; no new power claim.
- Power lane16 attempt01: point-separated strip path caused PSW-130/no annotated activity. All logs/reports/estimates excluded; source and artifacts unchanged.
- Power lane16 attempt02: slash-separated strip path, actual tool PASS, but file+implied sequential annotation only28.5561%. Rejected by the predeclared95% gate; propagated activity is not relabeled direct. No matched energy claim uses it.
- Activity04 both lanes: uses original V1 flow's `-debug_acc+all` and runtime `+vcs+dumparrays+65536`; exact outputs, timing and source identities pass. Full array contents are now present in the VCD. These are the candidate accepted activity inputs.
- Host formal01: unquoted source paths with spaces caused file lookup failure, before proof.
- Host formal02: corrected quoting; actual24-step SAT timeout at60 solver seconds, INCONCLUSIVE.
- Host formal03: unchanged safety obligations at12 steps, all three covers reachable; bounded module-control PASS. Cover wires have keep attributes solely in the verification overlay. No unbounded or integrated proof.
- Two early report-collection calls were refused while a PT receipt remained RUNNING; no reports were overwritten and no duplicate PT process was launched. Completion collection waits for the final receipt. These orchestration failures supply no measurement evidence.

- Final256 report collection initially timed out during an uncompressed104.8MB VCD/report transfer. The completed PT run was not rerun. Lossless gzip transport succeeds; identical decompressed report/VCD hashes are verified. [Transport failure](../verification/transport_failure.json).

- The initial final link audit checked its two generated manifest targets before emitting them. The exact failure is preserved in [validation attempt01](../verification/validation_attempt01.json). Only those two emitted targets are deferred until generation; all source/hash/link rules remain unchanged. Final validation passes.

Each capture has original/public log hashes. Collected power reports retain a raw-report ledger and executed flow files. Original unsanitized reports, full VCDs, fixtures and source snapshots remain private/server-side; public derivatives redact only location metadata. Negative simulator controls are expected failures and are excluded from positive counts/coverage.
