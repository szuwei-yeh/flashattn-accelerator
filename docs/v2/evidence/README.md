# Public claim → evidence → source boundary

[Results](../../RESULTS.md) · [Raw/public hash map](results/source-map.json) · [Source identities](results/source_revisions.json)

| Public claim | Classification | Exact supporting artifact | Source / scope limitation |
|---|---|---|---|
| Stateful host prefill/append/decode, no common reset |VERIFIED |[current SoC provenance](results/soc_provenance.json), fresh quick-start receipts |current SoC top/filelist, d16/d64,4Q:2KV, N16, host decodeL16/17/18 |
| Physical GQA KV sharing |VERIFIED / MEASURED traffic |[matched serial GQA records](results/gqa_comparison.json) |Matched serial GQA 4:4/4:2, d16 five commands; later writeback excluded; 3456→2304readB, KV2304→1152B |
| Single-head decode 3.12×/3.60× |MEASURED / VERIFIED |[single-head traversal before/after](results/decode_before_after.csv) |L16, pos15, noncausal, no read stalls/gaps, AXI0,256 DQ; 706→226/2659→739; not host-SoC speedup |
| True16-lane sharing |VERIFIED hierarchy / exact outputs |[current SoC source provenance](results/soc_provenance.json), fresh inventory/quick-start |16 actual dequantizer instances vs256; 256 PEs unchanged; ordered complete-tile publication |
| d16 SoC area−58.394905% |MEASURED pre-layout |[PPA CSV](results/ppa.csv),[endpoint raw/flow identities](results/ppa.json),[pair audit](results/pair_audit.json), retained area/QoR extracts |frozen d16 PPA baseline matched source 5b486419; gscl45nm,10 ns,81SRAM/16ROM logical BB; library units, not die area |
|16-lane d16 timing margin |MEASURED pre-layout |[PPA endpoints](results/ppa.json) |slack+0.459291458 ns vs+3.918528557 ns; no achieved Fmax/electrical signoff; retained max-cap2/1 |
| d16 full-SoC sharing latency+5.21% |MEASURED / VERIFIED |[SoC cycle CSV](results/soc_cycles.csv) |warm 4-headL16, compact, AXI0, accepted command→DONE with writeback; 1228→1292 |
| d64 functionality / DC readiness |VERIFIED |[SoC provenance](results/soc_provenance.json),[frozen status](results/d64_status.json) |Exact external outputs and analysis/elaboration/link; no mapped d64 PPA |
| d64 mapping |RESOURCE-LIMITED / NOT RUN |[d64 freeze](results/d64_status.json) |16lane watchdog12 GiB duringPass1Mapping; 256lane not run; no area/timing endpoint |
| d16 total job energy−8.34% |ESTIMATED using MEASURED activity |[power CSV](results/power.csv),[power comparison](results/power.json),[raw reports](results/pt_l256_power.rpt) /[16lane](results/pt_l16_power.rpt),[closure provenance](results/closure_provenance.json) |One warm 4Q2KV/L16, AXI0, compact job; 2.246159→2.058898 µJ; windowhostlatency+1 cycles; physical memory/system excluded |
| Dynamic job energy−0.30% |ESTIMATED, near-neutral |Same power CSV |1.895733→1.889972 µJ; not inferred from fewer cells/cycles; default/propagated annotation limits |
| Genuine integrated VCS |VERIFIED historical campaign |[source/test counts](results/soc_provenance.json),[actual coverage profiles](results/soc_coverage.json) |Four-stateRTL, not gate-level equivalence; curated-source fresh local runs are two-state Verilator |
| Formal safety |VERIFIED bounded / INCONCLUSIVE deeper |[cache 24 steps](results/cache_formal.json),[host12 steps contract](navigation/host_formal_contract.md),[gap review](navigation/formal_gap_review.md) |legalhost/paircomplete assumptions, arbitrarySRAM; host abstractchild/DMA, d16adapter only; integrated/deeper inconclusive |
| Code/FSM/functional coverage |MEASURED / PARTIAL |[component qualification ratios](results/controller_coverage.json),[current SoC profiles](results/soc_coverage.json) |Different campaigns/denominators; no merge, waivers,100% closure or SVA proof claim |
| RTL quick-start reproducibility |VERIFIED fresh curated-source |[executed receipt](results/quickstart_results.json) |Isolated curated snapshot, localMac tools, latency 0 representative cases; Linux/newlicensedcampaign UNEXECUTED |

Every retained derivative maps to original path/SHA256 in the immutable `results/source-map.json` and `verification/source-map.json` ledgers. Original raw reports and failed attempts remain preserved; a small public package is not a complete archive. Source SHA256s distinguish measured frozen d16 PPA baseline from recommended current SoC and older single-head/serial GQA comparison variants. Percentage calculations use matched endpoints; no cross-experiment end-to-end speedup, power or d64 PPA projection is published.

## Current evidence locations

The experiment subdirectories keep original receipt bytes and identities. [Original path mapping](../../../release/path-map-v1.json) plus [layout-v2](../../../release/layout-v2.json) resolve historical path strings; [evidence lock](../../../release/evidence-lock-v1.json) checks accepted hashes. Current tools use the [engineering reproduction guide](../../REPRODUCIBILITY.md).

Navigable supplements: [closure attempts](navigation/closure_attempts.md), [formal gaps](navigation/formal_gap_review.md), [host formal contract](navigation/host_formal_contract.md). Original text receipts remain immutable.
