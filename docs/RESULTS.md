# V2 measured results and experiment boundaries

[Home](../README.md) · [Architecture](ARCHITECTURE.md) · [Claim matrix](v2/evidence/README.md)

## Single-head traversal experiment — MEASURED / VERIFIED

![Decode cycles](v2/figures/decode_cycles.svg)

| Warm L16,256 lanes, absolute position15 |AXI latency 0 |20 |100 |
|---|---:|---:|---:|
|d16 baseline →compact |706→226 |726→246 |806→326 |
|d64 baseline →compact |2659→739 |2679→759 |2759→839 |

Single-head query start→done, not full host/writeback latency. Exact valid outputs, write retirement, phase counters and matched traffic passed. Update and normalization each257→17 cycles per PV chunk; d64 has four chunks. [Original measurement derivative](v2/evidence/results/decode_before_after.csv)

## Full SoC sharing latency — MEASURED / VERIFIED

| Warm4-head L16 host job, compact,256→16 lanes |AXI0 |20 |100 |
|---|---:|---:|---:|
|d16 |1228→1292 |1365→1427 |1925→1991 |
|d64 |4210→4272 |4455→4520 |5502→5566 |

Accepted-command→successful host DONE includes packing/writeback, excludes CSR programming/ACK/cache initialization. The current SoC records both lane/traversal variants; [records](v2/evidence/results/soc_cycles.csv). Phase activity overlaps; adding DMA/cache/compute counters would double-count time. Lane sharing raises d16 AXI0 job latency 5.21%; d64 cost is measured separately.

## Matched d16 full-SoC synthesis — MEASURED, pre-layout

![Area](v2/figures/soc_area.svg)

| frozen d16 PPA baseline top, N16/d16,10 ns |256 lanes |16 lanes |
|---|---:|---:|
|Standard-cell area, library units |5,758,250.967611 |2,395,725.809614 |
|Library leaf cells |1,549,670 |660,087 |
|Setup slack, ns |+3.918528557 |+0.459291458 |
|TNS / violating setup paths |0/0 |0/0 |
|Mapped latches / unresolved refs |0/0 |0/0 |
|Nonzero-limit max-cap violations |2 |1 |

58.394905% area reduction; same frozen RTL source, gscl45nm typical1.1V27C, DC R-2020.09-SP4, single-core ordinary compile,1 ns IO delays and logical81SRAM/16ROM blackboxes. Source checkpoint5b48641988babf22383feda43258eaf73b0d6b33; exact sources still match. Result registers are32,768 real bits. CSR/adapter and in-context mapping also change slightly; the pair audit does not attribute every cell difference to multipliers. No mapped equivalence, physical memory-inclusive area, achieved Fmax or electrical signoff. [PPA records](v2/evidence/results/ppa.json),[pair audit](v2/evidence/results/pair_audit.json).

## Physical GQA reuse — MEASURED

![GQA traffic](v2/figures/gqa_traffic.svg)

Matched d16 serial GQA, N16,4Q:4KV versus 4Q:2KV five-command workload:3456→2304 total AXI read payload bytes (−33.333333%); K/V2304→1152 (−50%), Q1152 unchanged. This earlier compute/cache scheduler experiment excludes later SoC writeback and AXI control overhead. Four query heads execute serially; KV storage/load redundancy halves, not array execution. [Matched measurements](v2/evidence/results/gqa_comparison.json)

## d16 workload energy — ESTIMATED from MEASURED activity

![Energy](v2/figures/soc_energy.svg)

| Warm4Q:2KV, d16/L16, AXI0, compact |256 lanes |16 lanes |
|---|---:|---:|
|Dynamic power, mW |154.25004817 |146.16956533 |
|Leakage power, mW |28.5131261 |13.0645866 |
|Total power, mW |182.763174 |159.234151 |
|Job window, ns |12290 |12930 |
|Total standard-cell energy,µJ |2.24615940846 |2.05889757243 |
|Dynamic energy,µJ |1.89573309201 |1.88997247972 |

The activity-power experiment uses the original matched d16 synthesis artifacts and library with PrimeTime R-2020.09-SP5-1 +PrimePower. VCS activity starts at accepted warm command and includes the full DONE cycle; window=(host latency+1)×10 ns, not host latency×10 ns. Prefill/cache initialization/CSR/ACK excluded. Total energy decreases8.336979%; dynamic energy decreases only 0.303873%, near-neutral within model limitations. Lower total energy largely reflects leakage savings.

Primary inputs100% direct; sequential file+implied96.72%; remaining sequential activity is propagated, not direct. All-net default1.9563%/1.4931%. Actual register CLK pins all clocked; unclocked warning pins are asynchronous S/R, not proven physical reset closure.97logical memories and an empty DW feedthrough have no physical memory power model. No routed parasitics, clock-tree/system energy or electrical signoff. [Power records](v2/evidence/results/power.json),[source/raw-public ledger](v2/evidence/results/source-map.json).

## d64 status — RESOURCE-LIMITED / NOT RUN

Full SoC Verilator/VCS and both lane DC analysis/elaboration/link PASS. One16-lane compile stopped under the 12 GiB RSS watchdog in Pass1Mapping; no valid mapped endpoint.256-lane mapping NOT RUN; matched d64 PPA NOT AVAILABLE. No new attempt is made on the current VM.32–64 GiB is a planning range for a separately authorized feasibility audit, not a proven requirement. [Frozen status](v2/evidence/results/d64_status.json)

Portable evidence is a curated derivative package. Complete original reports, failed attempts, DDC/netlists and activity captures are preserved locally; licensed/private artifacts are not bundled. Public reproduction verifies RTL and numerical fixtures; independently rerunning PPA needs licensed tools, library and original artifact hydration. See [reproduction](REPRODUCIBILITY.md) and [limits](LIMITATIONS.md).
