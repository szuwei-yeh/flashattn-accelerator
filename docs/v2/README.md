# V2 host-controlled attention SoC

V2 is the flagship implementation: persistent prefill/append/decode, four serial
query heads over two physical KV groups, shared compute, CSR/IRQ and AXI writeback.

[Architecture](../ARCHITECTURE.md) · [Results](../RESULTS.md) ·
[Verification](VERIFICATION.md) · [Reproduction](../REPRODUCIBILITY.md) ·
[Limitations](../LIMITATIONS.md) · [Evidence](evidence/README.md)

`rtl/v2/soc/soc.f` selects the current d16/d64 SoC. The frozen d16 synthesis
baseline is separately preserved in `rtl/v2/ppa_baseline/`; its mapped results do
not establish mapped PPA or equivalence for the current integration.

Public evidence is organized by purpose: `evidence/results/` holds accepted
measurements, provenance and source identities; `evidence/verification/` holds
bounded qualification records. `figures/` contains README assets.
Development preparation records are local-only under ignored `private/`.

[Ubuntu Verification run 38274178267](https://github.com/szuwei-yeh/flashattn-accelerator/actions/runs/38274178267)
passed all 12 jobs for commit `b0ee70d6565f85698d196f6f1c3abe9b498f8196`,
including four V2 d16/d64 × 256/16-lane SoC profiles. This result applies to that
tested commit; later documentation-only revisions require their own CI result.
This CI result is separate from the historical
campaigns; formal/coverage closure remains partial, d64 mapped PPA is unavailable,
and physical/electrical signoff is absent. Historical receipts retain their
original platform and source scope.
