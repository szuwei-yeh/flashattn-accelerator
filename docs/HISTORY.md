# Source identity and preserved V1 history

[Home](../README.md) · [Results](RESULTS.md) · [Release scope](RELEASE.md)

| Source / experiment | Identity and scope |
|---|---|
| Public V1 baseline | `3dd4f98f6126a07a1019c08cf5df82d9aade27df`; public ancestry is preserved |
| Original V1 core/top mapping | RTL `dfbd28e`, reports `cd99a09`; N64/d16 |
| Optional V1 dequantizer sweep | Original `d5f9123` tree equals public `dd68964`; original CI `068a33f` tree equals `6f5d45e` |
| Matched d16 V2 PPA baseline | Frozen source `5b48641988babf22383feda43258eaf73b0d6b33`; both lane endpoints use the same snapshot |
| Current d16/d64 SoC | Exact production source hashes are in [SoC provenance](v2/evidence/results/soc_provenance.json) |

The [source identity index](v2/evidence/results/source_revisions.json) and
[raw/public evidence mapping](v2/evidence/results/source-map.json) retain original
experiment names, checkpoint IDs and source hashes. These immutable records
identify measured inputs; they do not certify later source changes.
The [PPA baseline filelist](../rtl/v2/ppa_baseline/soc.f) is separate from the
[current SoC filelist](../rtl/v2/soc/soc.f).

## Release identity

`main` is the public primary branch. Read the current committed identity from Git:

```sh
git rev-parse HEAD
git status --short
```

Working-tree changes are not part of HEAD. The original public package uses
[manifest-v1](../release/manifest-v1.json), [path-map-v1](../release/path-map-v1.json),
[evidence-lock-v1](../release/evidence-lock-v1.json) and
[verification-v1](../release/verification-v1.json). Their original bytes remain intact.
[Layout-v2](../release/layout-v2.json) maps those paths to the functional directory
names and records reversible path-only source adaptations.
[Manifest-v2](../release/manifest-v2.json) inventories the current public payload;
its parent identity records the prior public release, not measured RTL or a self-referential current commit SHA.

Development plans, release-preparation logs and superseded editorial originals
are retained locally under ignored `private/`; they are outside the public payload.
The local release consolidates V2 publication and preparation commits into one
V2 commit above the exact public V1 baseline. All 39 V1 commits are preserved.
Original V2 history is recoverable from a private Git bundle; immutable evidence
retains its recorded checkpoint identities and hashes. Local history changes do
not update GitHub without a separately authorized publication. Current evidence
remains public so measured results and bounded verification can be checked.
