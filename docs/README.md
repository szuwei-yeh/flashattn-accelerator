# Engineering documentation

[Project overview and quick start](../README.md)

| Guide | Purpose |
|---|---|
| [Architecture](ARCHITECTURE.md) | V2 host SoC, shared compute, physical KV groups and arithmetic |
| [Results](RESULTS.md) | Separate measured experiments, PPA and activity-based estimates |
| [Reproduction](REPRODUCIBILITY.md) | Fresh RTL checks, dependencies and historical rerun boundaries |
| [V2 verification](v2/VERIFICATION.md) | Completion, ownership, cache publication, faults and common reset |
| [Limitations](LIMITATIONS.md) | Geometry, numerical, coverage/formal and physical limits |
| [Release scope](RELEASE.md) | Current package identity, manifests and platform qualification |
| [History](HISTORY.md) | Public ancestry and exact measured-source identities |
| [Evidence](v2/evidence/README.md) | Claim-to-evidence mapping and immutable receipts |

V1 remains the architectural foundation: [design](DESIGN.md),
[verification](VERIFICATION.md), [dequantizer experiment](DEQUANT_EXPERIMENT.md),
[coverage/power](COVERAGE_POWER.md), [numerical accuracy](NUMERICAL_ACCURACY.md)
and [synthesis flow](../syn/README.md). Its N64 measurements and CI receipts are
separate from the V2 host SoC.

## October-3 revision identities

Original measured `d5f9123` has the same Git tree as public implementation
`dd68964`; original CI `068a33f` has the same tree as `6f5d45e`.
Accepted original manifests keep their recorded identities. See [history](HISTORY.md).

The public package contains engineering guides and compact supporting evidence.
Full development records and private/licensed artifacts are retained locally in
ignored `private/`, outside the public payload.
Distribution is for [portfolio review with reuse licensing pending](../DISTRIBUTION.md).
