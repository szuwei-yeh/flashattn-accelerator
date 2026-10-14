# Engineering documentation

[Project overview and quick start](../README.md)

## Current V2 engineering guides

| Guide | Purpose |
|---|---|
| [Architecture](ARCHITECTURE.md) | V2 host SoC, shared compute, physical KV groups and arithmetic |
| [Results](RESULTS.md) | Separate measured experiments, PPA and activity-based estimates |
| [Limitations](LIMITATIONS.md) | Geometry, numerical, coverage/formal and physical limits |

## V1 historical foundation

These guides describe the V1 foundation and its recorded experiments. Its N64
measurements and CI receipts are separate from the V2 host SoC.

| Guide | Purpose |
|---|---|
| [Design](DESIGN.md) | V1 datapath, DMA, banking and prefetch architecture |
| [Verification](VERIFICATION.md) | V1 core/top checks and historical baselines |
| [Detailed results](RESULTS_EVIDENCE.md) | Preserved V1 measurements; introductory link to current V2 results |
| [Dequantizer experiment](DEQUANT_EXPERIMENT.md) | Historical matched lane-count study |
| [Coverage and power](COVERAGE_POWER.md) | Dated V1 coverage and activity-based power campaign |
| [Numerical accuracy](NUMERICAL_ACCURACY.md) | V1 fixed-point and quantization study |
| [Synthesis flow](../syn/README.md) | Historical synthesis reproduction requirements |

## Verification, reproducibility, release and evidence

| Guide | Purpose |
|---|---|
| [V2 verification](v2/VERIFICATION.md) | Completion, ownership, cache publication, faults and common reset |
| [Reproduction](REPRODUCIBILITY.md) | Fresh RTL checks, dependencies and historical rerun boundaries |
| [DECODE waveform](v2/waveform/README.md) | Authentic L17 RTL trace, accepted-edge checks and native viewer capture |
| [Qwen activation validation](v2/qwen/README.md) | Frozen activation fixtures and bounded production-RTL replay |
| [V2 tools](../tools/v2/README.md) | Verification, integrity, replay and figure-generation utilities |
| [Release scope](RELEASE.md) | Current package identity, manifests and platform qualification |
| [History](HISTORY.md) | Public ancestry and exact measured-source identities |
| [Evidence](v2/evidence/README.md) | Claim-to-evidence mapping and immutable receipts |

### October-3 revision identities

Original measured `d5f9123` has the same Git tree as public implementation
`dd68964`; original CI `068a33f` has the same tree as `6f5d45e`.
Accepted original manifests keep their recorded identities. See [history](HISTORY.md).

The public package contains engineering guides and compact supporting evidence.
Full development records and private/licensed artifacts are retained locally in
ignored `private/`, outside the public payload.
Distribution is for [portfolio review with reuse licensing pending](../DISTRIBUTION.md).
