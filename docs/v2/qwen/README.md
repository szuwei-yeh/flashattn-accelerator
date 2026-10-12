# Frozen Qwen2.5 real-activation validation

[Portfolio](../../../README.md#real-model-activation-validation--qwen25) ·
[Public fixtures](../../../data/qwen2_5_layer11/manifest.json) ·
[Accepted measurement receipt](accepted.json) · [Raw simulation log](qwen.log)

This experiment replays real pretrained-model attention activations through the
production V2 SoC. It checks fixed-point RTL correctness for one frozen workload;
it does not execute the transformer projections, RoPE or complete language model
in hardware. Production RTL, the original oracle/LUT and historical V1/V2 evidence
are unchanged.

## Reproduce from a fresh checkout

Requirements: Verilator **5.046**, C++ compiler/make, Python **3.11+** and NumPy.
The accepted local run used Python 3.11.9 and NumPy 1.26.4; CI uses its existing
Python 3.12 / NumPy 2.2.6 setup. No PyTorch, model weights, network access, private
archive or prior temporary directory is needed for the frozen RTL replay.

```sh
# Use an isolated environment outside the checkout if NumPy is not available.
python3 -m venv /tmp/flashattn-qwen-env
/tmp/flashattn-qwen-env/bin/python -m pip install numpy==1.26.4

/tmp/flashattn-qwen-env/bin/python -B tools/v2/qwen/test_fixtures.py
/tmp/flashattn-qwen-env/bin/python -B tools/v2/qwen/run.py \
  --output-dir /tmp/flashattn-qwen-results

# Check the release payload; keep the report outside the checkout.
/tmp/flashattn-qwen-env/bin/python -B tools/v2/validate_release.py \
  --output /tmp/flashattn-qwen-release.json
```

The environment/result directory names must be unused. The runner snapshots the
public payload, checks the registered release inventory and immutable evidence
locks, validates all fixture hashes, and recomputes the unchanged independent
oracle against the checked-in expectations before building the production
[filelist](../../../rtl/v2/soc/soc.f). It uses the unchanged
[SV observers](../../../sim/v2/soc/tb_soc.sv) and includes the original C++ host,
AXI responder/scoreboard and timing controls in the
[sibling driver](../../../sim/v2/qwen/tb_qwen.cpp).

Successful replay requires exactly five jobs and 4,608 exact external DRAM words.
All handshake, cache, result-owner, source-retirement, final-OKAY-B, IRQ and
ACK/release checks remain active. Two separate negative runs must fail: missing
fixtures, and a flipped expected-copy bit in the final word of L18 (sequence word
4,608). Neither control changes checked-in expected files or production RTL.

`results.json` records sources, fixtures, executable/output hashes, counters,
negative-control exits and verification scope. `outputs/N16.bin`, `L17.bin`,
`L18.bin` and `all.bin` contain observed external DRAM bytes, not copied goldens.
Binary words are signed INT32, little-endian, ordered by job, head, query, feature.
All generated files and builds stay outside Git. CI adds this one fixed replay to
the d64/256-lane V2 job; it never downloads or runs the model.

## Frozen inputs and measurement conditions

| Item | Frozen value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Model/tokenizer revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Transformer layer | 11, zero-based |
| Hardware Q slots → model Q heads | 0/1/2/3 → 0/1/7/8 |
| Physical KV groups | 0/0/1/1 for the four Q slots |
| Geometry | d64, 4Q:2KV, 18 actual tokens |
| Hardware parameters | HEAD_DIM=64, DEQUANT_LANES=256, COMPACT_DECODE_OUTPUT=1 |
| Endpoint timing | Modeled AXI latency 0, timing seed 0; canonical backpressure remains enabled |
| Encoded signed Q8.8 scales | Q: 34/27/55/33; K: 17/21; V: 6/11 |
| Sequence | Causal N16 PREFILL → APPEND token 17 → L17 DECODE → APPEND token 18 → L18 DECODE |

Prompt, with no added special tokens:

> A hardware accelerator computes causal attention by storing keys and values, processing queries, and updating the output as each new token arrives.

The first 18 token IDs, recorded in the fixture manifest, decode to:

> A hardware accelerator computes causal attention by storing keys and values, processing queries, and updating the

The prefix is deliberately unchanged; it was not completed or replaced after
observing errors. Q/K are post-RoPE and V is projected but unrotated. Extraction
uses the input-layernorm output and the original projection biases. Quantization
sets `S=max(1,ceil(256*max_abs/127))`, rounds `x/(S/256)` to even and clamps to
[-127,127]. Calibration is offline over all 18 positions, per head. K/V scales
remain constant across cache population, APPEND and DECODE; no scale retuning is
performed. Files are token-major with contiguous 64-byte features and one head
per file. Bases are 16-byte aligned.

## Accepted actual RTL measurements

| Job | Host cycles | Serial-engine cycles | DMA writeback cycles | Read/write bytes | Exact words |
|---|---:|---:|---:|---:|---:|
| N16 PREFILL | 31,643 | 24,126 | 7,511 | 8,192 / 16,384 | 4,096 |
| APPEND token 17 | 52 | 49 | 0 | 256 / 0 | 0 |
| L17 DECODE | 6,557 | 6,080 | 471 | 256 / 1,024 | 256 |
| APPEND token 18 | 52 | 49 | 0 | 256 / 0 | 0 |
| L18 DECODE | 6,557 | 6,080 | 471 | 256 / 1,024 | 256 |

The complete sequence takes **46,320 modeled RTL cycles including CSR service,
snapshot reads, IRQ clear and ACK/release**. Host-job counters sum to 44,861;
serial-engine counters include DMA, cache access, compute and result copy rather
than only MAC activity. These are Phase 2B real-activation simulation measurements,
not historical synthetic cycles, silicon throughput or physical timing.

All **4,608/4,608 externally written INT32 words** match the independent oracle,
with **zero mismatches**. N16/L17/L18 each retire one complete source stream and
receive the final accepted OKAY B before host success. The receipt includes
per-job AR/R/AW/W/B counts and persistent-cache/ownership safety observations.

Actual DRAM SHA-256 for the concatenated N16 → L17 → L18 output stream:

```text
8731bbef603802f82344b9fa7d5d87e05dc4a00cc0c3462eedf92e9c597379e5
```

The [curated receipt](accepted.json) retains the original run's Git base identity,
production/host/oracle/LUT hashes, executable hash, outputs, counters and negative
controls, and identifies the original results file by SHA-256. The three published
logs are byte-identical to that accepted run. This export adds readable provenance;
it does not rewrite original accepted evidence. A fresh replay records its own
current source and executable identities separately.

## Numerical fidelity: unchanged Phase 2A result

[accuracy.csv](accuracy.csv) is the unchanged original measurement table. It
contains per-head/per-job relative L2, RMSE and maximum absolute error for original
FP32 attention, reconstructed-quantized FP32 attention, and oracle output / 256.
FP32 attention is causal, latest-position aligned, and measured before `o_proj`.
The explicit reference matched the model's native pre-projection attention.

| Job | Overall relative L2, oracle / 256 vs original FP32 | Q7 | Q8 |
|---|---:|---:|---:|
| N16 | 4.165% | 24.248% | 28.959% |
| L17 | 3.040% | 24.248% | 37.069% |
| L18 | 3.541% | 24.246% | 41.042% |
| All 18 positions | **4.061%** | — | — |

No INT8 inputs clip (0/9,216). Q7 has 54/136, 7/17 and 9/18 eligible QK scores
saturated in the three jobs, all negative. Those keys carry at most about
1.35e-9 original attention probability per query in this case; saturation is not
the main output-error explanation. KV1's first-token V has 59/64 features rounded
to zero, producing about 24.24% vector error. Q8 has no QK saturation but remains
sensitive to quantization and fixed-point arithmetic at its small output amplitude.
The combined metric is dominated by larger Q0/Q1 output norms.

The original model's 16+1+1 KV-cache forward and full-prefix extraction produced
identical quantized Q/K/V (9,216 elements) and identical oracle words (4,608).
Their FP32 pre-projection outputs differed by at most 4.29e-6. This validates the
intended numerical sequence but is separate from the actual RTL comparison.

Bit-exact fixed-point agreement therefore coexists with **24%–41% selected-head
error**. The experiment remains a LIMITED GO for numerical fidelity: one prompt,
one layer and four heads, no complete Qwen inference, downstream task accuracy,
new PPA, formal closure or physical implementation claim.

## Optional model extraction and attribution

Model extraction is separate from routine RTL verification and is not a CI step.
To independently repeat it, use an isolated CPU environment with Python 3.11.9,
PyTorch 2.5.1, Transformers 4.46.3, NumPy 1.26.4, tokenizers 0.20.3,
huggingface-hub 0.26.2 and safetensors 0.4.5. Fetch the pinned revision above into
an external cache, load its BF16 weights as FP32, select the recorded first 18
tokens, and use `eval()` / `inference_mode()` with one CPU thread and deterministic
algorithms. Capture layer 11's normalized attention input, original q/k/v linear
projections and supplied cos/sin; use the official `apply_rotary_pos_emb` helper
once on Q/K, before `repeat_kv`. Select Q heads 0/1/7/8 and the two unrepeated KV
heads, then apply the frozen calibration rule. Compare the resulting INT8 stream
with the manifest hash; do not overwrite or silently substitute the public data.

The manifest records pinned model weights, original Phase 2A artifact identities,
the INT8 stream and every public fixture. Routine reproduction requires only the
published numerical data; the original private analysis bundle is not a dependency.

The upstream [model license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE)
is Apache-2.0. The data directory includes its verbatim [LICENSE](../../../data/qwen2_5_layer11/LICENSE)
and a [NOTICE](../../../data/qwen2_5_layer11/NOTICE) identifying Qwen Team / Alibaba
Cloud, the pinned source and extraction/quantization transformations. Upstream
has no separate NOTICE file at this revision. Numerical .hex data are distributed
under that attached data license; no model weights are redistributed. This does
not grant a project-level reuse license for RTL/code/evidence; the existing
[distribution status](../../../DISTRIBUTION.md) remains unchanged.
