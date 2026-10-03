# Numerical accuracy

[Home](../README.md) · [Design](DESIGN.md) · [Verification](VERIFICATION.md)

This analysis compares fixed-point hardware arithmetic with stable NumPy
float64 attention on **the same INT8 Q/K/V and encoded signed Q8.8 scales**.
Stored signed INT32 output is divided by 256. It excludes the error of quantizing
original floating-point inputs and does not measure model or application accuracy.

| Canonical N64 fixture | Relative L2 | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| d16, noncausal | 1.755046% | 0.003145457 | 0.010368812 |
| d64, noncausal | 1.513452% | 0.002998597 | 0.014310069 |
| d16, causal | 1.015910% | 0.003417666 | 0.011919151 |
| d64, causal | 1.088217% | 0.003521150 | 0.018871652 |

The report checks all four stored expected outputs against the hardware model
without modifying them. It additionally measures 48 generated cases:
d16/d64 × noncausal/causal × (nine random cases + three precision-limit cases).
Random cases use seeds 7, 42 and 20261003 at input amplitudes 0.25, 1 and 4.
The three directed cases exercise equal-score P clipping, saturation of distinct
logits, and signed scales. Outputs for generated cases are computed by the
hardware reference; generating this report alone is not an RTL regression.

Saturation can produce a large error. In `saturated_distinct`, alternate keys
produce raw QK values 128 and 254, which both saturate before the sqrt(d) shift.
Alternate V rows are -128 and 127. The fixed-point path loses the preference for
the larger logit; noncausal relative L2 is about **100.39%** in both dimensions.
This deliberate adversarial case is published alongside the ordinary fixtures.
The equal-score constant-V case instead gives the documented **0.390625%** bias
from P=255 and denominator=256.

Relative L2 is `||O_fixed - O_float||₂ / ||O_float||₂` over all output elements.
RMSE and maximum absolute error use the dequantized output units. When the
reference norm is zero, relative L2 is undefined and is recorded as JSON `null`;
absolute metrics remain available. Saturation counts include only visible scores
before the sqrt(d) shift, excluding masked causal entries.

Reproduce the analysis with Python 3 and NumPy:

```bash
python3 golden/test_accuracy.py
python3 golden/report_accuracy.py --output-dir /tmp/flashattn-accuracy
```

The [52-row CSV](analysis/2026-10-03/accuracy.csv) contains every metric. The
[JSON report](analysis/2026-10-03/accuracy.json) also records input/model/LUT hashes,
scale encodings and analysis scope. Generated data stays in memory; canonical
fixtures are never regenerated. CI publishes a fresh report artifact on each run.

The shared-dequantizer sweep reuses these generated inputs and checks exact
core/top output agreement, so reducing lane count preserves the same numerical
contract. See the [experiment guide](DEQUANT_EXPERIMENT.md).
