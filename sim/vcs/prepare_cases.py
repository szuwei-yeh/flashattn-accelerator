"""Prepare additional VCS fixtures from the existing fixed-point reference.

Uses Python 3 + NumPy. Outputs are temporary test inputs, never canonical data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "golden"))
from generate_hw_expected import load_exp_lut
from report_accuracy import generated_cases, fixed_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    args.output.mkdir(parents=True)
    lut = load_exp_lut(ROOT / "data/exp_lut.hex")
    selected = {
        "random_seed7_amp1",
        "uniform_bias",
        "saturated_distinct",
        "signed_scales",
    }
    cases = []
    for dim in (16, 64):
        for name, matrices, scales in generated_cases(dim):
            if name not in selected:
                continue
            for causal in (False, True):
                folder = args.output / ("d%d_%s_c%d" % (dim, name, causal))
                folder.mkdir()
                for label, array in zip(("q", "k", "v"), matrices):
                    (folder / (label + "_input.hex")).write_text(
                        "".join("%02x\n" % (int(x) & 255) for x in array.ravel())
                    )
                expected = fixed_attention(*matrices, scales, lut, causal)
                (folder / "expected.hex").write_text(
                    "".join("%08x\n" % (int(x) & 0xFFFFFFFF) for x in expected.ravel())
                )
                cases.append(
                    dict(
                        name=name,
                        dim=dim,
                        causal=int(causal),
                        scales=list(scales),
                        directory=folder.name,
                        hashes={
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(folder.iterdir())
                        },
                    )
                )
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    print("Prepared %d temporary exact fixtures" % len(cases))


if __name__ == "__main__":
    main()
