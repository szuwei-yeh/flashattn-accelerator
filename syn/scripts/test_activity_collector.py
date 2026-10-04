"""Exercise matched-report rejection without requiring licensed EDA tools."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sample():
    file_counts = dict(
        file=96, implied=0, propagated=4, default=0, not_annotated=0, total=100
    )
    rows = []
    for lane in (256, 32, 16):
        for name, mask in (
            ("canonical", 0),
            ("canonical", 1),
            ("random_seed7_amp1", 0),
            ("saturated_distinct", 0),
        ):
            rows.append(
                dict(
                    lanes=lane,
                    case=name,
                    causal=mask,
                    fixture_sha256={"expected.hex": "same"},
                    library_sha256="same",
                    begin_ns=10,
                    end_ns=110,
                    duration_ns=100,
                    tb_cycles=9,
                    cycles_including_start=10,
                    total_w=0.1,
                    dynamic_w=0.09,
                    leakage_w=0.01,
                    energy_nj=10,
                    scope_mapping=[],
                    annotation={
                        "Primary Input": dict(file=100, total=100),
                        "Sequential": dict(file_counts),
                        "Nets": dict(default=2, total=100),
                    },
                )
            )
    source = "sim/vcs/tb_attention.sv"
    flow = "syn/scripts/run_activity_power.py"
    return dict(
        status="PASS",
        source_sha256={
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        },
        script_sha256={flow: hashlib.sha256((ROOT / flow).read_bytes()).hexdigest()},
        measured_commit="synthetic_reporting_test",
        clock_period_ns=10,
        mode="test",
        rows=rows,
    )


class CollectorContracts(unittest.TestCase):
    def test_complete_comparison_and_rejected_mismatches(self):
        for mode in (
            "valid",
            "missing_case",
            "unmatched_fixture",
            "wrong_energy",
            "wrong_window",
            "unmatched_library",
            "poor_annotation",
        ):
            record = sample()
            if mode == "missing_case":
                record["rows"].pop()
            elif mode == "unmatched_fixture":
                record["rows"][-1]["fixture_sha256"]["expected.hex"] = "different"
            elif mode == "wrong_energy":
                record["rows"][-1]["energy_nj"] = 11
            elif mode == "wrong_window":
                record["rows"][-1]["end_ns"] = 111
            elif mode == "unmatched_library":
                record["rows"][-1]["library_sha256"] = "different"
            elif mode == "poor_annotation":
                record["rows"][-1]["annotation"]["Sequential"]["file"] = 94
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                source = folder / "input.json"
                source.write_text(json.dumps(record))
                command = [
                    sys.executable,
                    str(ROOT / "syn/scripts/summarize_activity_power.py"),
                    str(source),
                    "--output-dir",
                    str(folder / "export"),
                    "--flow-commit",
                    "test",
                    "--pt-version",
                    "test",
                    "--vcs-version",
                    "test",
                ]
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                self.assertEqual(result.returncode == 0, mode == "valid", result.stdout)
                self.assertEqual(
                    (folder / "export/power.csv").exists(), mode == "valid"
                )


if __name__ == "__main__":
    unittest.main()
