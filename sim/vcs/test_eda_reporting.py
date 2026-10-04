"""Licensed-flow reporting contracts; these tests do not claim EDA execution."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from run_coverage import run, PASS, validate_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "syn/scripts"))
from run_activity_power import normalize_scope_header, activity_counts, power_values


class ReportingContracts(unittest.TestCase):
    def test_simulation_false_success_is_rejected(self):
        for output, code, accepted in [
            (PASS, 0, True),
            (PASS, 2, False),
            ("Fatal: mismatch", 0, False),
            (PASS + "\nError-[SIM] problem", 0, False),
            ("stopped", 0, False),
        ]:
            with self.subTest(
                output=output, code=code
            ), tempfile.TemporaryDirectory() as tmp:
                command = [
                    sys.executable,
                    "-c",
                    "import sys; print(%r); sys.exit(%d)" % (output, code),
                ]
                if accepted:
                    run(command, Path(tmp) / "log", require_pass=True)
                else:
                    with self.assertRaises(RuntimeError):
                        run(command, Path(tmp) / "log", require_pass=True)

    def test_truncated_fixture_and_invalid_hex(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, width in [
                ("q_input.hex", 2),
                ("k_input.hex", 2),
                ("v_input.hex", 2),
                ("expected.hex", 8),
            ]:
                (folder / name).write_text(("0" * width + "\n") * 1024)
            self.assertEqual(len(validate_fixture(folder, 16)), 4)
            (folder / "expected.hex").write_text("00000000\n" * 1023)
            with self.assertRaises(ValueError):
                validate_fixture(folder, 16)
            (folder / "expected.hex").write_text("unknown\n" * 1024)
            with self.assertRaises(ValueError):
                validate_fixture(folder, 16)

    def test_vcd_mapping_preserves_activity(self):
        header = "$scope module gen_parallel_dequant.gen_dequant[0].u_deq $end\n$var reg 8 A Q_reg[255] [7:0] $end\n$var reg 8 B Q_reg[0] [7:0] $end\n$enddefinitions $end\n"
        values = "#30770000\nb10001010 A\nb01010101 B\n#30775000\nb00010001 A\n"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw"
            dst = Path(tmp) / "mapped"
            src.write_text(header + values)
            mappings = normalize_scope_header(src, dst)
            result = dst.read_text()
            self.assertEqual(
                mappings,
                [
                    dict(
                        rtl="gen_parallel_dequant.gen_dequant[0].u_deq",
                        mapped="gen_parallel_dequant_gen_dequant_0__u_deq",
                    )
                ],
            )
            self.assertIn("$var reg 8 A Q_reg [2047:2040] $end", result)
            self.assertIn("$var reg 8 B Q_reg [7:0] $end", result)
            self.assertEqual(result.split("$enddefinitions $end\n")[1], values)

    def test_power_units_and_sum(self):
        good = "Dynamic Power Units = 1 W\nLeakage Power Units = 1 W\nNet Switching Power = .01\nCell Internal Power = .02\nCell Leakage Power = .003\nTotal Power = .033\n"
        self.assertAlmostEqual(power_values(good, "")["dynamic_w"], 0.03)
        with self.assertRaises(ValueError):
            power_values(good.replace("1 W", "1 mW"), "")
        with self.assertRaises(ValueError):
            power_values(good.replace("= .033", "= .5"), "")

    def test_activity_requires_consistent_categories(self):
        row = "1(10.00%) 0(0%) 0(0%) 0(0%) 0(0%) 0(0%) 2(20%) 7(70%) 0(0%) 0(0%) 10"
        text = "\n".join(
            label + " " + row
            for label in ("Nets", "Primary Input", "Sequential", "Combinational")
        )
        counts = activity_counts(text)
        self.assertEqual(counts["Sequential"]["default"], 2)
        with self.assertRaises(ValueError):
            activity_counts(text.replace(" 10", " 11"))
        with self.assertRaises(ValueError):
            activity_counts("No switching activity annotated")


if __name__ == "__main__":
    unittest.main()
