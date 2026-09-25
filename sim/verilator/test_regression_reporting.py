"""Check the real coverage make target's exit-status propagation without RTL builds."""
import subprocess
import tempfile
import unittest
from pathlib import Path


class ScenarioReporting(unittest.TestCase):
    def test_success_and_failure_propagate(self):
        root = Path(__file__).resolve().parent
        # Exercise one real prerequisite edge. Other test groups are marked
        # already built so this checks reporting, not simulator availability.
        skipped = ('tb_quantizer', 'tb_exp_lut', 'tb_softmax', 'tb_sram',
                   'tb_addr_gen', 'tb_kv_buf', 'tb_top_all', 'tb_axi_slave',
                   'week6_all')
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temp:
                makefile = Path(temp) / 'reporting.mk'
                makefile.write_text(
                    'include Makefile\n'
                    'run:\n'
                    '\t@echo injected-scenario-result\n'
                    f'\t@exit {exit_code}\n')
                args = ['make', '--no-print-directory', '-f', str(makefile),
                        '-o', 'obj_dir/Vsystolic_array']
                for target in skipped:
                    args += ['-o', target]
                result = subprocess.run(args + ['coverage'], cwd=root,
                                        capture_output=True, text=True)
                output = result.stdout + result.stderr
                self.assertIn('injected-scenario-result', output)
                self.assertEqual(result.returncode == 0, exit_code == 0, output)
                self.assertEqual('Scenario test result: PASS' in output,
                                 exit_code == 0, output)
                self.assertNotIn('Total test cases : 391', output)
                self.assertNotIn('Total mismatches : 0', output)


if __name__ == '__main__':
    unittest.main()
