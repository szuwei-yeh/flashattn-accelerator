"""Check the real VCS make target without requiring the licensed simulator."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from run_vcs_test import PASS_MARKER


class VcsFailureHandling(unittest.TestCase):
    def test_make_rejects_false_success_and_stale_logs(self):
        root = Path(__file__).resolve().parent
        scenarios = (
            ('pass', PASS_MARKER, 0, 0, True),
            ('fatal_zero_exit', 'Fatal: injected mismatch', 0, 0, False),
            ('missing_pass', 'Simulation stopped early', 0, 0, False),
            ('pass_nonzero_exit', PASS_MARKER, 7, 0, False),
            ('pass_then_fatal', PASS_MARKER + '\nFatal: injected mismatch', 0, 0, False),
            ('error_zero_exit', PASS_MARKER + '\nError-[TEST] injected error', 0, 0, False),
            ('compile_failure', PASS_MARKER, 0, 7, False),
        )
        for name, output, sim_exit, compile_exit, accepted in scenarios:
            with self.subTest(scenario=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for filename in ('Makefile', 'run_vcs_test.py'):
                    shutil.copy2(root / filename, directory / filename)
                build = directory / 'obj_output_buffer_vcs'
                build.mkdir()
                (build / 'run.log').write_text(PASS_MARKER + '\n')
                # A failed compile must not run a stale successful executable.
                (build / 'simv').write_text('#!/bin/sh\necho ' + repr(PASS_MARKER) + '\n')
                (build / 'simv').chmod(0o700)
                simulator = ('#!/usr/bin/env python3\nimport sys\n'
                             f'print({output!r})\nsys.exit({sim_exit})\n')
                compiler = directory / 'fake_vcs'
                compiler.write_text(
                    '#!/usr/bin/env python3\nfrom pathlib import Path\nimport sys\n'
                    f'if {compile_exit}: sys.exit({compile_exit})\n'
                    f"Path('simv').write_text({simulator!r})\n"
                    "Path('simv').chmod(0o700)\n")
                compiler.chmod(0o700)
                result = subprocess.run(
                    ['make', '--no-print-directory', 'vcs_output_buffer_init',
                     'VCS=./../fake_vcs'], cwd=directory,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True)
                self.assertEqual(result.returncode == 0, accepted, result.stdout)
                if compile_exit == 0:
                    self.assertEqual((build / 'run.log').read_text(), output + '\n')


if __name__ == '__main__':
    unittest.main()
