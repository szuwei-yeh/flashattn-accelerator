"""Verify that actual testbenches reject bad inputs and deliberately broken DUTs.

Run through `make tb_oracle_checks` to build the required simulation binaries.
All corrupt fixtures and the missing-done RTL mutant live in temporary folders.
"""
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SIM = Path(__file__).resolve().parent
ROOT = SIM.parents[1]
CASES = [(kind, dim) for kind in ('core', 'top') for dim in (16, 64)]


def fixture(dim):
    return ROOT / 'data' / ('N64' if dim == 16 else 'N64_d64')


def run_attention(kind, dim, data=None, shape=None, causal=False):
    target = ('core_banked_prefetch_N64' if kind == 'core' else
              'dma_banked_prefetch_top_N64') + ('_d64' if dim == 64 else '')
    n, d = shape or (64, dim)
    command = [str(SIM / ('obj_' + target) / ('sim_' + target)),
               '--N', str(n), '--D', str(d), '--data', str(data or fixture(dim))]
    if kind == 'top' and dim == 64:
        command += ['--core_busy_ref', '45590']
    if causal:
        command += ['--causal']
    return subprocess.run(command, cwd=SIM, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=30)


class TestbenchOracles(unittest.TestCase):
    def assert_rejected(self, result, diagnostic=None):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('RESULT: PASS', result.stdout)
        if diagnostic is not None:
            self.assertIn(diagnostic, result.stdout)

    def test_canonical_controls_and_compiled_geometry(self):
        for kind, dim in CASES:
            with self.subTest(kind=kind, dim=dim):
                good = run_attention(kind, dim)
                self.assertEqual(good.returncode, 0, good.stdout)
                self.assertIn('RESULT: PASS', good.stdout)
            for shape in ((0, dim), (-16, dim), (16, dim), (64, 0),
                          (64, 64 if dim == 16 else 16)):
                with self.subTest(kind=kind, dim=dim, shape=shape):
                    self.assert_rejected(run_attention(kind, dim, shape=shape),
                                         'ERROR: testbench geometry')

    def test_one_bit_output_errors_are_not_hidden_by_tolerance(self):
        for kind, dim in CASES:
            for index in (0, 64 * dim - 1):
                with self.subTest(kind=kind, dim=dim, index=index), tempfile.TemporaryDirectory() as temp:
                    data = Path(temp) / 'data'
                    shutil.copytree(fixture(dim), data)
                    path = data / 'expected.hex'
                    words = path.read_text().splitlines()
                    words[index] = f'{int(words[index], 16) ^ 1:08x}'
                    path.write_text('\n'.join(words) + '\n')
                    self.assert_rejected(run_attention(kind, dim, data), 'RESULT: FAIL')

    def test_missing_or_short_vectors(self):
        for kind, dim in CASES:
            for name in ('q_input.hex', 'k_input.hex', 'v_input.hex', 'expected.hex'):
                for missing in (True, False):
                    with self.subTest(kind=kind, dim=dim, file=name, missing=missing), tempfile.TemporaryDirectory() as temp:
                        data = Path(temp) / 'data'
                        shutil.copytree(fixture(dim), data)
                        if missing:
                            (data / name).unlink()
                        else:
                            (data / name).write_text('00\n')
                        self.assert_rejected(run_attention(kind, dim, data))

    def test_scales_are_required_even_for_zero_output(self):
        for kind, dim in CASES:
            for fault in ('missing', 'partial', 'duplicate', 'malformed', 'overflow'):
                with self.subTest(kind=kind, dim=dim, fault=fault), tempfile.TemporaryDirectory() as temp:
                    data = Path(temp) / 'data'
                    shutil.copytree(fixture(dim), data)
                    # With V=0 and expected=0, silent scale defaults could pass.
                    (data / 'v_input.hex').write_text('00\n' * (64 * dim))
                    (data / 'expected.hex').write_text('00000000\n' * (64 * dim))
                    scales = data / 'scales.txt'
                    if fault == 'missing':
                        scales.unlink()
                    elif fault == 'partial':
                        scales.write_text('scale_q_q88 = 0x0007\n')
                    elif fault == 'duplicate':
                        scales.write_text(scales.read_text() + 'scale_q_q88 = 0x0007\n')
                    else:
                        bad = '0xGGGG' if fault == 'malformed' else '0x10000'
                        scales.write_text(re.sub(r'scale_q_q88 = \S+',
                                                 'scale_q_q88 = ' + bad, scales.read_text()))
                    self.assert_rejected(run_attention(kind, dim, data), 'ERROR:')

    def test_vectors_reject_overflow_extra_tokens_and_partial_hex(self):
        for kind, dim in CASES:
            for name in ('q_input.hex', 'k_input.hex', 'v_input.hex', 'expected.hex'):
                for fault in ('overflow', 'extra', 'junk', 'negative'):
                    with self.subTest(kind=kind, dim=dim, file=name, fault=fault), tempfile.TemporaryDirectory() as temp:
                        data = Path(temp) / 'data'
                        shutil.copytree(fixture(dim), data)
                        path = data / name
                        words = path.read_text().splitlines()
                        if fault == 'overflow':
                            bits = 32 if name == 'expected.hex' else 8
                            words[0] = f'{int(words[0], 16) + (1 << bits):X}'
                        elif fault == 'extra':
                            words.append('00')
                        elif fault == 'junk':
                            words[-1] += 'junk'
                        else:
                            words[0] = '-1'
                        path.write_text('\n'.join(words) + '\n')
                        self.assert_rejected(run_attention(kind, dim, data), 'ERROR:')

    def test_lut_low_and_high_value_corruption(self):
        for index in (0, 255):
            with self.subTest(address=index), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cwd = root / 'sim/verilator'
                cwd.mkdir(parents=True)
                (root / 'data').mkdir()
                words = (ROOT / 'data/exp_lut.hex').read_text().splitlines()
                words[index] = '0100' if index == 0 else '0000'
                (root / 'data/exp_lut.hex').write_text('\n'.join(words) + '\n')
                result = subprocess.run([str(SIM / 'obj_exp_lut/sim_exp_lut')],
                                        cwd=cwd, capture_output=True, text=True, timeout=30)
                self.assert_rejected(result, 'RESULT: FAIL')
                self.assertNotRegex(result.stdout, r'Result\s+: PASS')

    def test_loader_data_without_done_is_not_success(self):
        with tempfile.TemporaryDirectory(prefix='loader-oracle-') as temp:
            directory = Path(temp)
            mutant = directory / 'banked_tile_loader.sv'
            text, count = re.subn(r"done\s*<=\s*1'b1;", "done <= 1'b0;",
                                 (ROOT / 'rtl/memory/banked_tile_loader.sv').read_text())
            self.assertEqual(count, 1)
            mutant.write_text(text)
            command = shlex.split(os.environ.get('VERILATOR', 'verilator')) + [
                '--sv', '-cc', '--exe', '--build', '-j', '2', '-Wall',
                '--top-module', 'tb_banked_tile_loader', '--Mdir', str(directory / 'obj'),
                '-GHEAD_DIM=16', str(ROOT / 'rtl/memory/sram_1r1w.sv'),
                str(ROOT / 'rtl/memory/banked_scratchpad.sv'), str(mutant),
                str(SIM / 'tb_banked_tile_loader.sv'), str(SIM / 'tb_banked_tile_loader.cpp'),
                '-o', 'sim_mutant']
            build = subprocess.run(command, cwd=SIM, capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(directory / 'obj/sim_mutant'), '--D', '16'],
                                    cwd=SIM, capture_output=True, text=True, timeout=30)
            self.assert_rejected(result, 'TIMEOUT waiting for done')


if __name__ == '__main__':
    unittest.main()
