"""Frozen-fixture correctness and corruption controls; no model or RTL downloads."""
import json
import pathlib
import shutil
import tempfile
import unittest

from fixtures import prepare, sha

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = ROOT/'data/qwen2_5_layer11'


class FrozenFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen-fixture-control-')
        self.root = pathlib.Path(self.tmp.name)
        self.data = self.root/'data'
        shutil.copytree(DATA,self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_original_oracle_and_frozen_stream(self):
        before = {p.name:sha(p) for p in self.data.iterdir()}
        result = prepare(self.data,self.root/'out',ROOT)
        self.assertEqual(result['oracle_words_checked_against_phase2a'],4608)
        self.assertEqual(len((self.root/'out/expected_all.bin').read_bytes()),18432)
        self.assertEqual(sha(self.root/'out/expected_all.bin'),
                         '8731bbef603802f82344b9fa7d5d87e05dc4a00cc0c3462eedf92e9c597379e5')
        self.assertEqual(before,{p.name:sha(p) for p in self.data.iterdir()})

    def corrupt(self,name):
        p = self.data/name
        tokens = p.read_text().splitlines()
        tokens[-1] = f'{int(tokens[-1],16)^1:0{len(tokens[-1])}x}'
        p.write_text('\n'.join(tokens)+'\n')

    def test_int8_input_corruption_rejected(self):
        self.corrupt('k0.hex')
        with self.assertRaisesRegex(ValueError,'fixture hash mismatch'):
            prepare(self.data,self.root/'out',ROOT)

    def test_last_oracle_word_corruption_rejected(self):
        self.corrupt('dec18_h3.hex')
        with self.assertRaisesRegex(ValueError,'fixture hash mismatch'):
            prepare(self.data,self.root/'out',ROOT)

    def test_rehashed_manifest_cannot_accept_replacement_workload(self):
        self.corrupt('k0.hex')
        p = self.data/'manifest.json'
        value = json.loads(p.read_text())
        value['files']['k0.hex']['sha256'] = sha(self.data/'k0.hex')
        p.write_text(json.dumps(value,indent=2)+'\n')
        with self.assertRaisesRegex(ValueError,'manifest SHA256 mismatch'):
            prepare(self.data,self.root/'out',ROOT)


if __name__ == '__main__':
    unittest.main()
