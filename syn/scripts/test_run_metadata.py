import unittest
import os
import subprocess
import tempfile
import uuid
import shutil
from pathlib import Path
from run_metadata import parameters

class SynthesisGeometry(unittest.TestCase):
    def test_core_and_top_explicit_defaults(self):
        core=dict(x.split('=') for x in parameters('core','').split(','))
        top=dict(x.split('=') for x in parameters('top','').split(','))
        self.assertEqual(core['SEQ_LEN'],'64')
        self.assertEqual(core,{k:top[k] for k in core})
    def test_override_keeps_geometry(self):
        self.assertIn('SEQ_LEN=64',parameters('core','HEAD_DIM=64'))
    def test_invalid_synthesis_configuration_fails_before_dc(self):
        for bad in ['HEAD_DIM=32','SEQ_LEN=0','SEQ_LEN=63','HEAD_DIM=64,SEQ_LEN=128',
                    'SRAM_DEPTH=384','SEQ_LEN=64,typo=1','TILE_SIZE=8']:
            with self.subTest(parameters=bad), self.assertRaises(ValueError):
                parameters('core',bad)

class WrapperFailureHandling(unittest.TestCase):
    def test_zero_exit_is_not_sufficient_for_success(self):
        root=Path(__file__).resolve().parents[2]
        for scenario in ('tool_error','missing_artifact','complete'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                directory=Path(temp)
                library=directory/'fake.db'; library.write_text('test only')
                tag='metadata_test_'+uuid.uuid4().hex
                run=root/'syn/runs'/tag
                fake=directory/'fake_dc'
                fake.write_text('#!/bin/bash\n'
                    'mkdir -p artifacts\n'
                    'echo finished_at=test > manifest.txt\n' +
                    ('echo Error: injected_failure\n' if scenario=='tool_error' else '') +
                    ('echo dummy > artifacts/elaborated.ddc\n' if scenario!='missing_artifact' else '') +
                    'exit 0\n')
                fake.chmod(0o700)
                env=dict(os.environ,DC_TARGET_LIBRARY=str(library),DC_SHELL_BIN=str(fake))
                try:
                    result=subprocess.run(['bash',str(root/'syn/scripts/run_server.sh'),
                                           'core',tag,'elab'],env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode==0,scenario=='complete',result.stdout+result.stderr)
                    self.assertEqual((run/'core/SUCCESS').exists(),scenario=='complete')
                    self.assertEqual((run/'core/FAILED').exists(),scenario!='complete')
                finally:
                    if run.exists(): shutil.rmtree(run) # this test's unique generated run only

if __name__ == '__main__': unittest.main()
