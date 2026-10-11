"""Independent logical-row oracle checks against unchanged V1 arithmetic."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'golden'))
from generate_hw_expected import load_exp_lut
from report_accuracy import fixed_attention
spec=importlib.util.spec_from_file_location('integration_reference',Path(__file__).with_name('decode_reference.py'))
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
class Reference(unittest.TestCase):
    def test_prefill(self):
        lut=load_exp_lut(ROOT/'data/exp_lut.hex')
        for dim in (16,64):
            for causal in (False,True):
                rng=random.Random(110+dim+causal)
                q,k,v=[[[rng.randrange(-32,33) for _ in range(dim)] for _ in range(16)] for _ in range(3)]
                for scales in ((64,64,256),(-48,56,-144),(32767,-32768,1024)):
                    expected=fixed_attention(*(np.array(x,dtype=np.int8) for x in (q,k,v)),scales,lut,causal)
                    self.assertEqual(ref.prefill(q,k,v,scales,causal,lut),expected.tolist())
    def test_decode_prior_oracles(self):
        for dim,folder in ((16,'decode_d16'),(64,'decode_d64')):
            spec=importlib.util.spec_from_file_location('prior_'+folder,ROOT/'golden/v2'/folder/'decode_reference.py')
            prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
            rng=random.Random(dim)
            q=[rng.randrange(-127,128) for _ in range(dim)]
            k,v=[[[rng.randrange(-127,128) for _ in range(dim)] for _ in range(64)] for _ in range(2)]
            for length in (1,15,16,17,18,31,32,63,64):
                self.assertEqual(ref.decode(q,k[:length],v[:length],(-128,192,-256)),prior.decode(q,k[:length],v[:length],(-128,192,-256)))
    def test_position_guard(self):
        with self.assertRaises(ValueError):ref.decode([0]*16,[[0]*16],[[0]*16],(1,1,1),1)
if __name__=='__main__':unittest.main()
