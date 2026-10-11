import importlib.util
from pathlib import Path
import random
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('tail_reference',Path(__file__).with_name('decode_reference.py'))
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
sys.path.insert(0,str(ROOT/'golden'))
from generate_hw_expected import flash_attn_q_tile
import numpy as np


class ReferenceTests(unittest.TestCase):
    def test_all_lengths_uniform_and_negative(self):
        for length in range(1,65):
            for value in (0,127,-127):
                out,trace=ref.decode_with_trace([0]*64,[[17]*64]*length,[[value]*64]*length,(256,256,256))
                self.assertEqual(out,[value*255]*64)
                self.assertEqual(trace[-1]['sum'],256*length)
                self.assertEqual(sum(t['valid_keys'] for t in trace),length)
                for t in trace:self.assertEqual(t['p'][t['valid_keys']:],[0]*(16-t['valid_keys']))
    def test_old_golden_aligned(self):
        rng=random.Random(20261008)
        for length in (16,32,64):
            for scales in ((256,192,256),(-128,-256,-192),(32767,32767,-32768)):
                for sample in range(2):
                    q=[rng.randrange(-16,17) for _ in range(64)]
                    k=[[rng.randrange(-32,33) for _ in range(64)] for _ in range(length)]
                    v=[[rng.randrange(-64,65) for _ in range(64)] for _ in range(length)]
                    query=np.zeros((16,64),dtype=np.int8);query[15]=q
                    for causal in (False,True):
                        old=flash_attn_q_tile(query,np.array(k,dtype=np.int8),np.array(v,dtype=np.int8),*scales,ref.load_lut(),16,causal=causal,q_tile_idx=length//16-1)
                        self.assertEqual(ref.decode(q,k,v,scales,causal=causal),old[15])
    def test_negative_scores_tail_max(self):
        out,t=ref.decode_with_trace([1]*64,[[-1]*64],[[9]*64],(256,256,256))
        self.assertLess(t[0]['max'],0);self.assertEqual(t[0]['sum'],256);self.assertEqual(out,[9*255]*64)
    def test_boundary_rejections(self):
        for length in (0,65):
            with self.assertRaises(ValueError):ref.decode([0]*64,[[0]*64]*length,[[0]*64]*length,(256,256,256))
        for length in (1,15,16,17,31,32,63,64):
            with self.assertRaises(ValueError):ref.decode([0]*64,[[0]*64]*length,[[0]*64]*length,(256,256,256),query_position=length)
    def test_tile_order_and_sum_rescaling(self):
        k=[[-1]*64]*16+[[1]*64];v=[[12]*64]*16+[[-11]*64]
        _,t=ref.decode_with_trace([16]*64,k,v,(64,64,256))
        self.assertEqual([x['base'] for x in t],[0,16]);self.assertLess(t[1]['factor'],256)
        self.assertEqual(t[1]['sum'],((t[0]['sum']*t[1]['factor']+128)>>8)+256)
    def test_causal_equivalence_latest_position(self):
        rng=random.Random(8)
        for length in (1,15,16,17,31,32,63,64):
            k=[[rng.randrange(-8,9) for _ in range(64)] for _ in range(length)]
            v=[[rng.randrange(-32,33) for _ in range(64)] for _ in range(length)]
            q=[1]*64
            self.assertEqual(ref.decode(q,k,v,(256,256,256),causal=False),ref.decode(q,k,v,(256,256,256),causal=True))

if __name__=='__main__':unittest.main()
