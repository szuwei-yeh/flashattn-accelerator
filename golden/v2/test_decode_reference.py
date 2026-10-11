import sys
from pathlib import Path
import random
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_reference import decode, load_lut, signed, trunc_div, dequant_score, lut_address


class DecodeReferenceTests(unittest.TestCase):
    def setUp(self):
        self.q=[0]*16;self.k=[[0]*16 for _ in range(16)];self.v=[[1]*16 for _ in range(16)]
    def test_uniform_cap_bias(self):
        self.assertEqual(decode(self.q,self.k,self.v,(256,256,256)),[255]*16)
    def test_zero(self):
        self.assertEqual(decode(self.q,self.k,[[0]*16 for _ in range(16)],(256,256,256)),[0]*16)
    def test_negative_uniform(self):
        self.assertEqual(decode(self.q,self.k,[[-1]*16 for _ in range(16)],(256,256,256)),[-255]*16)
    def test_scale_encodings(self):
        self.assertEqual(signed(65535,16),-1);self.assertEqual(signed(0xffffffff,32),-1)
    def test_division(self):
        self.assertEqual(trunc_div(-11,3),-3)
        with self.assertRaises(ValueError):trunc_div(1,0)
    def test_round_tie(self):
        self.assertEqual(dequant_score(896,1,1),1) # (896+128)>>8 =4; sqrt shift ->1
        self.assertEqual(dequant_score(-128,1,1),0)
    def test_saturation_before_shift(self):
        self.assertEqual(dequant_score(100000,32767,32767),8191)
        self.assertEqual(dequant_score(-100000,32767,32767),-8192)
    def test_lut(self):
        lut=load_lut();self.assertEqual(lut[lut_address(0,0)],256)
        self.assertEqual(lut[lut_address(-32768,8191)],0)
    def test_unsupported_lengths(self):
        for n in (0,1,15,17,31,32,63,64):
            with self.assertRaises(ValueError):decode(self.q,self.k[:n],self.v[:n] if n<16 else self.v+[[1]*16]*(n-16),(1,1,1))
    def test_position(self):
        for p in (0,14,16,63):
            with self.assertRaises(ValueError):decode(self.q,self.k,self.v,(1,1,1),p)
    def test_int8_bounds(self):
        with self.assertRaises(ValueError):decode([128]*16,self.k,self.v,(1,1,1))
    def test_established_golden_last_row(self):
        # A compatibility check only; the new scalar oracle does not call this reference.
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from generate_hw_expected import flash_attn_q_tile
        rng=random.Random(20261007)
        for scales in ((32,40,96),(-48,56,-144),(32767,32767,32767),(256,256,256)):
            for _ in range(3):
                q=[rng.randrange(-128,128) for _ in range(16)]
                k=[[rng.randrange(-128,128) for _ in range(16)] for _ in range(16)]
                v=[[rng.randrange(-128,128) for _ in range(16)] for _ in range(16)]
                matrix=np.zeros((16,16),dtype=np.int8);matrix[15]=q
                for causal in (False,True):
                    old=flash_attn_q_tile(matrix,np.array(k,dtype=np.int8),np.array(v,dtype=np.int8),
                                         *scales,load_lut(),16,q_tile_idx=0,causal=causal)
                    self.assertEqual(decode(q,k,v,scales,causal=causal),old[15])

if __name__=='__main__':unittest.main()
