"""Contract checks; compare existing fixture files without regenerating them."""
import unittest
from pathlib import Path
import numpy as np
from generate_hw_expected import flash_attn_q_tile, load_scales, load_exp_lut, load_int8_hex

ROOT = Path(__file__).resolve().parents[1]
LUT = load_exp_lut(ROOT/'data/exp_lut.hex')

class ReferenceContract(unittest.TestCase):
    def test_saturation_before_sqrt_shift_and_signed_scale(self):
        q=np.zeros((16,16),dtype=np.int8); q[:,0]=2
        k=np.zeros((16,16),dtype=np.int8); k[:8,0]=64; k[8:,0]=127
        v=np.ones((16,16),dtype=np.int8); v[8:,:]=3
        # Both positive products saturate to +32767; both negative products
        # saturate to -32768. Equal saturated scores yield equal LUT weights.
        # 8*255*1 + 8*255*3, normalized by 16*256, gives 510 in Q8.8.
        for sq in (256, -256, 0xff00):
            result=flash_attn_q_tile(q,k,v,sq,256,256,LUT,16)
            self.assertEqual(result, [[510]*16 for _ in range(16)])
        result=flash_attn_q_tile(q,k,v,256,256,0xff00,LUT,16)
        self.assertEqual(result, [[-510]*16 for _ in range(16)])

    def test_preserved_canonical_fixtures(self):
        for name in ('N64','N64_d64','causal_N64','causal_N64_d64'):
            with self.subTest(config=name):
                directory=ROOT/'data'/name
                sq,sk,sv,n,d=load_scales(directory/'scales.txt')
                q,k,v=[load_int8_hex(directory/f'{x}_input.hex',n*d).reshape(n,d) for x in ('q','k','v')]
                words=[]
                for i in range(n//16):
                    result=flash_attn_q_tile(q[i*16:(i+1)*16],k,v,sq,sk,sv,LUT,16,
                                            causal=name.startswith('causal'),q_tile_idx=i)
                    words += [x & 0xffffffff for row in result for x in row]
                expected=[int(x,16) for x in (directory/'expected.hex').read_text().split()]
                self.assertEqual(words,expected)

if __name__ == '__main__':
    unittest.main()
