"""Contract checks; compare existing fixture files without regenerating them."""
import unittest
import math
import tempfile
from pathlib import Path
import numpy as np
from generate_hw_expected import (flash_attn_q_tile, load_scales, load_exp_lut,
                                  load_int8_hex, to_addr, generate)

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

    def test_preserved_single_head_fixtures(self):
        for name in ('', 'N64', 'N128', 'N256', 'N64_d64', 'N128_d64', 'N256_d64',
                     'causal_N64', 'causal_N256', 'causal_N64_d64', 'causal_N256_d64'):
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

    def test_lut_and_address_rounding_against_math(self):
        self.assertEqual(LUT, [math.floor(math.exp(-8 + i * 8 / 255) * 256 + 0.5)
                               for i in range(256)])
        # Complete difference range of two signed INT16 scores. Address spacing
        # is 8/255 in real units; include the half-step before integer rounding.
        for diff in range(-65535, 65536):
            expected = min(255, max(0, math.floor((diff + 2048) * 255 / 2048 + 0.5)))
            self.assertEqual(to_addr(diff, 0), expected)

    def test_bad_vector_and_scale_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.hex'
            path.write_text('00 7f 80 ff\n')
            self.assertEqual(load_int8_hex(path, 4).tolist(), [0, 127, -128, -1])
            for text in ('00 7f 80', '00 7f 80 ff 00', '100 7f 80 ff',
                         '-1 7f 80 ff', '00 7f 80 ffjunk'):
                with self.subTest(vector=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        load_int8_hex(path, 4)
            scales = (ROOT / 'data/N64/scales.txt').read_text()
            for text in (scales + 'scale_q_q88 = 0x0007\n',
                         scales.replace('0x0007', '0x10007'),
                         scales.replace('scale_q_q88', 'missing_q'),
                         scales.replace('0x0007', '0xGGGG')):
                with self.subTest(scales=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        load_scales(path)

    def test_invalid_generation_geometry_preserves_expected(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            expected = data / 'expected.hex'
            expected.write_text('existing data must survive\n')
            for n, d, tile in ((0,16,16), (17,16,16), (64,32,16), (64,16,8)):
                with self.subTest(n=n, d=d, tile=tile):
                    (data / 'scales.txt').write_text(
                        'scale_q_q88 = 0x0100\nscale_k_q88 = 0x0100\n'
                        f'scale_v_q88 = 0x0100\nN = {n}\nd = {d}\n')
                    with self.assertRaises(ValueError):
                        generate(data, ROOT / 'data/exp_lut.hex', tile)
                    self.assertEqual(expected.read_text(), 'existing data must survive\n')

if __name__ == '__main__':
    unittest.main()
