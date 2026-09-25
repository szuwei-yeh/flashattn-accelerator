"""Run existing N64 optimized-core binaries on fresh extreme-value fixtures.

All generated vectors stay in a TemporaryDirectory; checked-in data is untouched.
Build core_banked_prefetch_N64 and core_banked_prefetch_N64_d64 first.
"""
import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from generate_hw_expected import flash_attn_q_tile, load_exp_lut

ROOT=Path(__file__).resolve().parents[1]
lut=load_exp_lut(ROOT/'data/exp_lut.hex')
with tempfile.TemporaryDirectory(prefix='flashattn-numeric-') as temp:
    for dim in (16,64):
        rng=np.random.default_rng(2409+dim)
        q,k,v=[rng.integers(-128,128,size=(64,dim),dtype=np.int16).astype(np.int8) for _ in range(3)]
        for case,scales in enumerate(((256,256,256),(0xff00,256,0xfe00),(32767,0x8000,32767))):
            directory=Path(temp)/f'd{dim}_case{case}';directory.mkdir()
            for name,data in zip(('q','k','v'),(q,k,v)):
                (directory/f'{name}_input.hex').write_text(''.join(f'{int(x)&255:02x}\n' for x in data.flat))
            (directory/'scales.txt').write_text(''.join(f'scale_{name}_q88 = 0x{scale:04X}\n' for name,scale in zip(('q','k','v'),scales)))
            expected=[]
            for i in range(4):
                expected += [word for row in flash_attn_q_tile(q[i*16:(i+1)*16],k,v,*scales,lut,16) for word in row]
            (directory/'expected.hex').write_text(''.join(f'{x&0xffffffff:08x}\n' for x in expected))
            target='core_banked_prefetch_N64'+('_d64' if dim==64 else '')
            binary=ROOT/'sim/verilator'/('obj_'+target)/('sim_'+target)
            print(f'CORNER d={dim} scales={scales}',flush=True)
            subprocess.run([str(binary),'--N','64','--D',str(dim),'--data',str(directory)],cwd=ROOT/'sim/verilator',check=True)
print('RESULT: PASS six full-core saturation/signed-scale/extreme-value scenarios')
