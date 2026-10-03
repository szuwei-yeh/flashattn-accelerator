#!/usr/bin/env python3
"""Validate explicit synthesis geometry and record portable source fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path


def parameters(profile, override):
    if profile not in ('core', 'core_rtl', 'top', 'top_rtl'):
        return override.strip()
    values = dict(TILE_SIZE=16, HEAD_DIM=16, SEQ_LEN=64, SRAM_DEPTH=4096,
                  DEQUANT_LANES=256)
    if profile.startswith('top'):
        values.update(AXI_ADDR_W=32, AXI_DATA_W=64)
    if override.strip():
        for item in override.split(','):
            key, value = item.strip().split('=', 1)
            key = key.strip()
            if key not in values:
                raise ValueError(f'Unsupported parameter: {key}')
            values[key] = int(value.strip(), 10)
    t, d, n, depth = (values[k] for k in ('TILE_SIZE','HEAD_DIM','SEQ_LEN','SRAM_DEPTH'))
    if depth != 4096:
        raise ValueError('Supported SRAM_DEPTH is 4096 for fixed 12-bit internal interfaces')
    if t != 16 or d not in (16,64) or n <= 0 or n % 16 or n*d > depth:
        raise ValueError('Require TILE_SIZE=16, HEAD_DIM=16/64, aligned nonzero N and N*d <= SRAM_DEPTH')
    if values.get('AXI_ADDR_W',32) != 32 or values.get('AXI_DATA_W',64) != 64:
        raise ValueError('Supported AXI widths are 32/64')
    if values['DEQUANT_LANES'] not in (256,32,16):
        raise ValueError('Supported DEQUANT_LANES are 256/32/16')
    return ','.join(f'{k}={v}' for k,v in values.items())


def snapshot(root, filelist):
    paths = [filelist, 'syn/scripts/run_server.sh', 'syn/scripts/dc_run.tcl',
             'syn/scripts/run_metadata.py']
    paths += [s.strip() for s in (root/filelist).read_text().splitlines()
              if s.strip() and not s.lstrip().startswith('#')]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(paths))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=['parameters','capture','verify'])
    p.add_argument('--profile',default='')
    p.add_argument('--parameters',default='')
    p.add_argument('--root',type=Path)
    p.add_argument('--filelist')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode == 'parameters':
        print(parameters(a.profile,a.parameters)); return
    current=snapshot(a.root,a.filelist)
    if a.mode == 'capture':
        a.output.write_text(json.dumps(current,indent=2)+'\n')
    elif current != json.loads(a.output.read_text()):
        raise SystemExit('ERROR: synthesis sources changed during this run')

if __name__ == '__main__':
    main()
