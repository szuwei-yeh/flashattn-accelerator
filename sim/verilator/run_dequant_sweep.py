"""Build isolated 256/32/16-lane core/top variants and check exact outputs.

Run from a checkout without spaces. --smoke checks canonical noncausal/causal
outputs; the default additionally exercises every generated accuracy fixture.
Compilation and complete simulator logs remain in ignored obj_* directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

SIM = Path(__file__).resolve().parent
ROOT = SIM.parents[1]
sys.path.insert(0,str(ROOT/'golden'))
from report_accuracy import generated_cases, fixed_attention
from generate_hw_expected import load_exp_lut


def run(command, logfile, timeout=600):
    with logfile.open('w') as stream:
        result = subprocess.run(command,cwd=SIM,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    output=logfile.read_text()
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(map(str,command))}\n{output[-6000:]}")
    return output


def sources(kind):
    filelist=ROOT/'syn/filelists'/('core_banked_prefetch.f' if kind=='core' else 'top_dma_banked_prefetch.f')
    files=[str(ROOT/line.strip()) for line in filelist.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith('#')]
    if kind=='top':
        files += [str(ROOT/'rtl/interface/axi_mem_model.sv'),str(SIM/'tb_dma_banked_prefetch_harness.sv')]
    return files


def build(kind,lanes,dim,jobs):
    directory=SIM/f'obj_dequant_{kind}_l{lanes}_d{dim}'
    directory.mkdir(exist_ok=True)
    top='flash_attn_core_banked_prefetch' if kind=='core' else 'tb_dma_banked_prefetch_harness'
    test='tb_core_banked_prefetch.cpp' if kind=='core' else 'tb_dma_banked_prefetch.cpp'
    command=shlex.split(os.environ.get('VERILATOR','verilator'))+[
        '--sv','--cc','--exe','--build','-j',str(jobs),'-Wall','--top-module',top,
        '--Mdir',str(directory),'-GSEQ_LEN=64',f'-GHEAD_DIM={dim}',f'-GDEQUANT_LANES={lanes}',
        '--CFLAGS',f'-DTB_SEQ_LEN=64 -DTB_HEAD_DIM={dim}',*sources(kind),str(SIM/test),'-o','sim_dequant']
    run(command,directory/'build.log')
    return directory/'sim_dequant',directory


def check(binary,directory,kind,dim,name,data,causal):
    command=[str(binary),'--N','64','--D',str(dim),'--data',str(data)]
    if kind=='top' and dim==64:
        command += ['--core_busy_ref','45590']
    if causal:
        command += ['--causal']
    output=run(command,directory/f'{name}_causal{int(causal)}.log',timeout=120)
    if 'RESULT: PASS' not in output or re.search(r'RESULT:\s*FAIL',output):
        raise RuntimeError(f'Missing explicit PASS: {name}\n{output[-4000:]}')
    counts=[]
    if kind=='top':
        counts=[dict(read_latency=int(m[0]),tb_cycles=int(m[1]),perf_total=int(m[2]))
                for m in re.findall(r'^\s*(0|20|100)\s+PASS\s+(\d+)\s+(\d+)',output,re.M)]
        if len(counts)!=3:
            raise RuntimeError('Missing latency sweep results')
    else:
        found=re.search(r'done at cycle (\d+)',output,re.I)
        if found:
            counts=[dict(core_cycles=int(found[1]))]
    return dict(kind=kind,case=name,causal=causal,counts=counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes',type=int,choices=(256,32,16),required=True)
    parser.add_argument('--dim',type=int,choices=(16,64),required=True)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if ' ' in str(ROOT):
        parser.error('Verilator requires a checkout path without spaces')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    results=[]
    lut=load_exp_lut(ROOT/'data/exp_lut.hex')
    data_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT/'data').rglob('*') if p.is_file()}
    for kind in ('core','top'):
        print(f'BUILD {kind} d{args.dim} lanes={args.lanes}',flush=True)
        binary,directory=build(kind,args.lanes,args.dim,args.jobs)
        for causal in (False,True):
            folder=('causal_N64' if causal else 'N64')+('_d64' if args.dim==64 else '')
            result=check(binary,directory,kind,args.dim,'canonical',ROOT/'data'/folder,causal)
            if args.lanes==256 and not causal:
                expected = (7589 if args.dim==16 else 28337) if kind=='core' else (7800 if args.dim==16 else 29160)
                actual = result['counts'][0]['core_cycles' if kind=='core' else 'tb_cycles']
                if actual != expected:
                    raise RuntimeError(f'Parallel baseline cycle count changed: {actual} != {expected}')
            results.append(result)
            print(f'PASS {kind} d{args.dim} lanes={args.lanes} canonical causal={causal}: {result["counts"]}',flush=True)
            if not args.smoke:
                for name,matrices,scales in generated_cases(args.dim):
                    with tempfile.TemporaryDirectory(prefix='dequant-fixture-') as temp:
                        data=Path(temp)
                        for label,matrix in zip(('q','k','v'),matrices):
                            (data/f'{label}_input.hex').write_text(''.join(f'{int(v)&255:02x}\n' for v in matrix.flat))
                        (data/'scales.txt').write_text(''.join(f'scale_{n}_q88 = 0x{s&65535:04x}\n' for n,s in zip(('q','k','v'),scales)))
                        expected=fixed_attention(*matrices,scales,lut,causal)
                        (data/'expected.hex').write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in expected.flat))
                        results.append(check(binary,directory,kind,args.dim,name,data,causal))
    for name,digest in data_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Checked-in fixture changed: {name}')
    reports=SIM/'obj_dequant_reports'
    reports.mkdir(exist_ok=True)
    report=reports/f'report_l{args.lanes}_d{args.dim}.json'
    report.write_text(json.dumps(dict(lanes=args.lanes,dim=args.dim,smoke=args.smoke,
        verilator=subprocess.check_output(shlex.split(os.environ.get('VERILATOR','verilator'))+['--version'],text=True).strip(),
        results=results),indent=2)+'\n')
    print(f'RESULT: PASS {len(results)} exact core/top invocations; top invocations each include latency 0/20/100',flush=True)


if __name__=='__main__':
    main()
