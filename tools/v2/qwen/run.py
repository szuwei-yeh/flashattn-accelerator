#!/usr/bin/env python3
"""Bounded frozen-Qwen production SoC verification; all artifacts outside Git.

Example: python -B tools/v2/qwen/run.py --output-dir OUT
Requires Verilator 5.046, C++/make and NumPy. Never downloads weights or installs tools.
"""
import argparse
import csv
import importlib.util
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time

from fixtures import prepare, sha

ROOT = pathlib.Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def records(text, prefix):
    return [{k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)', line)}
            for line in re.findall(r'^'+re.escape(prefix)+r' (.+)$',text,re.M)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-dir',type=pathlib.Path,default=ROOT/'data/qwen2_5_layer11',
                   help='Frozen hash-pinned public dataset; defaults to the checked-in fixtures.')
    p.add_argument('--output-dir',type=pathlib.Path,required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        p.error('output directory must be outside the repository')
    out.mkdir(parents=True,exist_ok=False)
    src = out/'source'
    spec = importlib.util.spec_from_file_location('unchanged_release_validator',ROOT/'tools/v2/validate_release.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    payload = checks.payload(ROOT)
    hashes = {str(f.relative_to(ROOT)):sha(f) for f in payload}
    for f in payload:
        dest = src/f.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest)
    cwd = src/'sim/verilator'
    record = {'status':'RUNNING','scope':'frozen Qwen N16/APPEND/L17/APPEND/L18 production RTL only',
              'head_dim':64,'dequant_lanes':256,'compact_decode_output':1,'axi_latency':0,'timing_seed':0,
              'platform':platform.platform(),'python':sys.version,'commands':[],'negative_controls':[],
              'source_sha256':hashes,'output_word_format':'signed INT32 little-endian, job then head then query then feature',
              'phase2a_accuracy_scope':'unchanged prior numerical measurements; not new model inference',
              'cycle_scope':'modeled RTL cycles; serial-engine counter includes DMA/cache/compute/copy, not pure MAC time'}
    if (ROOT/'.git').exists():
        record['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        record['git_status_before'] = subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)

    def run(command, name, timeout=300, negative=None):
        start = time.monotonic()
        print('RUN',name,flush=True)
        log = out/(name+'.log')
        timed_out = False
        with log.open('w') as f:
            try:
                result = subprocess.run(list(map(str,command)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                code = 124
        raw = log.read_text()
        sanitized = raw.replace(str(out),'<OUT>').replace(str(ROOT),'<ROOT>')
        log.write_text(sanitized)
        record['commands'].append({'name':name,'argv':[str(x).replace(str(out),'<OUT>').replace(str(ROOT),'<ROOT>') for x in command],
                                   'exit':code,'seconds':round(time.monotonic()-start,3)})
        if timed_out:
            raise RuntimeError(name+': timeout; partial log retained')
        if negative:
            if code == 0 or negative not in raw or 'RESULT PASS QWEN' in raw:
                raise RuntimeError(name+': negative control not meaningfully rejected')
            record['negative_controls'].append({'name':name,'exit':code,'detector':negative,
                'diagnostic':next(line for line in sanitized.splitlines() if negative in line)})
        elif code:
            raise RuntimeError(name+' failed: '+sanitized[-1800:])
        return sanitized

    try:
        run([sys.executable,'-B',src/'tools/v2/validate_release.py','--snapshot',src,
             '--output',out/'integrity.json'],'integrity')
        record['integrity'] = json.loads((out/'integrity.json').read_text())
        run([sys.executable,'-B',src/'tools/v2/test_validate_release.py'],'integrity_controls')
        dataset = args.fixture_dir.resolve()
        if ROOT in dataset.parents:
            dataset = src/dataset.relative_to(ROOT)
        record['fixture_manifest'] = prepare(dataset,out,src)
        run([sys.executable,'-B',src/'golden/v2/stateful/test_reference.py'],'oracle_tests')
        sources = [src/n for n in (src/'rtl/v2/soc/soc.f').read_text().splitlines()]
        record['production_rtl_sha256'] = {str(f.relative_to(src)):sha(f) for f in sources}
        record['oracle_lut_sha256'] = {n:sha(src/n) for n in (
            'golden/v2/stateful/decode_reference.py','golden/v2/decode_reference.py','data/exp_lut.hex')}
        sources.append(src/'sim/v2/soc/tb_soc.sv')
        base = ['verilator','--sv','-Wall','--top-module','tb_soc_m6b3',
                '-GHEAD_DIM=64','-GDEQUANT_LANES=256','-GEXPECTED_LANES=256','-GCOMPACT_DECODE_OUTPUT=1',src/'rtl/v2/soc/lint.vlt']
        run(base+['--lint-only',*sources],'lint')
        run(base+['--json-only','--json-only-output',out/'hierarchy.json',
                  '--json-only-meta-output',out/'hierarchy.meta.json',*sources],'inventory')
        tree = json.loads((out/'hierarchy.json').read_text())
        mods = {m['addr']:m for m in tree['modulesp']}
        top = next(m for m in mods.values() if m.get('level')==1 and m['origName']=='tb_soc_m6b3')
        counts = {}

        def nodes(x):
            if isinstance(x,dict):
                yield x
                for v in x.values(): yield from nodes(v)
            elif isinstance(x,list):
                for v in x: yield from nodes(v)

        def descend(m):
            name = m['origName']
            counts[name] = counts.get(name,0)+1
            for cell in nodes(m):
                if cell.get('type')=='CELL': descend(mods[cell['modp']])

        descend(top)
        for name,count in {'dequantizer':256,'pe':256,'kv_cache_banked_decode_m2c3':2,
                           'sram_1r1w':81,'exp_lut':16,'flash_attn_core_decode_m3b':1}.items():
            if counts.get(name) != count:
                raise RuntimeError('unexpected production hierarchy: '+name)
        record['module_counts'] = counts
        write_json(out/'inventory.json',{'status':'PASS','module_counts':counts,'tree_sha256':sha(out/'hierarchy.json')})
        run(base+['--cc','--exe','--build','-j','2','--Mdir',out/'obj',
                  '--CFLAGS','-DCOMPACT=1 -DLANES=256 -DDIM=64',*sources,
                  src/'sim/v2/qwen/tb_qwen.cpp','-o','sim_qwen'],'build',timeout=600)
        binary = out/'obj/sim_qwen'
        record['binary_sha256'] = sha(binary)
        outputs = out/'outputs'
        outputs.mkdir()
        text = run([binary,out/'fixtures',outputs,''],'qwen')
        result = records(text,'RESULT PASS QWEN')
        jobs = records(text,'HOST_JOB')
        safety = records(text,'M7_SAFETY')
        if len(result)!=1 or result[0].get('successful')!=5 or result[0].get('exact_words')!=4608 or result[0].get('mismatches')!=0:
            raise RuntimeError('missing or incomplete exact sequence PASS')
        if len(safety)!=1 or safety[0]['completion_edges']!=3 or safety[0]['publication_edges']!=3:
            raise RuntimeError('missing required completion/publication safety observations')
        shape = [(0,16,8192,16384,4096),(1,16,256,0,0),(2,17,256,1024,256),(1,17,256,0,0),(2,18,256,1024,256)]
        if [tuple(j[k] for k in ('op','length','read_bytes','write_bytes','words')) for j in jobs] != shape:
            raise RuntimeError('unexpected command/output/traffic counts')
        for j in jobs:
            if j['host_cycles']<=0 or j['compute_cycles']<=0 or j['aw']!=j['b'] or j['b']!=j['okay']:
                raise RuntimeError('invalid measured cycle or final-response counters')
        record.update({'jobs':jobs,'sequence':records(text,'SEQUENCE'),'safety':safety,'comparison':result[0],
                       'output_sha256':{}})
        actual = b''
        for name,size in (('N16',4096),('L17',256),('L18',256)):
            raw = (outputs/(name+'.bin')).read_bytes()
            expected = (out/('expected_'+name+'.bin')).read_bytes()
            if len(raw)!=size*4 or raw!=expected:
                raise RuntimeError('exported external DRAM differs from independent oracle: '+name)
            record['output_sha256'][name] = sha(outputs/(name+'.bin'))
            actual += raw
        (outputs/'all.bin').write_bytes(actual)
        record['output_sha256']['all'] = sha(outputs/'all.bin')
        if record['output_sha256']['all'] != record['fixture_manifest']['expected_word_stream_sha256']:
            raise RuntimeError('actual output stream SHA256 mismatch')
        with (out/'cycles.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(jobs[0]))
            writer.writeheader()
            writer.writerows(jobs)
        bad = out/'negative_wrong_final_word_outputs'
        bad.mkdir()
        text = run([binary,out/'fixtures',bad,'expected'],'negative_wrong_final_word',
                   negative='M5C_CHECK corrupt/missing/duplicated byte or wrong head/word order')
        if 'word_index=4607 prior_exact_words=4352' not in text or 'job=5' not in text:
            raise RuntimeError('wrong-final-word control did not reach the final sequence word')
        empty = out/'empty_fixtures'
        empty.mkdir()
        bad = out/'negative_missing_fixture_outputs'
        bad.mkdir()
        run([binary,empty,bad,''],'negative_missing_fixture',negative='M5D_CHECK missing fixture')
        current = {str(f.relative_to(ROOT)):sha(f) for f in checks.payload(ROOT)}
        if current != hashes:
            raise RuntimeError('working source payload changed during verification')
        if any(sha(src/n)!=h for n,h in hashes.items()):
            raise RuntimeError('snapshot source changed during verification')
        if any(sha(out/'fixtures'/n)!=h for n,h in record['fixture_manifest']['fixture_sha256'].items()):
            raise RuntimeError('fixtures changed during simulation')
        record['source_payload_unchanged'] = True
        record['status'] = 'PASS'
    except Exception as e:
        record.update({'status':'FAIL','failure':str(e)})
        raise
    finally:
        record['verilator'] = subprocess.check_output(['verilator','--version'],text=True).strip()
        record['report_files_sha256'] = {n:sha(out/n) for n in (
            'fixture_manifest.json','integrity.json','cycles.csv','qwen.log','negative_wrong_final_word.log','negative_missing_fixture.log') if (out/n).exists()}
        write_json(out/'results.json',record)
    print('RESULT PASS QWEN_RUNNER exact_words=4608 mismatches=0',flush=True)


if __name__ == '__main__':
    main()
