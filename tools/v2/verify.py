#!/usr/bin/env python3
"""Portable bounded V2 verification; fresh immutable source builds, fail-closed controls."""
import argparse, hashlib, importlib.util, json, os, pathlib, platform, re, shlex, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--compact', type=int, choices=[0, 1], default=1)
    parser.add_argument('--dim',type=int,choices=[16,64],default=16)
    parser.add_argument('--lanes', type=int, choices=[16,256], default=256)
    parser.add_argument('--scope', choices=['prefill', 'stateful', 'warm', 'stress', 'append_fault', 'focus'], default='prefill')
    parser.add_argument('--latencies', default='0')
    parser.add_argument('--scopes',default='')
    parser.add_argument('--python-command', default=os.environ.get('V2_PYTHON', ''),
                        help='Oracle interpreter command; defaults to this Python executable. Optional platform launcher allowed.')
    parser.add_argument('--seeds', default='0', help='Comma-separated deterministic timing seeds; zero preserves canonical timing.')
    parser.add_argument('--final-b-hold', type=int, default=0, help='Extra cycles before offering final B; bounded to 0..1000.')
    parser.add_argument('--test-control', choices=['expected', 'missing_fixture'], default='',
                        help='Deliberately fail the positive verification path; exits nonzero.')
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(',')))
    if not seeds or any(x < 0 or x > 4294967295 for x in seeds) or not 0 <= args.final_b_hold <= 1000:
        parser.error('invalid seed or final B hold')
    python_command = shlex.split(args.python_command) if args.python_command else [sys.executable]
    latencies = list(map(int, args.latencies.split(',')))
    if not latencies or any(x < 0 or x > 1000 for x in latencies):
        parser.error('latencies must be integers in [0,1000]')
    scopes = args.scopes.split(',') if args.scopes else [args.scope]
    if not scopes or any(s not in ['prefill','stateful','warm','stress','append_fault','focus'] for s in scopes):
        parser.error('unsupported verification scope')
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    work = pathlib.Path(tempfile.mkdtemp(prefix='flashattn-m7-'))
    src = work / 'source'
    paths = set((ROOT / 'rtl/v2/soc/soc.f').read_text().splitlines())
    paths.update(['tools/v2/verify.py','tools/v2/soc/run_mutants.py','tools/v2/soc/fixtures.py','rtl/v2/soc/soc.f', 'rtl/v2/soc/lint.vlt', 'sim/v2/soc/tb_soc.sv',
                  'sim/v2/soc/tb_soc.cpp', 'sim/v2/common/axi_monitor.hpp','sim/v2/soc/timing.hpp'])
    paths.update(str(p.relative_to(ROOT)) for folder in ['golden', 'data']
                 for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    for path in paths:
        dest = src / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    snapshot = out / 'source'
    for path in paths:
        if path.startswith(('rtl/v2/soc/', 'sim/v2/soc/')):
            dest = snapshot / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / path, dest)
    (src / 'sim/verilator').mkdir(parents=True)
    record = dict(dim=args.dim,lanes=args.lanes, expected_lanes=args.lanes,status='RUNNING', compact=args.compact, scope=args.scope, test_control=args.test_control,
                  platform=platform.platform(), python_executable=sys.executable, oracle_command=python_command,
                  latencies=latencies, seeds=seeds, final_b_hold=args.final_b_hold, source_sha256={p: sha(src / p) for p in sorted(paths)},
                  commands=[], jobs=[], sequences=[], results=[], safety=[], negative_controls=[],
                  model='coherent external byte store; child-relative matched read stalls/latency; '
                        'seed0 canonical mod9/mod11 AW/W stalls; nonzero seeds legal bounded stalls and extra B delay; final B hold explicitly recorded')

    def run(command, name, timeout=300, expected=None):
        raw = work / (name + '.log')
        start = time.monotonic()
        timed_out = False
        with raw.open('w') as handle:
            try:
                result = subprocess.run(list(map(str, command)), cwd=src / 'sim/verilator',
                                        stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                result = subprocess.CompletedProcess(command, 124)
        text = raw.read_text().replace(str(work), '<WORK>').replace(str(ROOT), '<ROOT>')
        (out / (name + '.log')).write_text(text)
        record['commands'].append(dict(argv=[str(x).replace(str(work), '<WORK>').replace(str(ROOT), '<ROOT>') for x in command],
                                       exit=result.returncode, seconds=round(time.monotonic() - start, 3)))
        if timed_out:
            raise RuntimeError(name + ': timeout; partial log preserved')
        if expected:
            if result.returncode == 0 or expected not in text or 'RESULT PASS M5D' in text:
                raise RuntimeError(name + ': control not meaningfully rejected: ' + text[-2000:])
            record['negative_controls'].append(dict(name=name, marker=expected, exit=result.returncode,
                diagnostic=next(line for line in text.splitlines() if expected in line)))
            return text
        if result.returncode:
            raise RuntimeError(name + ' failed: ' + text[-1500:])
        return text

    try:
        spec = importlib.util.spec_from_file_location('accepted_fixture_generator', src / 'tools/v2/soc/fixtures.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        fixtures = out / 'fixtures'
        helper.fixtures(fixtures, 4, 2,args.dim)
        record['fixture_sha256'] = {str(p.relative_to(fixtures)): sha(p) for p in fixtures.rglob('*') if p.is_file()}
        run(python_command + ['-B', src / 'golden/v2/stateful/test_reference.py'], 'oracle')
        sources = [src / p for p in (src / 'rtl/v2/soc/soc.f').read_text().splitlines()]
        sources += [src / 'sim/v2/soc/tb_soc.sv']
        base = ['verilator', '--sv', '-Wall', '--top-module', 'tb_soc_m6b3',
                f'-GCOMPACT_DECODE_OUTPUT={args.compact}', f'-GDEQUANT_LANES={args.lanes}',f'-GHEAD_DIM={args.dim}',f'-GEXPECTED_LANES={args.lanes}', src / 'rtl/v2/soc/lint.vlt']
        run(base + ['--lint-only', *sources], 'lint')
        run(base + ['--json-only','--json-only-output',work/'hierarchy.json','--json-only-meta-output',work/'hierarchy.meta.json',*sources], 'inventory')
        tree=json.loads((work/'hierarchy.json').read_text())
        mods={m['addr']:m for m in tree['modulesp']}
        top=next(m for m in mods.values() if m.get('level')==1 and m['origName']=='tb_soc_m6b3')
        counts={}
        def nodes(x):
            if isinstance(x,dict):
                yield x
                for v in x.values(): yield from nodes(v)
            elif isinstance(x,list):
                for v in x: yield from nodes(v)
        def descend(m):
            typ=m['origName'];counts[typ]=counts.get(typ,0)+1
            for cell in nodes(m):
                if cell.get('type')=='CELL':descend(mods[cell['modp']])
        descend(top)
        if counts['dequantizer'] != args.lanes or counts['pe'] != 256:
            raise RuntimeError('incorrect physical resource selection')
        if counts['kv_cache_banked_decode_m2c3'] != 2 or counts['sram_1r1w'] != 81 or counts['exp_lut'] != 16:
            raise RuntimeError('incorrect cache/memory hierarchy')
        (out/'inventory.json').write_text(json.dumps(dict(status='PASS',physical_dequantizers=counts['dequantizer'],module_counts=counts,tree_sha256=sha(work/'hierarchy.json')),indent=2)+'\n')
        run(base + ['--cc', '--exe', '--build', '-j', '2', '--Mdir', work / 'obj',
                    '--CFLAGS', f'-DCOMPACT={args.compact} -DLANES={args.lanes} -DDIM={args.dim}', *sources,
                    src / 'sim/v2/soc/tb_soc.cpp', '-o', 'sim_soc'], 'build')
        binary = work / 'obj/sim_soc'
        record['binary_sha256'] = sha(binary)
        for selected_scope, latency, seed in [(s,l,seed) for s in scopes for l in latencies for seed in seeds]:
            print('RUN M7', selected_scope, 'dim', args.dim, 'lanes', args.lanes, 'latency', latency, flush=True)
            input_dir = work/'empty-fixtures' if args.test_control == 'missing_fixture' else fixtures.resolve()
            input_dir.mkdir(exist_ok=True) if args.test_control == 'missing_fixture' else None
            text = run([binary, input_dir, latency, selected_scope, args.test_control, seed, args.final_b_hold], f'{selected_scope}_latency{latency}_seed{seed}')
            markers = re.findall(r'^RESULT PASS M5D (.+)$', text, re.M)
            if len(markers) != 1:
                raise RuntimeError('missing or duplicate simulator PASS record')
            for prefix, key in [('HOST_JOB', 'jobs'), ('SEQUENCE', 'sequences'), ('RESULT PASS M5D', 'results'), ('M7_SAFETY', 'safety')]:
                record[key] += [{k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
                                for line in re.findall(r'^' + prefix + r' (.+)$', text, re.M)]
            if len(record['safety']) != len(record['results']):
                raise RuntimeError('missing safety sample record')
            result = record['results'][-1]
            if result.get('successful', 0) < 1 or result.get('exact_words', 0) < 4*args.dim:
                raise RuntimeError('no meaningful exact transactions in simulator PASS')
        empty = work/'empty-fixtures'
        empty.mkdir(exist_ok=True)
        for name, directory, control, marker in [
            ('wrong_final_word', fixtures.resolve(), 'expected', 'M5C_CHECK corrupt/missing/duplicated byte or wrong head/word order'),
            ('missing_fixture', empty, '', 'M5D_CHECK missing fixture')]:
            run([binary, directory, 0, 'prefill', control], 'negative_'+name, expected=marker)
        record['status'] = 'PASS'
    except Exception as error:
        record['status'] = 'FAIL'
        record['failure'] = str(error)
        raise
    finally:
        try:
            record['tool'] = subprocess.check_output(['verilator', '--version']).decode().strip()
        except Exception:
            record['tool'] = 'UNAVAILABLE'
        (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    print('M7 ' + record['status'], sum(r['successful'] for r in record['results']), 'commands',
          sum(r['exact_words'] for r in record['results']), 'exact words', flush=True)

if __name__ == '__main__':
    main()
