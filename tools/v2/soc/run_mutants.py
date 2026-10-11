#!/usr/bin/env python3
"""Local real host->compute->memory M5D tests; sequential immutable attempts."""
import argparse, hashlib, importlib.util, json, pathlib, re, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
MUTANTS = {
 'incomplete_score': ('rtl/quantization/dequantizer_tile.sv', "if (retire_group == GROUP_W'(GROUPS-1)) begin", "if (retire_group == GROUP_W'(GROUPS-2)) begin", 'stateful', 'M6B1_SCORE incomplete score publication'),
 'mapping': ('rtl/v2/soc/flash_attn_serial.sv', "assign selected_kv=2'(int'(q_head)/GROUP);", "assign selected_kv=2'(int'(q_head)/(GROUP-1));", 'prefill', 'M4_SAFETY wrong KV mapping'),
 'missing_append': ('rtl/v2/soc/flash_attn_serial.sv', ".w_en(w_en && w_dst!=0 && load_head==2'(h))", ".w_en(w_en && w_dst!=0 && r_op!=1 && load_head==2'(h))", 'stateful', 'M4_SAFETY missing accepted DMA cache write'),
 'stale_cache': ('rtl/v2/cache/kv_cache_banked_decode.sv', '.w_en(write_accept && !w_sel)', ".w_en(write_accept && !w_sel && cache_epoch==0)", 'stateful', 'M4_SAFETY stale cache data'),
 'head_address': ('rtl/v2/soc/gqa_output_packer.sv', "out_head<=out_head+2'd1;", "out_head<=out_head+2'd2;", 'prefill', 'M5C_CHECK corrupt/missing/duplicated byte or wrong head/word order'),
 'payload': ('rtl/v2/soc/gqa_output_packer.sv', 'src_data<={out_data,low_word};', "src_data<={out_data,low_word^32'd1};", 'prefill', 'M5C_CHECK corrupt/missing/duplicated byte or wrong head/word order'),
 'premature_done': ('rtl/v2/soc/soc_job_adapter.sv', '(phase==WRITEBACK && source_seen && response_seen && !wb_outstanding));', '(phase==WRITEBACK && !wb_outstanding));', 'prefill', 'M5D_CHECK host DONE before successful writeback/source retirement'),
 'irq_identity': ('rtl/v2/soc/axi_lite_csr.sv', 'assign new_events={reject_event,fatal_event,success_event};', 'assign new_events={reject_event,success_event,fatal_event};', 'prefill', 'M5D_CHECK completion IRQ pending identity'),
 'early_ack': ('rtl/v2/soc/soc_job_adapter.sv', 'compute_done<=1;pack_start<=1;', 'compute_done<=1;pack_start<=1;child_ack<=1;', 'prefill', 'M5D_CHECK premature/wrong-owner child ACK'),
 'wstrb': ('rtl/v2/interface/dma_write_engine.sv', "assign m_wstrb=8'hff;", "assign m_wstrb=8'hfe;", 'prefill', 'M5C_CHECK WSTRB'),
 'wlast': ('rtl/v2/interface/dma_write_engine.sv', "assign m_wlast=sent==beats-5'd1;", "assign m_wlast=sent==beats-5'd2;", 'prefill', 'M5C_CHECK incorrect WLAST'),
 'ignored_bresp': ('rtl/v2/interface/dma_write_engine.sv', 'if(m_bresp!=0)begin', 'if(m_bresp==3)begin', 'stress', 'M5D_CHECK'),
 'page_split': ('rtl/v2/interface/dma_write_engine.sv', 'if(page_beats<result)result=page_beats;', "if(page_beats<result)result=32'd16;", 'prefill', 'M5C_CHECK AW address/length or head destination'),
 'b_accounting': ('rtl/v2/interface/dma_write_engine.sv', "perf_b<=perf_b+1'b1;", "perf_b<=perf_b+32'd2;", 'prefill', 'M5D_CHECK genuine unexpected integrated fatal'),
 'atomic_publish': ('rtl/v2/cache/kv_cache_banked_decode.sv', "if (write_accept && !w_sel) k_stored<=k_stored+1'b1;", "if (write_accept && !w_sel) begin k_stored<=k_stored+1'b1; if(op==APPEND) cache_length_reg<=pending_length+7'd1; end", 'stateful', 'M4_SAFETY premature KV publication'),
 'irq_race': ('rtl/v2/soc/axi_lite_csr.sv', '(irq_pending & ~irq_clear) | new_events', '(irq_pending | new_events) & ~irq_clear', 'stress', 'M5D_CHECK host completion/IRQ association'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--compact', type=int, choices=[0, 1], default=1)
    parser.add_argument('--dim',type=int,choices=[16,64],default=16)
    parser.add_argument('--lanes', type=int, choices=[16,256], default=256)
    parser.add_argument('--expected-lanes',type=int,choices=[16,256])
    parser.add_argument('--scope', choices=['prefill', 'stateful', 'warm', 'stress', 'append_fault', 'focus'], default='prefill')
    parser.add_argument('--latencies', default='0')
    parser.add_argument('--scopes',default='')
    parser.add_argument('--control', default='')
    parser.add_argument('--mutant', choices=sorted(MUTANTS))
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    work = pathlib.Path(tempfile.mkdtemp(prefix='flashattn-m6b3-'))
    src = work / 'source'
    paths = set((ROOT / 'rtl/v2/soc/soc.f').read_text().splitlines())
    paths.update(['tools/v2/soc/run_mutants.py','tools/v2/soc/fixtures.py','rtl/v2/soc/soc.f', 'rtl/v2/soc/lint.vlt', 'sim/v2/reference/tb_soc.sv',
                  'sim/v2/reference/tb_soc.cpp', 'sim/v2/common/axi_monitor.hpp'])
    paths.update(str(p.relative_to(ROOT)) for folder in ['golden', 'data']
                 for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    for path in paths:
        dest = src / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
        if args.mutant and path == MUTANTS[args.mutant][0]:
            _, old, new, scope, marker = MUTANTS[args.mutant]
            content = dest.read_text()
            assert content.count(old) == 1, (args.mutant, 'nonunique mutation')
            dest.write_text(content.replace(old, new))
            (out / 'mutation.json').write_text(json.dumps(dict(path=path, before=old, after=new,
                original_sha256=sha(ROOT / path), mutant_sha256=sha(dest)), indent=2) + '\n')
    snapshot = out / 'source'
    for path in paths:
        if path.startswith(('rtl/v2/soc/', 'sim/v2/reference/')) or args.mutant and path == MUTANTS[args.mutant][0]:
            dest = snapshot / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / path, dest)
    (src / 'sim/verilator').mkdir(parents=True)
    latencies = list(map(int, args.latencies.split(',')))
    record = dict(dim=args.dim,lanes=args.lanes, expected_lanes=args.expected_lanes or args.lanes,status='RUNNING', compact=args.compact, scope=args.scope, control=args.control, mutant=args.mutant,
                  latencies=latencies, source_sha256={p: sha(src / p) for p in sorted(paths)},
                  commands=[], jobs=[], sequences=[], results=[],
                  model='coherent external byte store; child-relative matched read stalls/latency; '
                        'descriptor-relative mod9/mod11 AW/W stalls; B delay equals read latency')

    def run(command, name, timeout=300, expected=None):
        raw = work / (name + '.log')
        start = time.monotonic()
        with raw.open('w') as handle:
            result = subprocess.run(list(map(str, command)), cwd=src / 'sim/verilator',
                                    stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        text = raw.read_text().replace(str(work), '<WORK>').replace(str(ROOT), '<ROOT>')
        (out / (name + '.log')).write_text(text)
        record['commands'].append(dict(argv=[str(x).replace(str(work), '<WORK>').replace(str(ROOT), '<ROOT>') for x in command],
                                       exit=result.returncode, seconds=round(time.monotonic() - start, 3)))
        if expected:
            assert result.returncode != 0 and expected in text and 'RESULT PASS M5D' not in text, (name, 'control not meaningfully rejected', text[-2000:])
            record['failure_control'] = dict(marker=expected, exit=result.returncode,
                diagnostic=next(line for line in text.splitlines() if expected in line))
            return text
        if result.returncode:
            raise RuntimeError(name + ' failed: ' + text[-1500:])
        return text

    try:
        spec = importlib.util.spec_from_file_location('accepted_fixture_generator', ROOT / 'tools/v2/soc/fixtures.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        fixtures = out / 'fixtures'
        helper.fixtures(fixtures, 4, 2,args.dim)
        record['fixture_sha256'] = {str(p.relative_to(fixtures)): sha(p) for p in fixtures.rglob('*') if p.is_file()}
        run(['arch','-x86_64','/usr/local/bin/python3', '-B', src / 'golden/v2/stateful/test_reference.py'], 'oracle')
        sources = [src / p for p in (src / 'rtl/v2/soc/soc.f').read_text().splitlines()]
        sources += [src / 'sim/v2/reference/tb_soc.sv']
        base = ['verilator', '--sv', '-Wall', '--top-module', 'tb_soc_m6b3',
                f'-GCOMPACT_DECODE_OUTPUT={args.compact}', f'-GDEQUANT_LANES={args.lanes}',f'-GHEAD_DIM={args.dim}',f'-GEXPECTED_LANES={args.expected_lanes or args.lanes}', src / 'rtl/v2/soc/lint.vlt']
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
        assert counts['dequantizer']==args.lanes and counts['pe']==256
        assert counts['kv_cache_banked_decode_m2c3']==2 and counts['sram_1r1w']==81 and counts['exp_lut']==16
        (out/'inventory.json').write_text(json.dumps(dict(status='PASS',physical_dequantizers=counts['dequantizer'],module_counts=counts,tree_sha256=sha(work/'hierarchy.json')),indent=2)+'\n')
        run(base + ['--cc', '--exe', '--build', '-j', '2', '--Mdir', work / 'obj',
                    '--CFLAGS', f'-DCOMPACT={args.compact} -DLANES={args.lanes} -DDIM={args.dim}', *sources,
                    src / 'sim/v2/reference/tb_soc.cpp', '-o', 'sim_soc'], 'build')
        for selected_scope,latency in [(s,l) for s in (args.scopes.split(',') if args.scopes else [args.scope]) for l in latencies]:
            args.scope=selected_scope
            if args.mutant:
                assert args.scope == MUTANTS[args.mutant][3]
            expected = 'M6B1_SCORE incorrect physical resource selection' if args.expected_lanes and args.expected_lanes!=args.lanes else MUTANTS[args.mutant][4] if args.mutant else 'M5C_CHECK corrupt/missing/duplicated byte or wrong head/word order' if args.control == 'expected' else None
            print('RUN M5D', args.scope, 'compact', args.compact, 'latency', latency, flush=True)
            text = run([work / 'obj/sim_soc', fixtures.resolve(), latency, args.scope, args.control], f'{selected_scope}_latency{latency}', expected=expected)
            if expected:
                continue
            assert 'RESULT PASS M5D' in text
            for prefix, key in [('HOST_JOB', 'jobs'), ('SEQUENCE', 'sequences'), ('RESULT PASS M5D', 'results')]:
                record[key] += [{k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
                                for line in re.findall(r'^' + prefix + r' (.+)$', text, re.M)]
        record['status'] = 'EXPECTED_FAIL' if args.mutant or args.control == 'expected' or (args.expected_lanes and args.expected_lanes!=args.lanes) else 'PASS'
    except Exception as error:
        record['status'] = 'FAIL'
        record['failure'] = str(error)
        raise
    finally:
        record['tool'] = subprocess.check_output(['verilator', '--version']).decode().strip()
        (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    print('M5D ' + record['status'], sum(r['successful'] for r in record['results']), 'commands',
          sum(r['exact_words'] for r in record['results']), 'exact words', flush=True)

if __name__ == '__main__':
    main()
