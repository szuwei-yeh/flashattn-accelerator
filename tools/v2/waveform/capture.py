#!/usr/bin/env python3
"""Trace the existing focus regression without editing its checked-in bench/RTL.

Run verify.py first, then pass its fresh output directory as --regression.
All compilation, the instrumented bench, logs and VCD live in --output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reg = args.regression.resolve()
    record = json.loads((reg / 'results.json').read_text())
    assert record['status'] == 'PASS'
    assert (record['dim'], record['lanes'], record['compact']) == (16, 16, 1)
    assert record['latencies'] == [0] and record['seeds'] == [0]
    assert record['final_b_hold'] == 0
    assert (reg / 'focus_latency0_seed0.log').is_file()
    for name, digest in record['source_sha256'].items():
        assert sha(ROOT / name) == digest, f'Source changed since regression: {name}'
    src = out / 'source'
    names = (ROOT / 'rtl/v2/soc/soc.f').read_text().splitlines()
    names += ['rtl/v2/soc/lint.vlt', 'sim/v2/soc/tb_soc.sv',
              'sim/v2/soc/tb_soc.cpp', 'sim/v2/soc/timing.hpp',
              'sim/v2/common/axi_monitor.hpp', 'data/exp_lut.hex']
    for name in names:
        dest = src / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    cwd = src / 'sim/verilator'
    cwd.mkdir(parents=True)
    bench = src / 'sim/v2/soc/tb_soc.cpp'
    text = bench.read_text()
    replacements = {
        '#include "verilated.h"': '#include "verilated.h"\n#include "verilated_vcd_c.h"',
        'static Vtb_soc_m6b3 d;': 'static VerilatedVcdC* wave=nullptr;\nstatic bool wave_open=false;\nstatic Vtb_soc_m6b3 d;',
        'd.clk=1;d.eval();clocks++;d.clk=0;d.eval();':
            'if(wave_open)wave->dump(clocks*10000);\n'
            ' d.clk=1;d.eval();if(wave_open)wave->dump(clocks*10000+5000);\n'
            ' clocks++;d.clk=0;d.eval();\n'
            ' if(wave_open && job.p.op==2 && job.p.len==17 && job.terminal && clocks>=job.done_tick+64){\n'
            '  wave->dump(clocks*10000);wave->close();wave_open=false;\n'
            ' }',
        'configure(p);write(0x008,1,15,expected_id%3,6);':
            'configure(p);\n'
            ' if(p.op==2 && p.len==17 && p.pop==0){wave->open("' +
            str(out / 'decode.vcd') + '");wave_open=true;}\n'
            ' write(0x008,1,15,expected_id%3,6);',
        'Verilated::commandArgs(argc,argv);try{':
            'Verilated::commandArgs(argc,argv);Verilated::traceEverOn(true);\n'
            ' VerilatedVcdC wave_local;wave=&wave_local;d.trace(wave,5);try{',
    }
    for before, after in replacements.items():
        assert text.count(before) == 1, f'Unexpected testbench structure: {before}'
        text = text.replace(before, after)
    bench.write_text(text)
    commands = []

    def run(argv, name):
        argv = list(map(str, argv))
        commands.append(argv)
        with (out / name).open('w') as log:
            subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=300)

    run(['verilator', '--sv', '-Wall', '--top-module', 'tb_soc_m6b3',
         '-GCOMPACT_DECODE_OUTPUT=1', '-GDEQUANT_LANES=16', '-GHEAD_DIM=16',
         '-GEXPECTED_LANES=16', src / 'rtl/v2/soc/lint.vlt',
         '--trace', '--trace-depth', '5', '--trace-max-array', '1',
         '--cc', '--exe', '--build', '-j', '2', '--Mdir', out / 'obj',
         '--CFLAGS', '-DCOMPACT=1 -DLANES=16 -DDIM=16',
         *[src / p for p in (ROOT / 'rtl/v2/soc/soc.f').read_text().splitlines()],
         src / 'sim/v2/soc/tb_soc.sv', bench, '-o', 'sim_soc'], 'build.log')
    run([out / 'obj/sim_soc', reg / 'fixtures', '0', 'focus', '', '0', '0'], 'simulation.log')
    assert (out / 'simulation.log').read_bytes() == (reg / 'focus_latency0_seed0.log').read_bytes(), \
        'Tracing changed regression output'
    provenance = dict(status='PASS', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        tool=subprocess.check_output(['verilator', '--version'], text=True).strip(),
        commands=commands, config=dict(dim=16, lanes=16, compact=1, latency=0, seed=0,
        scope='focus', population=0, captured_job=3, sequence_length=17),
        clock_period_ps=10000, clock_note='Simulation-only trace timebase; not a timing/PPA result',
        original_bench_sha256=sha(ROOT / 'sim/v2/soc/tb_soc.cpp'),
        instrumented_bench_sha256=sha(bench), trace_sha256=sha(out / 'decode.vcd'),
        trace_bytes=(out / 'decode.vcd').stat().st_size,
        simulation_log_sha256=sha(out / 'simulation.log'),
        untraced_log_identical=True, source_sha256={n: sha(ROOT / n) for n in names},
        regression_results_sha256=sha(reg / 'results.json'))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('PASS: traced and untraced focus logs are byte-identical; ' + str(out / 'decode.vcd'))


if __name__ == '__main__':
    main()
