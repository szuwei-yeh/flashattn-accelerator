#!/usr/bin/env python3
"""Check accepted edges in the bounded simulator VCD; emit measured event times."""
import argparse
import json
from pathlib import Path


def read_vcd(path):
    scopes, names, values, samples = [], {}, {}, []
    now = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == '$scope':
            scopes.append(parts[2])
        elif parts[0] == '$upscope':
            scopes.pop()
        elif parts[0] == '$var':
            names['.'.join(scopes + [parts[4]])] = parts[3]
        elif line.startswith('#'):
            if now is not None:
                samples.append((now, values.copy()))
            now = int(line[1:])
        elif line[0] in '01':
            values[line[1:]] = int(line[0])
        elif line[0] == 'b':
            value, code = parts
            values[code] = int(value[1:], 2) if set(value[1:]) <= {'0', '1'} else None
    if now is not None:
        samples.append((now, values.copy()))
    return names, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vcd', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names, samples = read_vcd(args.vcd)
    top = 'TOP.'
    engine = 'TOP.tb_soc_m6b3.u_dut.u_engine.'

    def val(values, name):
        return values[names[name if name.startswith('TOP.') else top + name]]

    # Handshakes are sampled BEFORE the rising edge; post-edge values may
    # already describe the next transfer. VCD uses 10 ns cycles / 5 ns halves.
    edges = []
    for (pre_time, pre), (time, post) in zip(samples, samples[1:]):
        if val(pre, 'clk') == 0 and val(post, 'clk') == 1:
            assert time - pre_time == 5000
            edges.append((time, pre, post))
    assert edges
    command = [t for t, pre, _ in edges if val(pre, 'obs_command')]
    assert len(command) == 1
    start = command[0]
    done = [t for t, pre, post in edges if val(pre, 'obs_phase') != 4 and val(post, 'obs_phase') == 4]
    assert len(done) == 1
    done = done[0]
    active = [(t, pre, post) for t, pre, post in edges if start <= t <= done]
    for _, pre, _ in active:
        assert val(pre, 'rst_n') == 1 and val(pre, 'cache_valid') == 3
        assert val(pre, engine + 'g_cache[0].u_cache.cache_length') == 17
        assert val(pre, engine + 'g_cache[1].u_cache.cache_length') == 17
        assert val(pre, 'obs_cache_write') == 0
        if val(pre, 'obs_core_active'):
            assert val(pre, 'obs_kvhead') == val(pre, 'obs_qhead') // 2
    reads = [(t, val(pre, 'm_araddr')) for t, pre, _ in active
             if val(pre, 'm_arvalid') and val(pre, 'm_arready')]
    assert [a for _, a in reads] == [0x1000, 0x2000, 0x3000, 0x4000]

    def handshake(valid, ready):
        return [t for t, pre, _ in active if val(pre, valid) and val(pre, ready)]

    aw = handshake('m_awvalid', 'm_awready')
    w = handshake('m_wvalid', 'm_wready')
    b = handshake('m_bvalid', 'm_bready')
    r = handshake('m_rvalid', 'm_rready')
    assert (len(aw), len(w), len(b), len(r)) == (3, 32, 3, 8)
    assert all(val(pre, 'm_bresp') == 0 for _, pre, _ in active
               if val(pre, 'm_bvalid') and val(pre, 'm_bready'))
    assert sum(val(pre, 'm_wlast') for _, pre, _ in active
               if val(pre, 'm_wvalid') and val(pre, 'm_wready')) == 3
    assert done > b[-1] and (done - start) // 10000 == 2072
    starts = [(t, val(pre, 'obs_qhead')) for t, pre, _ in active if val(pre, 'obs_core_start')]
    ends = [(t, val(pre, 'obs_qhead')) for t, pre, _ in active if val(pre, 'obs_core_done')]
    assert [h for _, h in starts] == [0, 1, 2, 3]
    assert [h for _, h in ends] == [0, 1, 2, 3]
    assert all(a[0] < z[0] for a, z in zip(starts, ends))
    assert all(z[0] < a[0] for z, a in zip(ends, starts[1:]))
    assert sum(val(pre, 'obs_array_start') for _, pre, _ in active) == 16
    assert sum(val(pre, 'obs_cache_read') for _, pre, _ in active) == 68
    phases = sorted({val(pre, engine + 'u_core.u_fsm.state') for _, pre, _ in active})
    assert {1, 2, 3, 4, 6, 7, 9, 11} <= set(phases)
    assert val(active[-1][2], 'irq') == 1
    assert val(active[-1][2], 'obs_output_committed') == 1
    assert val(active[-1][2], 'obs_output_bytes') == 256
    result = dict(status='PASS', time_unit='ps', cycle_ps=10000,
                  trace_start_ps=samples[0][0], trace_end_ps=samples[-1][0],
                  command_accept_ps=start, done_ps=done, host_cycles=(done-start)//10000,
                  final_b_accept_ps=b[-1], done_after_final_b_cycles=(done-b[-1])//10000,
                  q_ar=[dict(time_ps=t, address=a) for t, a in reads],
                  core_start=[dict(time_ps=t, head=h) for t, h in starts],
                  core_done=[dict(time_ps=t, head=h) for t, h in ends],
                  aw_accept_ps=aw, b_accept_ps=b, w_beats=len(w), r_beats=len(r),
                  cache_reads=68, cache_writes=0, persistent_length=17,
                  tile_states_observed=phases, golden_words=64)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
