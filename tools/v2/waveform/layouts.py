#!/usr/bin/env python3
"""Create Surfer layouts from measured events and existing VCD signal names."""
import argparse
import json
from pathlib import Path
from check_trace import read_vcd

TOP = 'TOP.'
DUT = TOP + 'tb_soc_m6b3.u_dut.'
ENGINE = DUT + 'u_engine.'
CORE = ENGINE + 'u_core.'
# (signal, displayed label, format); aliases are observations in tb_soc.sv.
HOST = [('clk', 'clk', 'bit'), ('rst_n', 'rst_n', 'bit'),
        ('obs_command', 'cmd_accept', 'bit'),
        (DUT+'u_job.owned', 'job_owned', 'bit'),
        ('obs_phase', 'job_phase', 'JobPhase'), ('obs_irq_pending', 'IRQ_pending', 'unsigned')]
CACHE = [(ENGINE+'desc_valid', 'Q_DMA_req', 'bit'),
         ('m_araddr', 'Q_ARADDR', 'hex'),
         ('obs_ar', 'AR_accept', 'bit'), ('obs_r', 'R_accept', 'bit'),
         (ENGINE+'dma_done', 'Q_DMA_done', 'bit'),
         (ENGINE+'g_cache[0].u_cache.cache_length', 'KV0_length', 'unsigned'),
         (ENGINE+'g_cache[1].u_cache.cache_length', 'KV1_length', 'unsigned'),
         ('obs_kvhead', 'kv_group', 'unsigned'),
         ('obs_cache_read', 'KV_rd_accept', 'bit'),
         ('obs_cache_write', 'KV_wr_accept', 'bit')]
COMPUTE = [('obs_qhead', 'q_head', 'unsigned'),
           ('obs_core_active', 'core_active', 'bit'),
           (CORE+'u_fsm.state', 'tile_state', 'TilePhase'),
           (CORE+'tile_col', 'KV_base', 'unsigned'),
           (CORE+'gen_shared_dequant.batch_busy', 'dq_busy', 'bit'),
           (CORE+'gen_softmax[0].active_row.u_softmax.exp_out_valid', 'sfx_exp_valid', 'bit'),
           (CORE+'fused_update_en', 'update_en', 'bit'),
           (CORE+'norm_en', 'norm_en', 'bit')]
OUTPUT = [('obs_compute_done', 'compute_done', 'bit'),
          ('obs_pack_start', 'pack_start', 'bit'),
          ('m_awvalid', 'AWVALID', 'bit'), ('m_wvalid', 'WVALID', 'bit'),
          ('m_bvalid', 'BVALID', 'bit'), ('obs_wb_done', 'wb_final_OK', 'bit'),
          ('obs_output_committed', 'DONE', 'bit')]
DETAIL = [('clk', 'clk', 'bit'), ('obs_phase', 'job_phase', 'JobPhase'),
          (DUT+'u_job.source_seen', 'source_seen', 'bit'),
          ('m_awvalid', 'AWVALID', 'bit'), ('m_awready', 'AWREADY', 'bit'),
          ('m_awaddr', 'AWADDR', 'hex'), ('m_awlen', 'AWLEN', 'unsigned'),
          ('m_wvalid', 'WVALID', 'bit'), ('m_wready', 'WREADY', 'bit'),
          ('m_wlast', 'WLAST', 'bit'), ('m_bvalid', 'BVALID', 'bit'),
          ('m_bready', 'BREADY', 'bit'), ('m_bresp', 'BRESP', 'unsigned'),
          ('obs_wb_done', 'wb_final_OK', 'bit'),
          ('obs_wb_busy', 'wb_busy', 'bit'),
          ('obs_output_committed', 'DONE', 'bit'), ('irq', 'IRQ', 'bit')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vcd', type=Path)
    parser.add_argument('events', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names, _ = read_vcd(args.vcd)
    events = json.loads(args.events.read_text())
    assert events['status'] == 'PASS'
    args.output.mkdir(parents=True, exist_ok=True)

    def layout(groups, markers, start, end, name):
        commands, items = [], []
        for group, color, signals in groups:
            commands.append('divider_add group')
            items.append((None, group, None, color))
            for signal, label, fmt in signals:
                path = signal if signal.startswith(TOP) else TOP + signal
                assert path in names, f'Missing signal: {path}'
                commands.append('variable_add ' + path)
                items.append((path, label, fmt, color))
        for index, (_, label, fmt, color) in enumerate(items):
            width = len(f'{len(items):x}')
            code = ''.join('abcdefghijklmnop'[int(c, 16)] for c in f'{index:0{width}x}')
            commands += ['item_focus ' + code, 'item_rename ' + label, 'item_set_color ' + color]
            if fmt and fmt != 'bit':
                display_format = 'unsigned' if name == 'overview' and fmt == 'TilePhase' else fmt
                commands.append('item_set_format ' + ({'hex': 'Hexadecimal', 'unsigned': 'Unsigned'}.get(display_format, display_format)))
        commands += ['item_unfocus', 'variable_force_name_type Local',
                     'preference_set_clock_highlight None', f'zoom_to {start} {end}']
        for time, label in markers:
            commands.append(f'marker_set_at {time} {label}')
        # Side panel and toolbar are hidden through the viewer UI before capture.
        (args.output / (name + '.sucl')).write_text('\n'.join(commands) + '\n')
        return [dict(signal=p, label=l,
                     format='unsigned' if name == 'overview' and f == 'TilePhase' else f)
                for p, l, f, _ in items if p]

    overview = layout([
        ('A | Host', 'Blue', HOST),
        ('B | DMA / KV', 'Green', CACHE),
        ('C | Compute', 'Yellow', COMPUTE),
        ('D | Writeback', 'Pink', OUTPUT)],
        [(events['command_accept_ps'], 'ACCEPT'), (events['done_ps'], 'DONE')],
        events['trace_start_ps'] - 50000, events['trace_end_ps'] + 250000, 'overview')
    detail = layout([
        ('AXI completion', 'Pink', DETAIL)],
        [(events['final_b_accept_ps'], 'FINAL_B'),
         (events['done_ps'], 'DONE')],
        events['aw_accept_ps'][0] - 55000, events['done_ps'] + 85000, 'completion')
    compute = layout([
        ('Q0 compute', 'Yellow',
         [('clk', 'clk', 'bit')] + COMPUTE +
         [('obs_array_start', 'array_start', 'bit'),
          ('obs_array_compute', 'array_compute', 'bit'),
          ('obs_dq_retire', 'dq_retire', 'bit'),
          ('obs_dq_group', 'dq_batch', 'unsigned'),
          ('obs_dq_publish', 'dq_publish', 'bit'),
          ('obs_dq_consume', 'dq_consume', 'bit'),
          ('obs_core_done', 'core_done', 'bit')])],
        [(events['core_start'][0]['time_ps'], 'Q0 start'),
         (events['core_done'][0]['time_ps'], 'Q0 done')],
        events['core_start'][0]['time_ps'] - 35000,
        events['core_done'][0]['time_ps'] + 70000, 'compute')
    (args.output / 'signals.json').write_text(json.dumps(dict(
        overview=overview, completion=detail, compute=compute), indent=2) + '\n')


if __name__ == '__main__':
    main()
