#!/usr/bin/env python3
"""Render the README stateful dataflow derivative using only the Python standard library.

Run: python3 -B tools/v2/generate_stateful_dataflow.py
Use --output PATH to render a review copy without touching the public asset.
Token slots are conceptual logical positions, not physical SRAM banks.
No historical figure, provenance, RTL, fixture or measurement is regenerated.
"""
import argparse
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAVY, TEAL, MUTED, AMBER = '#152b4b', '#087e87', '#52647a', '#a66b16'
# Major functional routes, cross-checked against the current SoC production boundary.
RTL_ROUTES = {
    'host-command': ('rtl/v2/soc/flash_attn_soc.sv', 'child_start(child_start)'),
    'read-dma': ('rtl/v2/soc/flash_attn_serial.sv', 'dma_engine_vec u_dma'),
    'q-scratchpad': ('rtl/v2/compute/flash_attn_core_decode.sv', 'u_q_sp'),
    'kv-pairs': ('rtl/v2/cache/kv_cache_banked_decode.sv', 'assign pair_complete'),
    'kv-publication': ('rtl/v2/soc/flash_attn_serial.sv', 'cmd_commit(completion)'),
    'shared-array': ('rtl/v2/compute/flash_attn_core_decode.sv', 'array_controller #(.SIZE(SIZE)) u_array_ctrl'),
    'array-instance': ('rtl/systolic/array_controller.sv', 'systolic_array #(.SIZE(SIZE)) u_array'),
    'results-packer': ('rtl/v2/soc/flash_attn_soc.sv', 'gqa_output_packer_m6b3'),
    'writeback': ('rtl/v2/interface/dma_write_engine.sv', 'if(accept_b)'),
    'host-completion': ('rtl/v2/soc/soc_job_adapter.sv', 'source_seen && response_seen && !wb_outstanding'),
    'success-irq': ('rtl/v2/soc/axi_lite_csr.sv', 'irq_pending & irq_enable'),
    'host-ack': ('rtl/v2/soc/soc_job_adapter.sv', 'phase<=RELEASE;child_ack<=1'),
}


def render():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1230" viewBox="0 0 1200 1230" role="img" aria-labelledby="title desc">
<title id="title">V2 stateful execution: separate prefill, append and decode commands</title>
<desc id="desc">PREFILL and DECODE are two views of the same current SoC hardware; one host job is active at a time. PREFILL reads paired K/V for 16 tokens in both physical KV groups, then loads Q per serial head. Completed K/V pairs can supply prefill compute before valid sequence metadata is published at child completion. Prefill token sketches show the resulting valid logical positions after publication. DECODE loads only Q and reads committed resident K/V without modifying the cache. Its token sketch shows an L16 example; supported host decode lengths are 16, 17 and 18, at absolute query position L minus 1. Q0/Q1 share KV0 and Q2/Q3 share KV1. Four query heads are scheduled serially on one reused 16 by 16 systolic array for QK and PV; online softmax and normalization are summarized within the compute engine. Results remain owned in the child result buffer, are packed and sent through AXI output write DMA. Successful host DONE and enabled success IRQ wait for source retirement, the final accepted OKAY B response and no outstanding writes. ACK releases completed job ownership; it does not start computation or clear IRQ. A separate APPEND inset shows one new K/V token per group, with position L provisional during stores and valid only after complete pairs in both groups permit atomic L plus 1 publication. APPEND performs no Q load, compute or output writeback; it reaches DONE and enabled success IRQ after child completion and also requires host ACK. Appending to the L16 example produces L17. Logical token positions do not depict SRAM banks or storage geometry. HEAD_DIM is 16 or 64; cache capacity is 64. CLEAR invalidates metadata, not SRAM bits. Fatal AXI errors require common DUT and endpoint reset; external writes already completed are not rolled back. Arrows summarize routing and job lifetime, not simultaneous execution or port-level wiring.</desc>
<!-- README presentation derivative; historical accepted figures remain unchanged.
     Source references: docs/ARCHITECTURE.md, docs/v2/VERIFICATION.md,
     rtl/v2/soc/{flash_attn_soc,flash_attn_serial,soc_job_adapter,
     axi_lite_csr,gqa_output_packer}.sv,
     rtl/v2/cache/kv_cache_banked_decode.sv,
     rtl/v2/compute/flash_attn_core_decode.sv, rtl/systolic/array_controller.sv,
     rtl/v2/interface/dma_write_engine.sv.
     Reproduce with: python3 -B tools/v2/generate_stateful_dataflow.py -->
<defs>
<marker id="data-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="#087e87"/></marker>
<marker id="control-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="#52647a"/></marker>
</defs>
<style>
text { font-family: Arial, Helvetica, sans-serif; fill: #152b4b; }
.data { fill: none; stroke: #087e87; stroke-width: 3; stroke-linejoin: round; marker-end: url(#data-arrow); }
.control { fill: none; stroke: #52647a; stroke-width: 3; stroke-dasharray: 6 5; stroke-linejoin: round; marker-end: url(#control-arrow); }
</style>
<rect width="1200" height="1230" fill="#ffffff"/>''']

    def rect(x, y, w, h, fill='#f5f8fb', stroke='#cbd7e2', radius=11, extra=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.4" {extra}/>')

    def text(x, y, value, size=22, weight='normal', color=NAVY, center=False):
        anchor = ' text-anchor="middle"' if center else ''
        parts.append(f'<text x="{x}" y="{y}"{anchor} font-size="{size}" font-weight="{weight}" style="fill:{color}">{escape(value)}</text>')

    def arrow(path, kind='data', route=''):
        parts.append(f'<path class="{kind}" d="{path}" data-route="{route}"/>')

    def box(x, y, w, h, title, lines=(), kind='data', size=23):
        fill, stroke = {'data': ('#edf6fa', '#9dbccb'), 'control': ('#f1f4f8', '#b8c5d3'), 'state': ('#fff4e2', '#d9ad65')}[kind]
        rect(x, y, w, h, fill, stroke)
        text(x+w/2, y+((28 if h<80 else 35) if len(lines)<2 else 31), title, size, 'bold', center=True)
        for i, line in enumerate(lines):
            text(x+w/2, y+((52 if h<80 else 64) if len(lines)<2 else 58)+i*25,
                 line, 20 if len(lines)<2 else 19, color=MUTED, center=True)

    def slots(x, y, label_x, pending=False, step=13, width=11):
        # Sixteen logical positions; each visual slot represents a complete K/V token.
        for group in range(2):
            yy = y+29*group
            text(label_x, yy+14, f'KV{group}', 17, color=MUTED)
            for token in range(16):
                rect(x+step*token, yy, width, 17, '#e8be76', '#d6a558', 2,
                     f'data-group="{group}" data-token="{token}" data-valid="true"')
            if pending:
                rect(x+16*step+4, yy, 13, 17, '#ffffff', AMBER, 2,
                     f'stroke-dasharray="3 2" data-group="{group}" data-token="16" data-valid="false"')
                text(x+16*step+24, yy+14, 'L', 17, color=AMBER)

    text(40, 52, 'V2 stateful execution', 38, 'bold')
    text(40, 85, 'One active host job at a time · PREFILL / APPEND / DECODE are separate commands', 22, color=MUTED)
    parts.append('<circle cx="963" cy="46" r="6" fill="#d9ad65"/>')
    text(978, 52, 'Persistent state', 20, color=AMBER)

    for x, title, subtitle in [(40, 'Prefill', '16 query rows / head · establish L = 16'), (610, 'Decode', '1 query / head · query position L − 1')]:
        rect(x, 112, 550, 494, '#fbfcfe', '#dce5ed', 15)
        text(x+24, 153, title, 30, 'bold')
        text(x+24, 183, subtitle, 21, color=MUTED)
        box(x+20, 211, 510, 65, 'Host → AXI4-Lite CSR', ('Configure + start '+title.upper(),), 'control')
        arrow(f'M{x+275} 276 V322', 'control', 'host-command')
        text(x+289, 306, 'captured job', 19, color=MUTED)
        box(x+20, 322, 510, 70, 'External memory → AXI read DMA',
            ('Load K/V, then Q per head' if title=='Prefill' else 'Load Q only; no K/V append',))
        if title == 'Prefill':
            arrow('M315 392 V412 H153 V436', route='q-scratchpad')
            arrow('M315 392 V412 H424 V436', route='kv-pairs')
            text(446, 429, 'K / V', 19, 'bold', TEAL)
        else:
            # DECODE DMA has only a Q route: no link to persistent K/V stores.
            arrow('M885 392 V412 H723 V436', route='q-scratchpad')
        text(x+84, 429, 'Q', 19, 'bold', TEAL)
        box(x+20, 436, 186, 150, 'Query Q', ('Scratchpad',))
        rect(x+238, 436, 292, 150, '#fff4e2', '#d9ad65')
        text(x+384, 467, 'Persistent KV cache', 23, 'bold', center=True)
        text(x+384, 493, '16 valid tokens after publication' if title=='Prefill' else 'Committed tokens, reused',
             18 if title=='Prefill' else 20, color=MUTED, center=True)
        slots(x+295, 508, x+252)
        text(x+384, 576, 'Positions 0 … 15' if title=='Prefill' else 'L = 16 example · no cache writes', 18, color=MUTED, center=True)

    arrow('M153 586 V630 H280 V652', route='shared-array')
    arrow('M424 586 V652', route='shared-array')
    arrow('M723 586 V652', route='shared-array')
    arrow('M994 586 V630 H920 V652', route='shared-array')
    rect(220, 652, 760, 112, '#e8f6f3', '#83bfb7')
    text(600, 686, 'Shared compute engine · 4 serial Q heads', 28, 'bold', center=True)
    text(600, 717, 'One reused 16 × 16 array (QKᵀ + PV) · softmax + normalize', 22, color=MUTED, center=True)
    text(600, 747, 'Q0/Q1 → KV0 · Q2/Q3 → KV1', 21, color=MUTED, center=True)
    arrow('M300 764 V784 H190 V804', route='results-packer')
    text(318, 791, 'normalized results', 20, color=TEAL)

    box(40, 804, 300, 102, 'Result buffer → packer', ('AXI writeback → memory',), size=22)
    arrow('M340 855 H375', route='writeback')
    box(375, 804, 365, 102, 'Completion gate', ('Source retired + final OKAY B', 'No writes outstanding'), 'control')
    arrow('M740 855 H775', 'control', 'host-completion')
    box(775, 804, 190, 102, 'DONE / IRQ', ('IRQ if enabled',), 'control', size=23)
    arrow('M965 855 H1000', 'control', 'host-ack')
    box(1000, 804, 160, 102, 'Host ACK', ('Release owner',), 'control')

    # APPEND is an independent command, with no connection to the DECODE DMA or compute.
    rect(40, 932, 1120, 176, '#fbfcfe', '#dce5ed', 15)
    text(64, 966, 'APPEND — separate host command', 24, 'bold')
    text(745, 966, 'No Q load, compute or writeback', 20, color=MUTED)
    box(60, 986, 170, 85, 'Host / CSR', ('Start APPEND',), 'control', size=22)
    arrow('M230 1028 H265', 'control', 'host-command')
    box(265, 986, 180, 85, 'K/V read DMA', ('One token/group',), size=22)
    arrow('M445 1028 H480', route='kv-pairs')
    rect(480, 981, 345, 103, '#fff4e2', '#d9ad65')
    text(652, 1008, 'Paired K/V stores', 22, 'bold', center=True)
    slots(539, 1020, 493, pending=True, step=12, width=10)
    arrow('M825 1028 H860', route='kv-publication')
    rect(860, 981, 280, 103, '#fff4e2', '#d9ad65')
    text(1000, 1010, 'Atomic publish L + 1', 22, 'bold', center=True)
    text(1000, 1036, 'Both groups complete', 19, color=MUTED, center=True)
    for x, group in [(887, 0), (1000, 1)]:
        rect(x, 1048, 26, 20, '#d6a558', AMBER, 3, f'data-group="{group}" data-token="16" data-valid="true"')
        text(x+13, 1063, 'L', 16, 'bold', center=True)
        text(x+33, 1063, f'KV{group}', 17, color=MUTED)
    text(64, 1095, 'New slot L stays provisional until publication (L = 16 → 17 shown).', 18, color=MUTED)
    text(863, 1095, 'Then DONE / IRQ → ACK', 19, color=MUTED)

    rect(40, 1132, 1120, 44, '#fff4e2', '#e8c998')
    text(600, 1161, 'Two physical KV groups retain state across healthy jobs + ACK', 23, 'bold', AMBER, True)
    text(40, 1207, 'Logical token positions, not SRAM banks. IRQ clear is separate from ACK; CLEAR invalidates metadata.', 20, color=MUTED)
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/v2/figures/stateful_dataflow.svg')
    args = parser.parse_args()
    args.output.write_text(render(), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
