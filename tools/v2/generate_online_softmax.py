#!/usr/bin/env python3
"""Render the README online-softmax diagram with the Python standard library.

Conceptual tile flow; the ideal recurrence is distinguished from fixed-point
RTL. Use --output for a temporary preview. Original PNGs and accepted figures
are not inputs to this renderer and are never overwritten.
"""
import argparse
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAVY, TEAL, MUTED = '#152b4b', '#087e87', '#52647a'
RTL_ROUTES = {
    'shared-array': ('rtl/v2/compute/flash_attn_core_decode.sv', 'u_array_ctrl'),
    'score-mask': ('rtl/v2/compute/online_softmax_masked.sv', 'if (mask_r[idx[AIDX_W-1:0]])'),
    'weights': ('rtl/v2/compute/online_softmax_masked.sv', "16'h00FF"),
    'values': ('rtl/v2/compute/flash_attn_core_decode.sv', 'scale_v_reg'),
    'rescale': ('rtl/v2/compute/online_softmax_masked.sv', 'rescale_q88 <= lut_out'),
    'running-sum': ('rtl/v2/compute/flash_attn_core_decode.sv', 'norm_divisor_sel = sfx_running_sum'),
    'fused-output': ('rtl/memory/output_buffer.sv', 'first_tile_d ? data_in_d : rescaled_val + data_in_d'),
    'normalize': ('rtl/memory/output_buffer.sv', 'assign norm_numer = {old_data, 8\'b0}'),
}


def render():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="965" viewBox="0 0 1200 965" role="img" aria-labelledby="title desc">
<title id="title">Online softmax and shared QK/PV attention datapath</title>
<desc id="desc">Conceptual tile flow, not cycle timing or separate compute arrays. QK and PV are scheduled phases of one shared 16 by 16 systolic array. Q and resident K form scores, which are dequantized and masked. Per query row, the running maximum reduces over eligible keys; masked keys contribute zero exponential weight. The ideal real-arithmetic recurrence updates maximum m, rescale factor alpha, unnormalized weights p and running sum ell. The first tile uses its eligible tile maximum and fresh sum/output state. PV reuses the array with V from the KV cache; the output buffer fuses prior-output rescaling and current-tile accumulation. Normalize only after the last KV tile. m and ell are scalars per query row; the output accumulator has D elements, O(D) running state excluding KV storage, scratchpads and tile staging. Prefill has 16 query rows per head and 16 valid cached keys; causal eligibility varies by query row. Supported L17 decode has one query per head, a first tile with 16 valid keys and a second 16-slot tile with one valid key and 15 masked tail slots. D is 16 or 64. The displayed recurrence is ideal mathematics: RTL uses LUT exponentials, fixed-point arithmetic, scale_v, rounding, saturation and wrap. Raw Q8.8 LUT weights contribute to the denominator, while PV weights are capped at 255. First-tile updates ignore prior output SRAM contents. Final RTL normalization is signed truncating division of output accumulator shifted left by 8 by the running denominator; it is not a floating-point exactness claim. Arrows summarize values and state factors, not simultaneous hardware activity.</desc>
<!-- Reproduce: python3 -B tools/v2/generate_online_softmax.py
     Engineering sources: docs/DESIGN.md, docs/ARCHITECTURE.md,
     docs/NUMERICAL_ACCURACY.md, golden/v2/stateful/decode_reference.py,
     rtl/v2/compute/{flash_attn_core_decode,online_softmax_masked}.sv,
     rtl/memory/output_buffer.sv. -->
<defs>
<marker id="data-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="#087e87"/></marker>
<marker id="state-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="#52647a"/></marker>
</defs>
<style>
text { font-family: Arial, Helvetica, sans-serif; fill: #152b4b; }
.data { fill: none; stroke: #087e87; stroke-width: 3; stroke-linejoin: round; marker-end: url(#data-arrow); }
.state { fill: none; stroke: #52647a; stroke-width: 2.5; stroke-dasharray: 6 5; stroke-linejoin: round; marker-end: url(#state-arrow); }
</style>
<rect width="1200" height="965" fill="#ffffff"/>''']

    def text(x, y, value, size=20, weight='normal', color=NAVY, center=False):
        anchor = ' text-anchor="middle"' if center else ''
        parts.append(f'<text x="{x}" y="{y}"{anchor} font-size="{size}" font-weight="{weight}" style="fill:{color}">{escape(value)}</text>')

    def rect(x, y, w, h, fill='#f5f8fb', stroke='#cbd7e2', radius=11, extra=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.4" {extra}/>')

    def arrow(path, route='', kind='data'):
        parts.append(f'<path class="{kind}" d="{path}" data-route="{route}"/>')

    text(40, 52, 'Online softmax · shared attention datapath', 36, 'bold')
    text(40, 86, 'Stream KV tiles · update per-query running state · normalize after the last tile', 22, color=MUTED)
    rect(40, 112, 1120, 54, '#edf6fa', '#9dbccb')
    text(600, 145, 'One shared 16 × 16 array (256 PEs) · QK and PV are scheduled phases', 23, 'bold', TEAL, True)

    rect(40, 200, 150, 185, '#eef7f2', '#98bfa9')
    text(115, 234, 'Input tile', 23, 'bold', center=True)
    text(115, 270, 'Q : Bq × D', 20, center=True)
    text(115, 299, 'K : 16 × D', 20, center=True)
    text(115, 328, 'V : 16 × D', 20, center=True)
    text(115, 363, 'K/V from cache', 17, color=MUTED, center=True)

    rect(230, 200, 170, 185, '#edf6fa', '#9dbccb')
    text(315, 235, '1. QKᵀ', 26, 'bold', center=True)
    text(315, 266, 'array phase 1', 18, color=TEAL, center=True)
    text(315, 306, 'signed INT8', 21, center=True)
    text(315, 335, '→ INT32 scores', 20, center=True)
    text(315, 365, 'Bq × 16', 20, color=MUTED, center=True)

    rect(440, 200, 250, 185, '#fff4e2', '#d9ad65')
    text(565, 235, '2. Scale + mask', 23, 'bold', center=True)
    text(565, 276, 'Q/K scales → dequantize', 19, center=True)
    text(565, 306, 'Saturation → 1/√D shift', 19, center=True)
    text(565, 344, 'Key validity + causal mask', 18, color=MUTED, center=True)
    text(565, 369, 's = eligible scaled scores', 18, center=True)

    rect(730, 200, 430, 185, '#fff0f2', '#d5a9b4')
    text(945, 232, '3. Online update · per query row', 23, 'bold', center=True)
    text(750, 264, 'm′ = max(m, max{sⱼ : j eligible})', 21)
    text(750, 291, 'α = exp(m − m′)', 21)
    text(750, 318, 'pⱼ = exp(sⱼ − m′); masked pⱼ = 0', 20)
    text(750, 347, 'ℓ′ = αℓ + Σⱼ pⱼ', 23)
    text(750, 373, 'First tile: m′ = tile max; prior ℓ = 0', 17, color=MUTED)

    arrow('M190 289 H230', 'shared-array')
    arrow('M400 289 H440', 'score-mask')
    arrow('M690 289 H730')

    rect(935, 430, 225, 150, '#edf6fa', '#9dbccb')
    text(1048, 463, '4. PV', 26, 'bold', center=True)
    text(1048, 491, 'same array · phase 2', 19, color=TEAL, center=True)
    text(1048, 522, 'unsigned P / signed V', 18, center=True)
    text(1048, 553, 'apply scale_v', 20, center=True)

    rect(580, 430, 315, 150, '#eef7f2', '#98bfa9')
    text(737, 464, '5. Fused output update', 23, 'bold', center=True)
    text(737, 505, 'o′ = αo + P V', 29, center=True)
    text(737, 539, 'Rescale + accumulate', 20, center=True)
    text(737, 565, 'One buffer traversal', 18, color=MUTED, center=True)

    rect(325, 430, 215, 150, '#f1effc', '#bbb1d9')
    text(433, 464, '6. Normalize', 25, 'bold', center=True)
    text(433, 507, 'O = o / ℓ', 29, center=True)
    text(433, 544, 'after last KV tile', 20, color=MUTED, center=True)

    rect(40, 430, 245, 150, '#edf6fa', '#9dbccb')
    text(162, 466, 'Output tile', 25, 'bold', center=True)
    text(162, 505, 'Bq × D · result buffer', 20, center=True)
    text(162, 545, 'Then SoC writeback', 20, color=MUTED, center=True)

    arrow('M1048 385 V430', 'weights')
    text(1063, 415, 'P', 21, 'bold', TEAL)
    arrow('M790 385 V416 H737 V430', 'rescale', 'state')
    text(800, 414, 'α', 20, color=MUTED)
    arrow('M750 385 V402 H433 V430', 'running-sum', 'state')
    text(452, 423, 'ℓ', 20, color=MUTED)
    arrow('M935 505 H895', 'fused-output')
    arrow('M580 505 H540', 'normalize')
    arrow('M325 505 H285')
    arrow('M40 328 H20 V610 H1048 V580', 'values')
    text(40, 638, 'V from KV cache', 19, 'bold', TEAL)
    text(410, 638, 'Repeat steps 1–5 per KV tile; step 6 runs only after the final tile.', 19, color=MUTED)

    rect(40, 665, 540, 160, '#fbfcfe', '#dce5ed')
    rect(620, 665, 540, 160, '#fbfcfe', '#dce5ed')
    text(64, 699, 'Prefill · L16', 24, 'bold')
    text(64, 726, 'Bq = 16 · one 16-slot KV tile', 19, color=MUTED)
    text(644, 699, 'Decode · L17', 24, 'bold')
    text(1040, 699, 'Bq = 1', 19, color=MUTED)
    for token in range(16):
        rect(64+27*token, 741, 22, 23, '#b5d9d4', '#6ca49e', 3,
             f'data-example="prefill" data-token="{token}" data-valid="true"')
    text(64, 797, 'All tokens valid; causal eligibility varies by row.', 18, color=MUTED)
    for tile, y in ((0, 728), (1, 771)):
        text(644, y+17, f'Tile {tile}', 19, color=MUTED)
        for slot in range(16):
            valid = tile == 0 or slot == 0
            rect(726+24*slot, y, 20, 21, '#b5d9d4' if valid else '#f2f4f7',
                 '#6ca49e' if valid else '#bdc6d1', 3,
                 f'data-example="decode" data-token="{tile*16+slot}" data-valid="{str(valid).lower()}"' +
                 ('' if valid else ' stroke-dasharray="3 2"'))
    text(644, 815, '16 valid + 1 valid; 15 tail slots masked', 18, color=MUTED)

    text(40, 857, 'Per query row: m scalar · ℓ scalar · o has D elements → O(D) running state', 22, 'bold')
    text(40, 884, 'D = 16 or 64. Running-state size excludes KV storage, scratchpads and tile staging.', 18, color=MUTED)
    text(40, 910, 'Ideal formulas above. RTL: LUT / fixed-point arithmetic; raw weights feed ℓ, PV weights cap at 255.', 18, color=MUTED)
    text(40, 936, 'First tile ignores old output state. RTL final output: (o_int << 8) / ℓ_int, with signed truncation.', 18, color=MUTED)
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/v2/figures/online_softmax.svg')
    args = parser.parse_args()
    for path, anchor in RTL_ROUTES.values():
        if anchor not in (ROOT/path).read_text():
            parser.error('RTL reference changed: '+path+' / '+anchor)
    args.output.write_text(render(), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
