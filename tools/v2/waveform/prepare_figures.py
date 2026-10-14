#!/usr/bin/env python3
"""Crop genuine Surfer screenshots and add separate caption bands.

Never draw or alter waveform pixels. Crop bounds below match the captured
2000-pixel-wide macOS viewer window; inspect new captures before reuse.
Requires Pillow. Raw screenshots remain available beside the simulation trace.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    regular = '/System/Library/Fonts/Supplemental/Arial.ttf'
    bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    title_font = ImageFont.truetype(bold, 36)
    text_font = ImageFont.truetype(regular, 24)
    small_font = ImageFont.truetype(regular, 22)
    entries = [
        ('overview_raw.png', 'decode_waveform.png', 1575,
         'FlashAttention V2 — Full DECODE RTL Waveform',
         'L17 · d16 · 4Q : 2KV · 16 dequantizers · AXI latency 0 · accepted command to DONE: 2,072 cycles',
         ['ACCEPT = 86,755 ns   |   DONE = 107,475 ns   |   KV0 / KV1 stay at 17; external reads are Q-only.',
          'tile_state: 3 = QK, 4 = dequant wait / softmax, 6 = PV, 7 = update, 9 = normalize. IRQ_pending: 4 = reject, 1 = success.']),
        ('completion_raw.png', 'decode_waveform_detail.png', 950,
         'FlashAttention V2 — AXI Writeback & DONE',
         'Same L17 trace · 3 bursts · 32 accepted W beats · 3 OKAY responses · 256 output bytes',
         ['Final B accepted: 107,455 ns   |   DONE / success IRQ: 107,475 ns   |   two rising edges later.',
          'source_seen records retired packing; wb_busy clears after final B. BRESP = 0 is OKAY.']),
        ('compute_raw.png', 'decode_waveform_compute.png', 815,
         'FlashAttention V2 — Shared Compute / Query Head 0',
         'Same L17 trace · Q0 visits resident KV tiles at base tokens 0 and 16 · one array runs QK and PV',
         ['Q0 core start: 86,835 ns   |   Q0 core_done accepted: 91,135 ns. Clock period: 10 ns (simulation timebase).',
          'SM = S_UPDATE_SOFTMAX, including dequant wait. dq_publish / dq_consume follow all 16 batches; normalization follows tile 16.']),
    ]
    for raw_name, output_name, bottom, title, subtitle, notes in entries:
        raw = Image.open(args.raw_directory / raw_name).convert('RGB')
        assert raw.width == 2000 and raw.height >= bottom
        body = raw.crop((0, 57, raw.width, bottom))
        top_band, bottom_band = 106, 84
        final = Image.new('RGB', (body.width, top_band + body.height + bottom_band), '#0b151d')
        final.paste(body, (0, top_band))
        draw = ImageDraw.Draw(final)
        draw.text((22, 12), title, font=title_font, fill='#e5edf5')
        draw.text((22, 61), subtitle, font=text_font, fill='#b4c5d5')
        for index, note in enumerate(notes):
            assert draw.textbbox((0, 0), note, font=small_font)[2] < body.width - 44
            draw.text((22, top_band + body.height + 12 + 32*index), note,
                      font=small_font, fill='#b4c5d5')
        # All pixels in the waveform area must remain exactly as captured.
        assert final.crop((0, top_band, body.width, top_band + body.height)).tobytes() == body.tobytes()
        dest = args.output / output_name
        assert not dest.exists(), f'Refusing to overwrite {dest}'
        final.save(dest, optimize=True)
        print(f'{dest}: {final.width}x{final.height}, {dest.stat().st_size} bytes')


if __name__ == '__main__':
    main()
