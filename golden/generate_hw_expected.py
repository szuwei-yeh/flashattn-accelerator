"""
generate_hw_expected.py — Generate hardware-accurate expected.hex for any N.

Simulates the EXACT hardware computation pipeline:
  1. QK^T matmul (INT8×INT8→INT32)
  2. Dequantize: (QK * sq_q88 * sk_q88 + 128) >> 8  → Q8.8
  3. Scale by 1/sqrt(d): >> 2  (for HEAD_DIM=16, 1/sqrt(16)=0.25=2^-2)
  4. Per-tile independent softmax using the exp LUT (tile_start=tile_last=1 always)
  5. Quantize P to INT8 (lower byte of Q8.8 normalized output, signed)
  6. P_int8 @ V_int8 → INT32 (PV matmul)
  7. Scale by sv_q88: (PV * sv_q88) >> 8
  8. Accumulate across all KV tiles for each Q tile

Output: expected.hex (INT32, 8 hex chars per line, in Q8.8 units × 256)

Usage:
    python golden/generate_hw_expected.py --data data/N64
    python golden/generate_hw_expected.py --data data/N128
    python golden/generate_hw_expected.py --data data/N256
"""

import numpy as np
import argparse
import os


def load_int8_hex(path, n):
    vals = []
    with open(path) as f:
        for line in f:
            v = int(line.strip(), 16)
            if v >= 0x80:
                v -= 0x100
            vals.append(v)
    return np.array(vals[:n], dtype=np.int8)


def load_scales(path):
    scales = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=')
                scales[k.strip()] = v.strip()
    sq_q88 = int(scales['scale_q_q88'], 16)
    sk_q88 = int(scales['scale_k_q88'], 16)
    sv_q88 = int(scales['scale_v_q88'], 16)
    N = int(scales['N'])
    d = int(scales['d'])
    return sq_q88, sk_q88, sv_q88, N, d


def load_exp_lut(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f]


def to_addr(score_q88, maxv_q88):
    """Exact replication of hardware to_addr() function."""
    diff = int(score_q88) - int(maxv_q88)
    numer = diff * 255 + 523264
    result = numer >> 11
    if result > 255:
        return 255
    if result < 0:
        return 0
    return result


def hw_softmax_tile(scores_q88, lut):
    """
    Simulate hardware online_softmax for one tile with tile_start=tile_last=True.
    Returns: list of DIM signed INT8 values (lower byte of Q8.8 normalized output).
    """
    DIM = len(scores_q88)

    # S_FIND_MAX: find max of tile scores
    tile_max = scores_q88[0]
    for s in scores_q88[1:]:
        if s > tile_max:
            tile_max = s

    # S_RESCALE: is_first_r=True → running_max = tile_max, running_sum = 0
    running_max = tile_max

    # S_ACCUM: compute exp values using LUT (1-cycle latency accounted for)
    # Hardware: in RESCALE_WAIT lut_addr=to_addr(sc[0],max), in ACCUM lut_addr=to_addr(sc[1],max), ...
    # exp_vals[i] = lut[to_addr(scores[i], running_max)]
    exp_vals = [lut[to_addr(s, running_max)] for s in scores_q88]

    running_sum = sum(exp_vals)

    # S_NORM: normalize
    norm_denom = max(running_sum, 1)
    result = []
    for ev in exp_vals:
        norm_val = ((ev << 8) + (norm_denom >> 1)) // norm_denom
        if norm_val > 0xFFFF:
            norm_val = 0xFFFF
        low_byte = norm_val & 0xFF
        if low_byte >= 128:
            low_byte -= 256   # interpret as signed INT8
        result.append(low_byte)
    return result


def generate(data_dir, exp_lut_path, tile_size=16):
    sq_q88, sk_q88, sv_q88, N, d = load_scales(
        os.path.join(data_dir, 'scales.txt'))
    lut = load_exp_lut(exp_lut_path)

    Q = load_int8_hex(os.path.join(data_dir, 'q_input.hex'), N * d).reshape(N, d)
    K = load_int8_hex(os.path.join(data_dir, 'k_input.hex'), N * d).reshape(N, d)
    V = load_int8_hex(os.path.join(data_dir, 'v_input.hex'), N * d).reshape(N, d)

    num_q_tiles  = N // tile_size
    num_kv_tiles = N // tile_size

    print(f"N={N} d={d} tile_size={tile_size}")
    print(f"sq_q88=0x{sq_q88:04X} sk_q88=0x{sk_q88:04X} sv_q88=0x{sv_q88:04X}")
    print(f"Q tiles={num_q_tiles}  KV tiles={num_kv_tiles}  total pairs={num_q_tiles*num_kv_tiles}")

    output = np.zeros((N, d), dtype=np.int64)

    for i in range(num_q_tiles):
        qs = i * tile_size
        qe = qs + tile_size
        Q_tile = Q[qs:qe]                          # [TILE_SIZE, d]

        for j in range(num_kv_tiles):
            kvs = j * tile_size
            kve = kvs + tile_size
            K_tile = K[kvs:kve]                    # [TILE_SIZE, d]
            V_tile = V[kvs:kve]                    # [TILE_SIZE, d]

            # 1. QK^T matmul (Q tile @ K tile transposed)
            QKT = Q_tile.astype(np.int32) @ K_tile.astype(np.int32).T   # [TS, TS]

            # 2. Dequantize with SHIFT=8: (QKT * sq * sk + 0.5) >> 8
            score_q88 = ((QKT.astype(np.int64) * sq_q88 * sk_q88) + 128) >> 8

            # 3. Scale by 1/sqrt(d) = >>2  (d=16 → 1/4)
            score_scaled = score_q88 >> 2

            # 4. Per-tile independent softmax (tile_start=tile_last=True always)
            P_int8 = np.zeros((tile_size, tile_size), dtype=np.int8)
            for r in range(tile_size):
                row = [int(score_scaled[r, c]) for c in range(tile_size)]
                p_row = hw_softmax_tile(row, lut)
                P_int8[r] = np.array(p_row, dtype=np.int8)

            # 5. PV matmul (INT8 × INT8 → INT32)
            PV = P_int8.astype(np.int32) @ V_tile.astype(np.int32)     # [TS, d]

            # 6. Scale by sv_q88: (PV * sv_q88) >> 8
            PV_scaled = ((PV.astype(np.int64) * sv_q88) >> 8).astype(np.int32)

            # 7. Accumulate into output tile
            output[qs:qe] += PV_scaled

        if (i + 1) % max(1, num_q_tiles // 4) == 0:
            print(f"  Q tile {i+1}/{num_q_tiles} done")

    # Write expected.hex
    out_path = os.path.join(data_dir, 'expected.hex')
    with open(out_path, 'w') as f:
        for val in output.flatten():
            f.write(f'{int(val) & 0xFFFFFFFF:08X}\n')
    print(f"Written: {out_path}  ({N*d} entries)")
    print(f"output[0,:8] = {output[0, :8].tolist()}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      type=str, required=True,
                        help='Directory containing q/k/v_input.hex and scales.txt')
    parser.add_argument('--tile_size', type=int, default=16)
    parser.add_argument('--exp_lut',   type=str, default='data/exp_lut.hex',
                        help='Path to exp_lut.hex (relative to project root)')
    args = parser.parse_args()

    generate(args.data, args.exp_lut, args.tile_size)
