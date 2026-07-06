// ============================================================
//  tb_banked_tile_loader.cpp — unit test for banked_tile_loader (3B-1).
//
//  Covers (per build of HEAD_DIM):
//    - Q  tile load,  row-major layout check
//    - KV tile load,  K and V read concurrently, both layouts checked
//    - a nonzero tile base address (tile not at scratchpad address 0)
//  Run once with HEAD_DIM=16 and once with HEAD_DIM=64 (two builds).
// ============================================================
#include "Vtb_banked_tile_loader.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

static Vtb_banked_tile_loader* dut = nullptr;
static const int TILE = 16;

static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

// byte b of a 128-bit Verilator vector (WData[4], little-endian words)
template <typename T>
static uint8_t vbyte(const T& vec, int b) { return (vec[b / 4] >> ((b % 4) * 8)) & 0xFF; }

static uint8_t qpat(int f) { return (uint8_t)((f * 7  + 1) & 0xFF); }
static uint8_t kpat(int f) { return (uint8_t)((f * 13 + 2) & 0xFF); }
static uint8_t vpat(int f) { return (uint8_t)((f * 5  + 9) & 0xFF); }

static void reset() {
    dut->rst_n = 0; dut->start = 0; dut->f_we = 0; dut->f_sel = 0;
    dut->f_addr = 0; dut->f_data = 0; dut->mode = 0; dut->base_addr = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

// fill scratchpad `sel` (0=Q,1=K,2=V) at [base, base+TILE*D) with pat(flat)
static void fill(int sel, int base, int D, uint8_t (*pat)(int)) {
    for (int f = 0; f < TILE * D; f++) {
        dut->f_we = 1; dut->f_sel = sel; dut->f_addr = base + f; dut->f_data = pat(f);
        tick();
    }
    dut->f_we = 0; tick(); tick();
}

// run the loader; capture the tile-register write stream into out[3][TILE*D]
static void run_load(int mode, int base, int D, std::vector<uint8_t> out[3]) {
    for (int s = 0; s < 3; s++) out[s].assign(TILE * D, 0xEE);
    dut->mode = mode; dut->base_addr = base;
    dut->start = 1; tick(); dut->start = 0;
    for (int i = 0; i < 100000; i++) {
        tick();
        if (dut->wr_en) {
            int idx = dut->wr_index;
            for (int b = 0; b < 16; b++) {
                if (mode == 0) {
                    out[0][idx + b] = vbyte(dut->q_stripe, b);
                } else {
                    out[1][idx + b] = vbyte(dut->k_stripe, b);
                    out[2][idx + b] = vbyte(dut->v_stripe, b);
                }
            }
        }
        if (dut->done) break;
    }
}

// row-major layout check: reg[r*D + c] == pat(r*D + c)
static int check(const std::vector<uint8_t>& reg, int D, uint8_t (*pat)(int), const char* tag) {
    int err = 0;
    for (int r = 0; r < TILE; r++)
        for (int c = 0; c < D; c++) {
            int f = r * D + c;
            if (reg[f] != pat(f)) {
                if (err < 5) printf("    %s FAIL reg[r=%d,c=%d] (flat %d): got %02x exp %02x\n",
                                    tag, r, c, f, reg[f], pat(f));
                err++;
            }
        }
    return err;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    int D = 16;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);

    dut = new Vtb_banked_tile_loader;
    reset();

    const int base = 16 * D;          // nonzero tile base (e.g. tile_row/tile_col = 16)
    int errors = 0;
    printf("=== banked_tile_loader: HEAD_DIM=%d  TILE_SIZE=%d  NUM_STRIPES=%d  base=%d ===\n",
           D, TILE, TILE * D / 16, base);

    // ── Q tile load ──────────────────────────────────────────────────
    fill(0, base, D, qpat);
    std::vector<uint8_t> q_out[3];
    run_load(/*mode=Q*/0, base, D, q_out);
    int q_err = check(q_out[0], D, qpat, "Q");
    printf("  Q  load  (d=%d, base=%d) row-major : %s\n", D, base, q_err ? "FAIL" : "PASS");
    errors += q_err;

    // ── KV tile load (K and V concurrent) ────────────────────────────
    fill(1, base, D, kpat);
    fill(2, base, D, vpat);
    std::vector<uint8_t> kv_out[3];
    run_load(/*mode=KV*/1, base, D, kv_out);
    int k_err = check(kv_out[1], D, kpat, "K");
    int v_err = check(kv_out[2], D, vpat, "V");
    printf("  KV load  (d=%d, base=%d) K row-major: %s\n", D, base, k_err ? "FAIL" : "PASS");
    printf("  KV load  (d=%d, base=%d) V row-major: %s\n", D, base, v_err ? "FAIL" : "PASS");
    errors += k_err + v_err;

    bool pass = (errors == 0);
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : banked_tile_loader\n");
    printf("Scenarios covered: q_load_d%d, kv_load_d%d_concurrent, nonzero_base, row_major\n", D, D);
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", errors);
    printf("Result           : %s\n", pass ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    dut->final(); delete dut;
    return pass ? 0 : 1;
}
