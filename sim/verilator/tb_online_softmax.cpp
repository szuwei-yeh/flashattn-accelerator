// tb_online_softmax.cpp — Verilator testbench for online_softmax.sv
//
// Test 1 — Single tile (DIM=16):
//   Random Q8.8 scores, compare hardware output vs reference softmax.
//   Acceptance: max relative error < 1%.
//
// Test 2 — Dual tile:
//   32 random Q8.8 scores split into two tiles of 16.
//   Hardware output (last tile) compared against global softmax over all 32.
//   Acceptance: max relative error < 1%.
//
// Test 3 — Multi-sequence WITHOUT rst_n (running_max reset check):
//   Run a second sequence via tile_start=1 WITHOUT calling reset().
//   New scores are in a much smaller range than Test 2 to maximise
//   the chance that a stale running_max corrupts the result.
//   A correct RTL resets running_max on tile_start; a buggy one FAILS here.
//
// Build & run (from sim/verilator/):
//   verilator --cc --exe --build -Wall \
//     ../../rtl/softmax/exp_lut.sv \
//     ../../rtl/softmax/online_softmax.sv \
//     tb_online_softmax.cpp \
//     -o sim_softmax --Mdir obj_softmax
//   ./obj_softmax/sim_softmax

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "Vonline_softmax.h"
#include "verilated.h"

static const int DIM = 16;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void tick(Vonline_softmax *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

static void reset(Vonline_softmax *dut) {
    dut->rst_n      = 0;
    dut->tile_start = 0;
    dut->tile_valid = 0;
    dut->tile_last  = 0;
    memset(&dut->scores_flat, 0, sizeof(dut->scores_flat));
    tick(dut); tick(dut);
    dut->rst_n = 1;
    tick(dut);
}

// Pack DIM signed Q8.8 values into flat 256-bit word (VlWide<8>)
static void pack_scores(Vonline_softmax *dut, const int16_t sc[DIM]) {
    memset(&dut->scores_flat, 0, sizeof(dut->scores_flat));
    for (int i = 0; i < DIM; i++) {
        uint16_t v  = (uint16_t)sc[i];
        int word    = (i * 16) / 32;
        int shift   = (i * 16) % 32;
        dut->scores_flat[word] |= ((uint32_t)v << shift);
    }
}

// Unpack DIM Q8.8 values from flat output
static void unpack_softmax(const Vonline_softmax *dut, int16_t out[DIM]) {
    for (int i = 0; i < DIM; i++) {
        int word  = (i * 16) / 32;
        int shift = (i * 16) % 32;
        out[i] = (int16_t)((dut->softmax_flat[word] >> shift) & 0xFFFF);
    }
}

// Reference softmax over n float scores → Q8.8 output
static void ref_softmax_q88(const double scores_f[], int n, int16_t out_q88[]) {
    double max_s = scores_f[0];
    for (int i = 1; i < n; i++) if (scores_f[i] > max_s) max_s = scores_f[i];
    double sum = 0.0;
    std::vector<double> ev(n);
    for (int i = 0; i < n; i++) { ev[i] = exp(scores_f[i] - max_s); sum += ev[i]; }
    for (int i = 0; i < n; i++) {
        int q = (int)round(ev[i] / sum * 256.0);
        if (q < 0)     q = 0;
        if (q > 65535) q = 65535;
        out_q88[i] = (int16_t)q;
    }
}

// Wait up to max_cycles for out_valid; return true if it fires
static bool wait_for_valid(Vonline_softmax *dut, int max_cycles = 200) {
    for (int c = 0; c < max_cycles; c++) {
        tick(dut);
        if (dut->out_valid) return true;
    }
    return false;
}

// Wait for FSM to return to IDLE (out_valid deasserted) without full reset
static void wait_idle(Vonline_softmax *dut, int max_cycles = 50) {
    for (int c = 0; c < max_cycles; c++) {
        tick(dut);
        if (!dut->out_valid) return;
    }
}

// Compare hw vs ref; return max relative error (ignoring ±2 LSB quantisation noise)
static double compare(const int16_t hw[], const int16_t ref[], int n, bool verbose) {
    double max_rel = 0.0;
    for (int i = 0; i < n; i++) {
        double hw_f  = (double)hw[i]  / 256.0;
        double ref_f = (double)ref[i] / 256.0;
        double abs_e = fabs(hw_f - ref_f);
        double rel_e = 0.0;
        if (abs((int)hw[i] - (int)ref[i]) > 2)
            rel_e = (ref_f > 1e-4) ? abs_e / ref_f : 0.0;
        if (rel_e > max_rel) max_rel = rel_e;
        if (verbose)
            printf("  [%2d] hw=%6.4f ref=%6.4f  abs=%.4f  rel=%.2f%%\n",
                   i, hw_f, ref_f, abs_e, rel_e * 100.0);
    }
    return max_rel;
}

// Send one tile and optionally wait for valid output
static bool send_tile(Vonline_softmax *dut,
                      const int16_t sc[DIM],
                      bool is_start, bool is_last,
                      int16_t out[DIM])
{
    pack_scores(dut, sc);
    dut->tile_start = is_start ? 1 : 0;
    dut->tile_valid = 1;
    dut->tile_last  = is_last  ? 1 : 0;
    tick(dut);
    dut->tile_start = 0;
    dut->tile_valid = 0;
    dut->tile_last  = 0;

    if (!is_last) {
        for (int c = 0; c < 60; c++) tick(dut);
        return true;
    }

    bool got = wait_for_valid(dut);
    if (got && out) unpack_softmax(dut, out);
    return got;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    Vonline_softmax *dut = new Vonline_softmax{ctx};

    srand(42);
    int total_pass = 0, total_fail = 0;

    // ── Test 1: Single tile ───────────────────────────────────────────────
    printf("=== Test 1: Single tile (DIM=%d) ===\n", DIM);
    reset(dut);

    int16_t sc1[DIM];
    double  sc1_f[DIM];
    for (int i = 0; i < DIM; i++) {
        sc1[i]   = (int16_t)(-(rand() % (4 * 256)));  // Q8.8 in [-1024, 0]
        sc1_f[i] = (double)sc1[i] / 256.0;
    }
    int16_t ref1[DIM], hw1[DIM] = {};
    ref_softmax_q88(sc1_f, DIM, ref1);

    bool got1 = send_tile(dut, sc1, /*start*/true, /*last*/true, hw1);

    printf("  DEBUG sc1 [0..3] Q8.8: %d %d %d %d\n",
           sc1[0], sc1[1], sc1[2], sc1[3]);
    printf("  DEBUG ref1[0..3]     : %d %d %d %d\n",
           ref1[0], ref1[1], ref1[2], ref1[3]);
    printf("  DEBUG hw1 [0..3]     : %d %d %d %d\n",
           hw1[0], hw1[1], hw1[2], hw1[3]);

    double err1 = got1 ? compare(hw1, ref1, DIM, false) : 1.0;
    printf("  out_valid: %s\n", got1 ? "YES" : "NO (TIMEOUT)");
    printf("  Max relative error  : %.4f%%  (threshold 1.00%%)\n", err1 * 100.0);
    bool pass1 = got1 && (err1 < 0.01);
    printf("  RESULT: %s\n\n", pass1 ? "PASS" : "FAIL");
    if (pass1) total_pass++; else total_fail++;

    // ── Test 2: Dual tile ─────────────────────────────────────────────────
    printf("=== Test 2: Dual tile (2 × DIM=%d) ===\n", DIM);
    reset(dut);

    int16_t sc2[2 * DIM];
    double  sc2_f[2 * DIM];
    for (int i = 0; i < 2 * DIM; i++) {
        sc2[i]   = (int16_t)(-(rand() % (4 * 256)));
        sc2_f[i] = (double)sc2[i] / 256.0;
    }
    // Reference: global softmax over all 32 scores; hw produces last 16
    int16_t ref2_all[2 * DIM];
    ref_softmax_q88(sc2_f, 2 * DIM, ref2_all);
    int16_t *ref2_last = ref2_all + DIM;

    // Tile 1 (no output expected)
    send_tile(dut, sc2,        /*start*/true,  /*last*/false, nullptr);
    // Tile 2
    int16_t hw2[DIM] = {};
    bool got2 = send_tile(dut, sc2 + DIM, /*start*/false, /*last*/true,  hw2);

    double err2 = got2 ? compare(hw2, ref2_last, DIM, false) : 1.0;
    printf("  out_valid: %s\n", got2 ? "YES" : "NO (TIMEOUT)");
    printf("  Comparing hw vs global softmax[%d..%d]\n", DIM, 2 * DIM - 1);
    printf("  Max relative error  : %.4f%%  (threshold 1.00%%)\n", err2 * 100.0);
    bool pass2 = got2 && (err2 < 0.01);
    printf("  RESULT: %s\n\n", pass2 ? "PASS" : "FAIL");
    if (pass2) total_pass++; else total_fail++;

    // ── Test 3: Multi-sequence WITHOUT rst_n ──────────────────────────────
    //
    // After Test 2, running_max is set to some value from those scores.
    // We now start a NEW sequence by asserting tile_start=1 — WITHOUT
    // calling reset() — using scores in a much smaller range [-0.5, 0].
    //
    // A CORRECT RTL resets running_max to tile_max when tile_start=1.
    // A BUGGY RTL keeps the stale large running_max, causing:
    //   - all LUT addrs ≈ 0  (exp ≈ 0)
    //   - running_sum ≈ 0   → division by near-zero → garbage output
    //
    // Scores are in [-128, 0] (Q8.8 = [-0.5, 0]) — much smaller than
    // Test 2's [-4, 0] — to maximise the gap with the stale running_max.
    printf("=== Test 3: New sequence via tile_start=1 (NO rst_n) ===\n");
    printf("  (Checks that running_max resets correctly on tile_start)\n");

    // Let FSM settle back to IDLE after Test 2
    wait_idle(dut);

    int16_t sc3[DIM];
    double  sc3_f[DIM];
    for (int i = 0; i < DIM; i++) {
        sc3[i]   = (int16_t)(-(rand() % 128));   // Q8.8 in [-0.5, 0]
        sc3_f[i] = (double)sc3[i] / 256.0;
    }
    int16_t ref3[DIM], hw3[DIM] = {};
    ref_softmax_q88(sc3_f, DIM, ref3);

    // New sequence: tile_start=1, tile_last=1, NO reset()
    bool got3 = send_tile(dut, sc3, /*start*/true, /*last*/true, hw3);

    printf("  DEBUG sc3 [0..3] Q8.8: %d %d %d %d\n",
           sc3[0], sc3[1], sc3[2], sc3[3]);
    printf("  DEBUG ref3[0..3]     : %d %d %d %d\n",
           ref3[0], ref3[1], ref3[2], ref3[3]);
    printf("  DEBUG hw3 [0..3]     : %d %d %d %d\n",
           hw3[0], hw3[1], hw3[2], hw3[3]);

    double err3 = got3 ? compare(hw3, ref3, DIM, false) : 1.0;
    printf("  out_valid: %s\n", got3 ? "YES" : "NO (TIMEOUT)");
    printf("  Max relative error  : %.4f%%  (threshold 1.00%%)\n", err3 * 100.0);
    bool pass3 = got3 && (err3 < 0.01);
    printf("  RESULT: %s\n", pass3 ? "PASS" : "FAIL");
    if (!pass3)
        printf("  NOTE: FAIL here = running_max not reset on tile_start.\n"
               "        Fix in online_softmax.sv S_RESCALE:\n"
               "          running_max <= is_first_r ? tile_max : next_max;\n");
    printf("\n");
    if (pass3) total_pass++; else total_fail++;

    // ── Summary ───────────────────────────────────────────────────────────
    printf("=== Summary: %d/3 tests PASS ===\n", total_pass);
    printf("RESULT: %s\n", (total_fail == 0) ? "PASS" : "FAIL");

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : online_softmax\n");
    printf("Scenarios covered: single_tile, dual_tile, running_max_reset\n");
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", total_fail);
    printf("Result           : %s\n", (total_fail == 0) ? "PASS" : "FAIL");

    dut->final();
    delete dut;
    delete ctx;
    return (total_fail == 0) ? 0 : 1;
}