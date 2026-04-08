// tb_exp_lut.cpp — Verilator testbench for exp_lut.sv
//
// Scans all 256 addresses, compares the Q8.8 output against
// the analytical exp(x) value, and reports max relative error.
// Acceptance criterion: max relative error < 0.5 % (Q8.8 precision floor).
//
// Build & run (from sim/verilator/):
//   verilator --cc --exe --build -Wall \
//     ../../rtl/softmax/exp_lut.sv tb_exp_lut.cpp \
//     -o sim_exp_lut --Mdir obj_exp_lut
//   ./obj_exp_lut/sim_exp_lut

#include <cstdint>
#include <cstdio>
#include <cmath>

#include "Vexp_lut.h"
#include "verilated.h"

static void tick(Vexp_lut *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    Vexp_lut *dut = new Vexp_lut{ctx};

    // addr = 0   → x = -8.0   (exp(-8)  ≈ 0.000335)
    // addr = 255 → x =  0.0   (exp( 0)  = 1.0     )
    // x(addr) = (addr / 255.0) * 8.0 - 8.0  =  addr * (8/255) - 8

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;  // for exp(x) >= 0.5
    int    fail_count  = 0;

    const double ABS_THRESHOLD = 0.002;    // 0.5 LSB + margin (Q8.8 = 1/256=0.0039)
    const double REL_THRESHOLD = 0.005;    // 0.5 % for exp >= 0.5

    uint16_t results[256];

    // 1-cycle registered output: drive addr, tick → exp_val = lut[addr]
    for (int a = 0; a < 256; a++) {
        dut->addr = (uint8_t)a;
        tick(dut);
        results[a] = dut->exp_val;
    }

    printf("addr   x        exp_true    lut_q88   lut_float   abs_err    rel_err\n");
    for (int a = 0; a < 256; a++) {
        double x        = (double)a / 255.0 * 8.0 - 8.0;
        double exp_true = exp(x);
        double lut_f    = (double)results[a] / 256.0;
        double abs_err  = fabs(lut_f - exp_true);
        double rel_err  = (exp_true > 1e-6) ? abs_err / exp_true : 0.0;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (exp_true >= 0.5 && rel_err > max_rel_err) max_rel_err = rel_err;

        if (a == 0 || a == 127 || a == 255 || rel_err > REL_THRESHOLD) {
            printf("  %3d  %6.3f   %10.6f   0x%04x    %8.6f  %9.6f  %7.4f%%\n",
                   a, x, exp_true, results[a], lut_f, abs_err, rel_err * 100.0);
        }

        if (exp_true >= 0.5 && rel_err > REL_THRESHOLD) fail_count++;
    }

    printf("\nSummary:\n");
    printf("  Max absolute error         : %.6f  (threshold %.3f)\n",
           max_abs_err, ABS_THRESHOLD);
    printf("  Max relative error (exp≥0.5): %.4f%%  (threshold %.1f%%)\n",
           max_rel_err * 100.0, REL_THRESHOLD * 100.0);

    if (max_abs_err < ABS_THRESHOLD && max_rel_err < REL_THRESHOLD) {
        printf("RESULT: PASS\n");
    } else {
        printf("RESULT: FAIL  (%d entries exceed threshold)\n", fail_count);
    }

    dut->final();
    delete dut;
    delete ctx;

    int mismatches = (max_abs_err < ABS_THRESHOLD && max_rel_err < REL_THRESHOLD) ? 0 : fail_count;
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : exp_lut\n");
    printf("Scenarios covered: full_lut_sweep\n");
    printf("Test cases run   : 256\n");
    printf("Mismatches       : %d\n", mismatches);
    printf("Result           : %s\n", mismatches == 0 ? "PASS" : "FAIL");

    return mismatches == 0 ? 0 : 1;
}
