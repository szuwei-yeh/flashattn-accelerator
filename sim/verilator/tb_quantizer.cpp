// tb_quantizer.cpp — Verilator testbench for quantizer.sv
//
// Generates 100 random Q8.8 (data_in) and Q8.8 power-of-two scale values,
// drives the DUT, and compares against the C++ reference model:
//   out = clamp(round(data_in_real / scale_real), -128, 127)
//
// Build & run (from sim/verilator/):
//   verilator --cc --exe --build -Wall \
//     ../../rtl/quantization/quantizer.sv tb_quantizer.cpp \
//     -o sim_quantizer --Mdir obj_quantizer
//   ./obj_quantizer/sim_quantizer

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "Vquantizer.h"
#include "verilated.h"

// ── Reference model ──────────────────────────────────────────────────────────
static int8_t ref_quantize(int16_t data_in_q88, int16_t scale_q88) {
    // result = clamp(round_half_away_from_zero(data_in_int / scale_int), -128, 127)
    // Matches RTL which adds -(|scale|/2) for negative quotients and
    // +(|scale|/2) for positive, then truncates toward zero.
    if (scale_q88 == 0) return 0;
    int32_t abs_num = abs((int32_t)data_in_q88);
    int32_t abs_den = abs((int32_t)scale_q88);
    int32_t q_abs   = (abs_num + abs_den / 2) / abs_den;  // round-half-up magnitude
    int32_t q       = ((data_in_q88 < 0) != (scale_q88 < 0)) ? -q_abs : q_abs;
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

// ── Helper: tick clock ────────────────────────────────────────────────────────
static void tick(Vquantizer *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    Vquantizer *dut = new Vquantizer{ctx};

    // Reset
    dut->rst_n    = 0;
    dut->valid_in = 0;
    dut->data_in  = 0;
    dut->scale    = 256;  // Q8.8 of 1.0
    tick(dut); tick(dut);
    dut->rst_n = 1;

    // Collect inputs / expected values
    const int N = 100;
    int16_t data_arr[N], scale_arr[N];
    int8_t  expected[N];

    // Use fixed seed for reproducibility
    srand(42);

    // Scales: powers of 2 in Q8.8: 32(=0.125), 64(=0.25), 128(=0.5), 256(=1.0), 512(=2.0)
    const int16_t scales[] = {32, 64, 128, 256, 512};
    const int N_SCALES = 5;

    for (int i = 0; i < N; i++) {
        // Random Q8.8 in [-128*256, 127*256] = [-32768, 32512]
        data_arr[i]  = (int16_t)((rand() % 65024) - 32512);
        scale_arr[i] = scales[rand() % N_SCALES];
        expected[i]  = ref_quantize(data_arr[i], scale_arr[i]);
    }

    // Drive DUT: 1-cycle latency — drive input[i], tick, then sample output[i].
    int pass = 0, fail = 0;
    int max_err = 0;

    for (int i = 0; i < N; i++) {
        dut->valid_in = 1;
        dut->data_in  = data_arr[i];
        dut->scale    = scale_arr[i];
        tick(dut);

        // Output for input[i] is ready after the clock edge above
        if (dut->valid_out) {
            int8_t got = (int8_t)dut->data_out;
            int8_t exp = expected[i];
            int err    = abs((int)got - (int)exp);
            if (err > max_err) max_err = err;
            if (err == 0) {
                pass++;
            } else {
                if (fail < 5)
                    printf("FAIL[%d]: data_in=%d scale=%d  got=%d  exp=%d\n",
                           i, data_arr[i], scale_arr[i], (int)got, (int)exp);
                fail++;
            }
        }
    }
    // Drain pipeline
    dut->valid_in = 0; tick(dut);

    printf("quantizer: %d/%d PASS, %d FAIL, max_err=%d\n", pass, pass+fail, fail, max_err);
    if (fail == 0)
        printf("RESULT: PASS\n");
    else
        printf("RESULT: FAIL\n");

    dut->final();
    delete dut;
    delete ctx;

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : quantizer\n");
    printf("Scenarios covered: random_q88, scale_sweep\n");
    printf("Test cases run   : %d\n", N);
    printf("Mismatches       : %d\n", fail);
    printf("Result           : %s\n", (fail == 0) ? "PASS" : "FAIL");

    return (fail == 0) ? 0 : 1;
}
