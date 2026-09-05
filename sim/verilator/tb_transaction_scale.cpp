#include "Vtb_transaction_scale.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>

namespace {

constexpr int16_t SCALE_A_Q = 256;
constexpr int16_t SCALE_A_K = 256;
constexpr int16_t SCALE_B_Q = 128;
constexpr int16_t SCALE_B_K = 256;
constexpr int16_t SCALE_C_Q = 64;
constexpr int16_t SCALE_C_K = 256;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        failures++;
    }
}

void tick(Vtb_transaction_scale* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

int32_t combined(int16_t scale_q, int16_t scale_k) {
    return static_cast<int32_t>(scale_q) * static_cast<int32_t>(scale_k);
}

int16_t dequantize(int32_t data, int32_t combined_scale) {
    const int64_t rounded = static_cast<int64_t>(data) * combined_scale + 128;
    const int64_t shifted = rounded >> 8;
    if (shifted > 32767) return 32767;
    if (shifted < -32768) return -32768;
    return static_cast<int16_t>(shifted);
}

void drive_accepted_start(Vtb_transaction_scale* dut) {
    dut->start = 1;
    dut->clk = 0;
    dut->eval();
    check(dut->dbg_start_accepted == 1,
          "start must be accepted while the controller is idle");
    dut->clk = 1;
    dut->eval();
    dut->start = 0;
}

bool wait_for_dequant(Vtb_transaction_scale* dut, int16_t expected,
                      const char* label) {
    for (int cycle = 0; cycle < 2000; cycle++) {
        tick(dut);
        if (dut->dbg_dequant_valid) {
            const int16_t actual = static_cast<int16_t>(dut->dbg_dequant_out);
            const int32_t acc = static_cast<int32_t>(dut->dbg_array_acc);
            if (actual != expected) {
                std::printf("FAIL: %s dequant output=%d expected=%d acc=%d\n",
                            label, actual, expected, acc);
                failures++;
                return false;
            }
            return true;
        }
    }
    std::printf("FAIL: %s timed out waiting for dequant output\n", label);
    failures++;
    return false;
}

bool wait_for_done(Vtb_transaction_scale* dut, const char* label) {
    for (int cycle = 0; cycle < 10000; cycle++) {
        tick(dut);
        if (dut->done) return true;
    }
    std::printf("FAIL: %s timed out waiting for done\n", label);
    failures++;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vtb_transaction_scale;

    dut->rst_n = 0;
    dut->start = 0;
    dut->q_we = 0;
    dut->k_we = 0;
    dut->v_we = 0;
    dut->q_waddr = 0;
    dut->k_waddr = 0;
    dut->v_waddr = 0;
    dut->q_wdata = 1;
    dut->k_wdata = 1;
    dut->v_wdata = 1;
    dut->scale_q = static_cast<uint16_t>(SCALE_A_Q);
    dut->scale_k = static_cast<uint16_t>(SCALE_A_K);
    tick(dut);
    tick(dut);
    check(static_cast<int32_t>(dut->dbg_combined_scale) == 0,
          "combined-scale register must reset to zero");

    dut->rst_n = 1;
    tick(dut);

    // Preload one 16x16 Q/K/V tile with ones.
    for (int address = 0; address < 256; address++) {
        dut->q_we = 1;
        dut->k_we = 1;
        dut->v_we = 1;
        dut->q_waddr = address;
        dut->k_waddr = address;
        dut->v_waddr = address;
        tick(dut);
    }
    dut->q_we = 0;
    dut->k_we = 0;
    dut->v_we = 0;
    tick(dut);

    // Cases A and E: the first post-reset transaction samples scale A normally.
    drive_accepted_start(dut);
    check(static_cast<int32_t>(dut->dbg_combined_scale) ==
              combined(SCALE_A_Q, SCALE_A_K),
          "Cases A/E: first accepted transaction must capture scale A");

    // Case B: changing live scale pins after acceptance must not affect A.
    dut->scale_q = static_cast<uint16_t>(SCALE_B_Q);
    dut->scale_k = static_cast<uint16_t>(SCALE_B_K);
    tick(dut);
    check(static_cast<int32_t>(dut->dbg_combined_scale) ==
              combined(SCALE_A_Q, SCALE_A_K),
          "Case B: post-accept scale change must not update the register");

    // Case C: raw start while busy is not accepted and must not capture scale C.
    dut->scale_q = static_cast<uint16_t>(SCALE_C_Q);
    dut->scale_k = static_cast<uint16_t>(SCALE_C_K);
    dut->start = 1;
    dut->clk = 0;
    dut->eval();
    check(dut->dbg_start_accepted == 0,
          "Case C: busy-time raw start must not be accepted");
    dut->clk = 1;
    dut->eval();
    dut->start = 0;
    check(static_cast<int32_t>(dut->dbg_combined_scale) ==
              combined(SCALE_A_Q, SCALE_A_K),
          "Case C: busy-time raw start must not update the register");

    const int32_t qk_acc = 16;  // all-ones 16-element dot product
    const int16_t expected_a = dequantize(
        qk_acc, combined(SCALE_A_Q, SCALE_A_K));
    wait_for_dequant(dut, expected_a,
                     "Cases A/B/C active transaction");
    wait_for_done(dut, "transaction A");

    // Case D: output SRAM has no transaction-clear traversal, so the core
    // deliberately remains single-shot until reset.  A second start must not
    // be accepted or overwrite the first transaction's locked scale.
    tick(dut);
    dut->scale_q = static_cast<uint16_t>(SCALE_B_Q);
    dut->scale_k = static_cast<uint16_t>(SCALE_B_K);
    dut->start = 1;
    dut->clk = 0;
    dut->eval();
    check(dut->dbg_start_accepted == 0,
          "Case D: post-done start must be rejected until reset");
    dut->clk = 1;
    dut->eval();
    dut->start = 0;
    check(static_cast<int32_t>(dut->dbg_combined_scale) ==
              combined(SCALE_A_Q, SCALE_A_K),
          "Case D: rejected post-done start must not update locked scale");

    if (failures == 0) {
        std::printf("PASS Case A: normal accepted-start sampling\n");
        std::printf("PASS Case B: post-accept scale changes are isolated\n");
        std::printf("PASS Case C: busy-time start cannot overwrite scale\n");
        std::printf("PASS Case D: post-done start rejected until reset\n");
        std::printf("PASS Case E: first transaction after reset captures scale\n");
        std::printf("=== Transaction-scale semantic tests PASS ===\n");
    }

    delete dut;
    return failures == 0 ? 0 : 1;
}
