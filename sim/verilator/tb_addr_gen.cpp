#include "Vaddr_gen.h"
#include "verilated.h"
#include <cstdio>

static void tick(Vaddr_gen *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    Vaddr_gen *dut = new Vaddr_gen{ctx};

    // Reset
    dut->rst_n    = 0;
    dut->cnt_en   = 0;
    dut->cnt_clr  = 0;
    dut->seq_len  = 0;
    dut->head_dim = 0;
    dut->tile_size = 0;
    dut->tile_row  = 0;
    dut->tile_col  = 0;
    tick(dut); tick(dut);
    dut->rst_n = 1;

    // ── Test 1: 溢位修正確認 ──────────────────────
    // tile_row=1024, head_dim=64 → 修好應該是 65536
    // 修前（16-bit truncate）會是 0
    dut->tile_row  = 1024;
    dut->head_dim  = 64;
    dut->tile_col  = 512;
    dut->tile_size = 64;
    dut->eval();

    printf("q_global_offset = %u (expected 65536)\n", dut->q_global_offset);
    printf("k_global_offset = %u (expected 32768)\n", dut->k_global_offset);

    bool pass1 = (dut->q_global_offset == 65536) &&
                 (dut->k_global_offset == 32768);
    printf("Test 1 (overflow fix): %s\n\n", pass1 ? "PASS" : "FAIL");

    // ── Test 2: counter 計到底會停 ───────────────
    // tile_size=4, head_dim=4 → total_elements=16，應計 0~15
    dut->tile_row  = 0;
    dut->tile_col  = 0;
    dut->tile_size = 4;
    dut->head_dim  = 4;
    dut->cnt_clr   = 1; tick(dut);
    dut->cnt_clr   = 0;
    dut->cnt_en    = 1;

    int last_addr = -1;
    bool done_seen = false;
    for (int i = 0; i < 20; i++) {
        tick(dut);
        printf("  cycle %2d: sram_addr=%d cnt_done=%d\n",
               i, (int)dut->sram_addr, (int)dut->cnt_done);
        if (dut->cnt_done) done_seen = true;
        last_addr = dut->sram_addr;
    }
    bool pass2 = done_seen && (last_addr == 15);
    printf("Test 2 (counter stops at 15): %s\n\n", pass2 ? "PASS" : "FAIL");

    printf("RESULT: %s\n", (pass1 && pass2) ? "PASS" : "FAIL");

    int mismatches = (pass1 ? 0 : 1) + (pass2 ? 0 : 1);
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : addr_gen\n");
    printf("Scenarios covered: overflow_fix, counter_stop\n");
    printf("Test cases run   : 2\n");
    printf("Mismatches       : %d\n", mismatches);
    printf("Result           : %s\n", mismatches == 0 ? "PASS" : "FAIL");

    dut->final();
    delete dut; delete ctx;
    return (pass1 && pass2) ? 0 : 1;
}