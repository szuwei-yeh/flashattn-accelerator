// ============================================================
//  tb_dma_vec_bench.cpp — Stage-3A: byte-serial vs vector DMA fill comparison.
//
//  Loads the same 256-byte tile into two DRAM models, then fills two banked
//  scratchpads:
//    A) dma_engine      — 1 byte/cycle scratchpad writes
//    B) dma_engine_vec  — 16 byte/stripe scratchpad writes
//  Reports total DMA load cycles, scratchpad write cycles, bytes/cycle, and
//  read-back correctness vs the original DRAM tile for both paths.
// ============================================================
#include "Vtb_dma_vec_bench.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

static Vtb_dma_vec_bench* dut = nullptr;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_dma_vec_bench;

    const uint32_t DRAM_BASE = 0x0400;
    const int      TILE = 256;
    auto pat = [](int i) -> uint8_t { return (uint8_t)((i * 11 + 5) & 0xFF); };

    // Reset
    dut->rst_n = 0; dut->start_a = 0; dut->start_b = 0; dut->init_we = 0;
    dut->r_a_addr = 0; dut->r_b_addr = 0; dut->rd_latency = 10;
    dut->dram_base = DRAM_BASE; dut->tile_bytes = TILE;
    tick(); tick();
    dut->rst_n = 1; tick();

    // Backdoor-load the tile into BOTH DRAMs
    for (int i = 0; i < TILE; i++) {
        dut->init_we = 1; dut->init_addr = DRAM_BASE + i; dut->init_data = pat(i); tick();
    }
    dut->init_we = 0; tick();

    // ── Path A: scalar fill ──────────────────────────────────────────
    int a_load_cycles = 0, a_wr_cycles = 0;
    dut->start_a = 1; tick(); dut->start_a = 0;
    while (!dut->done_a && a_load_cycles < 100000) {
        tick(); a_load_cycles++;
        if (dut->o_a_we) a_wr_cycles++;
    }

    // ── Path B: vector fill ──────────────────────────────────────────
    int b_load_cycles = 0, b_wr_cycles = 0;
    dut->start_b = 1; tick(); dut->start_b = 0;
    while (!dut->done_b && b_load_cycles < 100000) {
        tick(); b_load_cycles++;
        if (dut->o_b_we) b_wr_cycles++;
    }

    // ── Read-back correctness ────────────────────────────────────────
    int a_err = 0, b_err = 0;
    for (int a = 0; a < TILE; a++) {
        dut->r_a_addr = a; dut->r_b_addr = a; tick();
        if ((dut->r_a_sdata & 0xFF) != pat(a)) a_err++;
        if ((dut->r_b_sdata & 0xFF) != pat(a)) b_err++;
    }

    bool a_ok = (a_err == 0), b_ok = (b_err == 0);

    printf("=== DMA fill: byte-serial vs vector (tile = %d B) ===\n\n", TILE);
    printf("  fill path                 load cyc   sp-write cyc   bytes/cyc(sp-write)   correctness\n");
    printf("  ----------------------    --------   ------------   -------------------   -----------\n");
    printf("  A scalar  (dma_engine)    %8d   %12d   %19.1f   %s\n",
           a_load_cycles, a_wr_cycles, (double)TILE / a_wr_cycles, a_ok ? "PASS" : "FAIL");
    printf("  B vector  (dma_engine_vec)%8d   %12d   %19.1f   %s\n",
           b_load_cycles, b_wr_cycles, (double)TILE / b_wr_cycles, b_ok ? "PASS" : "FAIL");

    double wr_speedup   = (double)a_wr_cycles   / b_wr_cycles;
    double load_speedup = (double)a_load_cycles / b_load_cycles;
    printf("\n  scratchpad-write speedup : %.1fx  (%d → %d writes)\n", wr_speedup, a_wr_cycles, b_wr_cycles);
    printf("  total DMA-load speedup   : %.1fx  (%d → %d cycles)\n", load_speedup, a_load_cycles, b_load_cycles);

    // Expectations: scalar writes ~256 (1/byte), vector writes 16 (1/stripe)
    bool pass = a_ok && b_ok && (a_wr_cycles == TILE) && (b_wr_cycles == TILE / 16)
                && (b_load_cycles < a_load_cycles);

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : dma_vec_bench_top\n");
    printf("Scenarios covered: scalar_fill, vector_fill, readback_correctness\n");
    printf("Test cases run   : 2\n");
    printf("Mismatches       : %d\n", a_err + b_err);
    printf("Result           : %s\n", pass ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    dut->final(); delete dut;
    return pass ? 0 : 1;
}
