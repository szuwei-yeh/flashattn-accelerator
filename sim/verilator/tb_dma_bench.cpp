// ============================================================
//  tb_dma_bench.cpp — Stage-2 DMA→banked_scratchpad microbenchmark.
//
//  (a) DMA loads a 256-byte tile from the DRAM model into banked_scratchpad.
//  (b) Drain it two ways and compare cycles + correctness:
//        - scalar:  1 byte/cycle  → 256 cycles
//        - stripe:  16 bytes/cycle → 16 beats (~17 cycles incl. latency)
//  Both drains must reproduce the original DRAM tile (DMA-load correctness).
// ============================================================
#include "Vtb_dma_bench.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

static Vtb_dma_bench* dut = nullptr;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

static void vget(uint8_t bytes[16]) {
    for (int w = 0; w < 4; w++)
        for (int k = 0; k < 4; k++)
            bytes[w * 4 + k] = (dut->sr_out_data[w] >> (8 * k)) & 0xFF;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_dma_bench;

    const uint32_t DRAM_BASE = 0x0400;
    const int      TILE = 256;            // 16 rows × 16 bytes
    auto pat = [](int i) -> uint8_t { return (uint8_t)((i * 11 + 5) & 0xFF); };

    // Reset
    dut->rst_n = 0; dut->start = 0; dut->sr_start = 0; dut->init_we = 0;
    dut->ext_r_en = 0; dut->ext_r_addr = 0; dut->rd_latency = 10;
    dut->dram_base = DRAM_BASE; dut->tile_bytes = TILE; dut->sr_num_rows = 16;
    tick(); tick();
    dut->rst_n = 1; tick();

    // Backdoor-load the tile into DRAM
    for (int i = 0; i < TILE; i++) {
        dut->init_we = 1; dut->init_addr = DRAM_BASE + i; dut->init_data = pat(i); tick();
    }
    dut->init_we = 0; tick();

    // (a) DMA load tile → banked_scratchpad
    int dma_cycles = 0;
    dut->start = 1; tick(); dut->start = 0;
    while (!dut->dma_load_done && dma_cycles < 100000) { tick(); dma_cycles++; }

    int errors = 0;
    if (!dut->dma_load_done) { printf("DMA load TIMEOUT\n"); dut->final(); delete dut; return 1; }

    // (b1) scalar drain: 1 byte/cycle
    int scalar_cycles = 0;
    bool scalar_ok = true;
    for (int a = 0; a < TILE; a++) {
        dut->ext_r_en = 1; dut->ext_r_addr = a; tick(); scalar_cycles++;
        uint8_t got = dut->ext_r_sdata & 0xFF;
        if (got != pat(a)) { if (scalar_ok && errors < 5) printf("  scalar FAIL @%d: %02x vs %02x\n", a, got, pat(a)); scalar_ok = false; errors++; }
    }
    dut->ext_r_en = 0; tick();

    // (b2) stripe drain: 16 bytes/cycle
    int stripe_cycles = 0, beats = 0;
    bool stripe_ok = true;
    dut->sr_start = 1; tick(); dut->sr_start = 0;
    while (!dut->sr_done && stripe_cycles < 1000) {
        if (dut->sr_out_valid) {
            uint8_t row[16]; vget(row);
            int r = dut->sr_out_row;
            for (int b = 0; b < 16; b++)
                if (row[b] != pat(r * 16 + b)) { if (stripe_ok && errors < 5) printf("  stripe FAIL row%d lane%d\n", r, b); stripe_ok = false; errors++; }
            beats++;
        }
        tick(); stripe_cycles++;
    }
    // capture a trailing valid coincident with done
    if (dut->sr_out_valid) {
        uint8_t row[16]; vget(row); int r = dut->sr_out_row;
        for (int b = 0; b < 16; b++) if (row[b] != pat(r * 16 + b)) { stripe_ok = false; errors++; }
        beats++;
    }

    printf("=== DMA → banked_scratchpad microbenchmark (tile = %d B) ===\n", TILE);
    printf("  DMA load (DRAM→scratchpad, 1 B/cyc writes) : %d cycles, correctness=%s\n",
           dma_cycles, (scalar_ok && stripe_ok) ? "PASS" : "FAIL");
    printf("\n  tile-load read path     cycles   bytes/cyc   correctness\n");
    printf("  -------------------     ------   ---------   -----------\n");
    printf("  scalar (1 B/cyc)        %6d   %9.1f   %s\n", scalar_cycles, (double)TILE / scalar_cycles, scalar_ok ? "PASS" : "FAIL");
    printf("  banked stripe (16 B)    %6d   %9.1f   %s   (%d beats)\n", stripe_cycles, (double)TILE / stripe_cycles, stripe_ok ? "PASS" : "FAIL", beats);
    double speedup = (double)scalar_cycles / stripe_cycles;
    printf("\n  stripe speedup vs scalar tile-load : %.1fx\n", speedup);

    bool beats_ok = (beats == 16);
    bool pass = scalar_ok && stripe_ok && beats_ok && (stripe_cycles <= 20) && (speedup >= 10.0);
    if (!beats_ok) printf("  WARN: expected 16 stripe beats, got %d\n", beats);

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : dma_bench_top\n");
    printf("Scenarios covered: dma_load_into_banked, scalar_drain, stripe_drain_16Bpc\n");
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", errors);
    printf("Result           : %s\n", pass ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    dut->final(); delete dut;
    return pass ? 0 : 1;
}
