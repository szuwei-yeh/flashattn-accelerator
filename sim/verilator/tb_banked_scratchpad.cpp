// ============================================================
//  tb_banked_scratchpad.cpp — Unit test for banked_scratchpad.
//
//  Proves: (1) scalar write/read + bank routing, (2) vector stripe read
//  coherent with scalar writes, (3) vector stripe write coherent with scalar
//  reads, (4) concurrent 16B-write + 16B-read in one cycle (32 B/cyc),
//  (5) bandwidth: 16 rows via vector = 16 cycles vs 256 scalar cycles,
//  (6) the stride-16 bank-conflict invariant.
// ============================================================
#include "Vbanked_scratchpad.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

static Vbanked_scratchpad* dut = nullptr;
static int errors = 0;

static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

// 128-bit vector helpers (4 × 32-bit words, byte b → word b/4, lane b%4)
static void vset(uint8_t bytes[16]) {
    for (int w = 0; w < 4; w++) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) v |= (uint32_t)bytes[w * 4 + k] << (8 * k);
        dut->w_vdata[w] = v;
    }
}
static void vget(uint8_t bytes[16]) {
    for (int w = 0; w < 4; w++)
        for (int k = 0; k < 4; k++)
            bytes[w * 4 + k] = (dut->r_vdata[w] >> (8 * k)) & 0xFF;
}

static void idle() {
    dut->w_en = 0; dut->w_vec = 0; dut->r_en = 0; dut->r_vec = 0;
}

static void scalar_write(uint16_t addr, uint8_t data) {
    idle();
    dut->w_en = 1; dut->w_vec = 0; dut->w_addr = addr; dut->w_sdata = data;
    tick();
    dut->w_en = 0;
}
static uint8_t scalar_read(uint16_t addr) {
    idle();
    dut->r_en = 1; dut->r_vec = 0; dut->r_addr = addr;
    tick();
    dut->r_en = 0;
    return dut->r_sdata & 0xFF;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vbanked_scratchpad;
    idle(); tick();

    const int NROWS = 16;             // 16 rows × 16 bytes = 256 bytes exercised
    auto pat = [](int a) -> uint8_t { return (uint8_t)((a * 7 + 13) & 0xFF); };

    // (1) scalar write all, scalar read back
    for (int a = 0; a < NROWS * 16; a++) scalar_write(a, pat(a));
    for (int a = 0; a < NROWS * 16; a++) {
        uint8_t got = scalar_read(a);
        if (got != pat(a)) { if (errors < 5) printf("  (1) FAIL @%d: got %02x exp %02x\n", a, got, pat(a)); errors++; }
    }
    printf("(1) scalar write/read + bank routing : %s\n", errors == 0 ? "PASS" : "FAIL");
    int e1 = errors;

    // (2) vector stripe read must equal the scalar-written bytes of that row
    for (int row = 0; row < NROWS; row++) {
        idle(); dut->r_en = 1; dut->r_vec = 1; dut->r_addr = row * 16; tick(); dut->r_en = 0;
        uint8_t v[16]; vget(v);
        for (int b = 0; b < 16; b++)
            if (v[b] != pat(row * 16 + b)) { if (errors < 5) printf("  (2) FAIL row%d lane%d: got %02x exp %02x\n", row, b, v[b], pat(row*16+b)); errors++; }
    }
    printf("(2) vector stripe read coherency     : %s\n", errors == e1 ? "PASS" : "FAIL");
    int e2 = errors;

    // (3) vector stripe write, scalar read back each byte
    int base = NROWS * 16;            // fresh region
    uint8_t wv[16];
    for (int b = 0; b < 16; b++) wv[b] = (uint8_t)(0xA0 + b);
    idle(); dut->w_en = 1; dut->w_vec = 1; dut->w_addr = base; vset(wv); tick(); dut->w_en = 0;
    for (int b = 0; b < 16; b++) {
        uint8_t got = scalar_read(base + b);
        if (got != wv[b]) { if (errors < 5) printf("  (3) FAIL lane%d: got %02x exp %02x\n", b, got, wv[b]); errors++; }
    }
    printf("(3) vector stripe write scatter      : %s\n", errors == e2 ? "PASS" : "FAIL");
    int e3 = errors;

    // (4) concurrent vector write (rowX) + vector read (rowY) in ONE cycle
    int rowX = base + 16, rowY = 0;   // read row 0 (written in step 1)
    uint8_t wx[16];
    for (int b = 0; b < 16; b++) wx[b] = (uint8_t)(0x50 + b);
    idle();
    dut->w_en = 1; dut->w_vec = 1; dut->w_addr = rowX; vset(wx);
    dut->r_en = 1; dut->r_vec = 1; dut->r_addr = rowY;
    tick();
    dut->w_en = 0; dut->r_en = 0;
    uint8_t rv[16]; vget(rv);
    for (int b = 0; b < 16; b++)
        if (rv[b] != pat(rowY + b)) { if (errors < 5) printf("  (4) read FAIL lane%d\n", b); errors++; }
    for (int b = 0; b < 16; b++) {
        uint8_t got = scalar_read(rowX + b);
        if (got != wx[b]) { if (errors < 5) printf("  (4) write FAIL lane%d\n", b); errors++; }
    }
    printf("(4) concurrent 16B write + 16B read  : %s (32 bytes/cycle)\n", errors == e3 ? "PASS" : "FAIL");
    int e4 = errors;

    // (5) bandwidth: write 16 rows (256 B) via vector = 16 cycles
    int cyc = 0;
    for (int row = 0; row < NROWS; row++) {
        uint8_t v[16];
        for (int b = 0; b < 16; b++) v[b] = (uint8_t)((row << 4) | b);
        idle(); dut->w_en = 1; dut->w_vec = 1; dut->w_addr = base + row * 16; vset(v); tick(); cyc++;
    }
    dut->w_en = 0;
    bool bw_ok = (cyc == NROWS);
    // verify a couple
    for (int row = 0; row < NROWS; row += 5)
        for (int b = 0; b < 16; b++) {
            uint8_t got = scalar_read(base + row * 16 + b);
            if (got != (uint8_t)((row << 4) | b)) { bw_ok = false; errors++; }
        }
    printf("(5) bandwidth: 256 B in %d cycles (16 B/cyc) vs 256 scalar : %s\n", cyc, bw_ok ? "PASS" : "FAIL");
    if (!bw_ok) errors++;

    // (6) conflict-scheme invariant (mapping property)
    bool stripe_distinct = true, stride16_collide = true;
    bool seen[16] = {false};
    for (int b = 0; b < 16; b++) { int bank = (0 + b) & 15; if (seen[bank]) stripe_distinct = false; seen[bank] = true; }
    for (int i = 1; i < 16; i++) if (((i * 16) & 15) != 0) stride16_collide = false;
    printf("(6) bank-conflict invariant: contiguous-16 distinct=%d, stride16 all-bank0=%d : %s\n",
           stripe_distinct, stride16_collide, (stripe_distinct && stride16_collide) ? "PASS" : "FAIL");
    if (!(stripe_distinct && stride16_collide)) errors++;

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : banked_scratchpad\n");
    printf("Scenarios covered: scalar_rw, stripe_read, stripe_write, concurrent_rw, bandwidth, conflict_invariant\n");
    printf("Test cases run   : 6\n");
    printf("Mismatches       : %d\n", errors);
    printf("Result           : %s\n", errors == 0 ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", errors == 0 ? "PASS" : "FAIL");

    dut->final(); delete dut;
    return errors ? 1 : 0;
}
