#include "Vkv_tile_buffer.h"
#include "verilated.h"
#include <cstdio>

static void tick(Vkv_tile_buffer *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

static void reset(Vkv_tile_buffer *dut) {
    dut->rst_n      = 0;
    dut->swap_banks = 0;
    dut->we_ext     = 0;
    dut->waddr_ext  = 0;
    dut->wdata_ext  = 0;
    dut->re_int     = 0;
    dut->raddr_int  = 0;
    tick(dut); tick(dut);
    dut->rst_n = 1;
}

int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    Vkv_tile_buffer *dut = new Vkv_tile_buffer{ctx};

    reset(dut);

    // ── Test 1: Write Bank0, read Bank1 (bank_sel=0) ──────────────
    // bank_sel=0 after reset: write→Bank0, read→Bank1
    // Write 0xAA to addr=10 in Bank0
    dut->we_ext    = 1;
    dut->waddr_ext = 10;
    dut->wdata_ext = 0xAA;
    tick(dut);
    dut->we_ext = 0;

    // Read addr=10 from Bank1 (should be uninitialized, not 0xAA)
    dut->re_int    = 1;
    dut->raddr_int = 10;
    tick(dut); tick(dut);
    bool pass1 = (dut->rdata_int != 0xAA);  
    printf("Test 1 (write Bank0, read Bank1 before swap): rdata=0x%02X  %s\n",
           dut->rdata_int, pass1 ? "PASS" : "FAIL");

    
    dut->re_int     = 0;
    dut->swap_banks = 1;
    tick(dut);              // bank_sel flips to 1
    dut->swap_banks = 0;

    // Now bank_sel=1: write→Bank1, read→Bank0
    // Read addr=10 from Bank0 (should be 0xAA)
    dut->re_int    = 1;
    dut->raddr_int = 10;
    tick(dut); tick(dut);
    bool pass2 = (dut->rdata_int == 0xAA);
    printf("Test 2 (after swap, read Bank0 gets 0xAA): rdata=0x%02X  %s\n",
           dut->rdata_int, pass2 ? "PASS" : "FAIL");


    dut->re_int = 0;
    dut->we_ext = 1;
    for (int i = 0; i < 8; i++) {
        dut->waddr_ext = i;
        dut->wdata_ext = (uint8_t)(i * 3 + 1);
        tick(dut);
    }
    dut->we_ext = 0;

   
    dut->swap_banks = 1;
    tick(dut);
    dut->swap_banks = 0;

    bool pass3 = true;
    dut->re_int = 1;
    for (int i = 0; i < 8; i++) {
        dut->raddr_int = i;
        tick(dut); tick(dut);
        uint8_t expected = (uint8_t)(i * 3 + 1);
        if (dut->rdata_int != expected) {
            printf("  FAIL at addr=%d: got 0x%02X expected 0x%02X\n",
                   i, dut->rdata_int, expected);
            pass3 = false;
        }
    }
    printf("Test 3 (ping-pong full cycle): %s\n", pass3 ? "PASS" : "FAIL");

    bool all_pass = pass1 && pass2 && pass3;
    printf("\nRESULT: %s\n", all_pass ? "PASS" : "FAIL");

    int mismatches = (pass1 ? 0 : 1) + (pass2 ? 0 : 1) + (pass3 ? 0 : 1);
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : kv_tile_buffer\n");
    printf("Scenarios covered: write_before_swap, read_after_swap, pingpong_full_cycle\n");
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", mismatches);
    printf("Result           : %s\n", all_pass ? "PASS" : "FAIL");

    dut->final();
    delete dut; delete ctx;
    return all_pass ? 0 : 1;
}