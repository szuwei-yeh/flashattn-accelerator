#include "Vsram_1r1w.h"
#include "verilated.h"
#include <cstdio>

static void tick(Vsram_1r1w *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    Vsram_1r1w *dut = new Vsram_1r1w{ctx};

    dut->we    = 1;
    dut->waddr = 5;
    dut->wdata = 0xAB;
    dut->re    = 0;
    dut->raddr = 0;
    tick(dut);

    dut->we    = 0;
    dut->re    = 1;
    dut->raddr = 5;
    tick(dut); 

    tick(dut);  
    bool pass1 = (dut->rdata == 0xAB);
    printf("Test 1 (write then read): rdata=0x%02X  %s\n",
           dut->rdata, pass1 ? "PASS" : "FAIL");

    dut->we = 1; dut->re = 0;
    for (int i = 0; i < 8; i++) {
        dut->waddr = i;
        dut->wdata = i * 10;
        tick(dut);
    }
    dut->we = 0; dut->re = 1;

    bool pass2 = true;
    for (int i = 0; i < 8; i++) {
        dut->raddr = i;
        tick(dut); tick(dut);  
        if (dut->rdata != (uint8_t)(i * 10)) {
            printf("  FAIL at addr=%d: got %d, expected %d\n",
                   i, dut->rdata, i * 10);
            pass2 = false;
        }
    }
    printf("Test 2 (sequential write/read): %s\n", pass2 ? "PASS" : "FAIL");

    printf("\nRESULT: %s\n", (pass1 && pass2) ? "PASS" : "FAIL");

    int mismatches = (pass1 ? 0 : 1) + (pass2 ? 0 : 1);
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : sram_1r1w\n");
    printf("Scenarios covered: write_then_read, sequential_write_read\n");
    printf("Test cases run   : 2\n");
    printf("Mismatches       : %d\n", mismatches);
    printf("Result           : %s\n", mismatches == 0 ? "PASS" : "FAIL");

    dut->final();
    delete dut; delete ctx;
    return (pass1 && pass2) ? 0 : 1;
}