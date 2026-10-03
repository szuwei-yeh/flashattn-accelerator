#include "Vdequantizer_tile.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>

static void tick(Vdequantizer_tile& dut) {
    dut.clk = 0; dut.eval();
    dut.clk = 1; dut.eval();
    dut.clk = 0; dut.eval();
}
static int16_t expected(int32_t value, int32_t scale) {
    const int64_t shifted = (int64_t(value) * int64_t(scale) + 128) >> 8;
    return int16_t(std::max<int64_t>(-32768, std::min<int64_t>(32767, shifted)));
}
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vdequantizer_tile dut;
    dut.rst_n = 0; dut.valid_in = 0; tick(dut);
    dut.rst_n = 1; tick(dut);
    std::mt19937 random(20261003);
    const std::array<int32_t,12> directed = {
        0,1,-1,127,128,-128,-129,32767,-32768,1048576,INT32_MAX,INT32_MIN};
    for (int trial = 0; trial < 200; trial++) {
        const bool tagged = trial >= int(directed.size()) && trial%4 == 0;
        const int32_t scale = tagged ? 256 : trial < int(directed.size()) ? directed[trial] : int32_t(random());
        std::array<int16_t,256> reference;
        dut.combined_scale = uint32_t(scale);
        for (int i = 0; i < 256; i++) {
            const int32_t value = tagged ? i*13-1664 : i < int(directed.size()) ? directed[i] : int32_t(random());
            dut.data_in[i] = uint32_t(value);
            reference[i] = expected(value,scale);
        }
        dut.valid_in = 1; tick(dut); dut.valid_in = 0;
        if (!dut.busy || dut.tile_ready || dut.done) return 1;
        if (trial == 50) {
            // Abort during partially assembled scores; no completion after reset.
            tick(dut); tick(dut); tick(dut);
            dut.rst_n = 0; tick(dut); dut.rst_n = 1; tick(dut);
            if (dut.busy || dut.tile_ready || dut.done) return 1;
            dut.valid_in = 1; tick(dut); dut.valid_in = 0;
        }
        int cycles = 0;
        while (!dut.done && cycles++ < 40) {
            if (dut.tile_ready) return 1;
            tick(dut);
        }
        if (!dut.done || dut.busy || !dut.tile_ready) return 1;
        for (int i = 0; i < 256; i++) {
            if (int16_t(dut.data_out[i]) != reference[i]) {
                printf("FAIL trial=%d index=%d actual=%d expected=%d\n",
                       trial,i,int16_t(dut.data_out[i]),reference[i]);
                return 1;
            }
        }
        tick(dut);
        if (dut.done || !dut.tile_ready) return 1;
    }
    puts("RESULT: PASS 200 tiles, full signed products, rounding, saturation, repeated tiles and partial-reset recovery");
    return 0;
}
