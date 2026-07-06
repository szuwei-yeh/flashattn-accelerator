// tb_flash_attn_top_dma.cpp — End-to-end test for flash_attn_top_dma.
//
// The accelerator pulls Q/K/V from a behavioral AXI4 DRAM model via its DMA
// engine (no direct SRAM preload).  We sweep the DRAM read latency to show
// that correctness is latency-independent and that per-tile prefetch hides
// most of the latency behind compute.
//
// Usage: ./sim_dma_top [--N <n>] [--D <d>] [--data <dir>]
// Reuses the existing data/N*/ golden vectors.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "Vtb_dma_harness.h"
#include "verilated.h"

static bool load_int8_hex(const char *path, std::vector<int8_t> &out, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("ERROR: cannot open %s\n", path); return false; }
    out.resize(n);
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%X", &v) != 1) { printf("ERROR: short read %s[%d]\n", path, i); fclose(f); return false; }
        out[i] = (int8_t)(v & 0xFF);
    }
    fclose(f); return true;
}
static bool load_int32_hex(const char *path, std::vector<int32_t> &out, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("ERROR: cannot open %s\n", path); return false; }
    out.resize(n);
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%X", &v) != 1) { printf("ERROR: short read %s[%d]\n", path, i); fclose(f); return false; }
        out[i] = (int32_t)v;
    }
    fclose(f); return true;
}
static bool load_scales(const char *path, uint16_t &sq, uint16_t &sk, uint16_t &sv) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    unsigned a = 0x0100, b = 0x0100, c = 0x0100; char line[128];
    while (fgets(line, sizeof(line), f)) {
        unsigned v;
        if (sscanf(line, "scale_q_q88 = 0x%X", &v) == 1) a = v;
        if (sscanf(line, "scale_k_q88 = 0x%X", &v) == 1) b = v;
        if (sscanf(line, "scale_v_q88 = 0x%X", &v) == 1) c = v;
    }
    fclose(f); sq = a; sk = b; sv = c; return true;
}

static void tick(Vtb_dma_harness *dut) { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); }

struct RunResult { bool pass; int fail_count; int total_cycles; double max_rel; };

static RunResult run_once(int N, int D, int rd_latency,
                          const std::vector<int8_t> &q, const std::vector<int8_t> &k,
                          const std::vector<int8_t> &v, const std::vector<int32_t> &expected,
                          uint16_t sq, uint16_t sk, uint16_t sv) {
    Vtb_dma_harness *dut = new Vtb_dma_harness;
    const int MAT = N * D;

    // Reset
    dut->rst_n = 0; dut->start = 0; dut->causal = 0;
    dut->init_we = 0; dut->out_raddr = 0;
    dut->rd_latency = rd_latency;
    dut->scale_q = sq; dut->scale_k = sk; dut->scale_v = sv;
    tick(dut); tick(dut);
    dut->rst_n = 1; tick(dut);

    // Backdoor-load DRAM image: Q@0, K@MAT, V@2*MAT
    auto dram_load = [&](int base, const std::vector<int8_t> &src) {
        for (int i = 0; i < (int)src.size(); i++) {
            dut->init_we = 1; dut->init_addr = base + i; dut->init_data = (uint8_t)src[i];
            tick(dut);
        }
        dut->init_we = 0; tick(dut);
    };
    dram_load(0,       q);
    dram_load(MAT,     k);
    dram_load(2 * MAT, v);

    // Kick off: DMA streams Q then KV tiles; core starts after tile 0
    dut->start = 1; tick(dut); dut->start = 0;

    int max_cycle = (N * N * 15 * (D / 16)) + 50000;
    int cycles = 0; bool finished = false;
    while (cycles < max_cycle) {
        tick(dut); cycles++;
        if (dut->done) { finished = true; break; }
    }
    if (!finished) {
        printf("  [lat=%3d] TIMEOUT after %d cycles\n", rd_latency, max_cycle);
        dut->final(); delete dut;
        return {false, MAT, cycles, 0.0};
    }

    // Drain output buffer
    std::vector<int32_t> hw(MAT);
    for (int i = 0; i < MAT; i++) {
        dut->out_raddr = (uint16_t)i; tick(dut); tick(dut);
        hw[i] = dut->out_rdata;
    }

    double max_rel = 0.0; int fail = 0;
    for (int i = 0; i < MAT; i++) {
        double hwf = hw[i] / 256.0, ef = expected[i] / 256.0;
        double ae = fabs(hwf - ef), ref = fabs(ef);
        double re = (ref > 1e-3) ? ae / ref : 0.0;
        if (re > max_rel) max_rel = re;
        if (re > 0.05 && ref > 1e-3) fail++;
    }
    dut->final(); delete dut;
    return {fail == 0, fail, cycles, max_rel};
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    int N = 64, D = 16;
    std::string data_dir = "../../data/N64";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
    }
    const int MAT = N * D;

    std::vector<int8_t> q, k, v; std::vector<int32_t> expected;
    if (!load_int8_hex((data_dir + "/q_input.hex").c_str(), q, MAT)) return 1;
    if (!load_int8_hex((data_dir + "/k_input.hex").c_str(), k, MAT)) return 1;
    if (!load_int8_hex((data_dir + "/v_input.hex").c_str(), v, MAT)) return 1;
    if (!load_int32_hex((data_dir + "/expected.hex").c_str(), expected, MAT)) return 1;
    uint16_t sq = 0x0100, sk = 0x0100, sv = 0x0100;
    load_scales((data_dir + "/scales.txt").c_str(), sq, sk, sv);

    printf("=== flash_attn_top_dma test: N=%d D=%d (DMA-fed from AXI DRAM model) ===\n", N, D);
    printf("  DRAM layout: Q@0  K@%d  V@%d   scales q=%04X k=%04X v=%04X\n",
           MAT, 2 * MAT, sq, sk, sv);

    int latencies[] = {0, 20, 100};
    int base_cycles = 0; int overall_fail = 0;
    printf("\n  rd_latency   result   cycles(start->done)   exposed-vs-lat0\n");
    printf("  ----------   ------   -------------------   ---------------\n");
    for (int li = 0; li < 3; li++) {
        RunResult r = run_once(N, D, latencies[li], q, k, v, expected, sq, sk, sv);
        if (li == 0) base_cycles = r.total_cycles;
        printf("  %8d     %-4s     %12d        %+8d\n",
               latencies[li], r.pass ? "PASS" : "FAIL", r.total_cycles,
               r.total_cycles - base_cycles);
        if (!r.pass) { overall_fail++; printf("     (max_rel=%.3f%% fails=%d)\n", r.max_rel * 100, r.fail_count); }
    }

    bool pass = (overall_fail == 0);
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : flash_attn_top_dma\n");
    printf("Scenarios covered: dma_fed_N%d_lat{0,20,100}\n", N);
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", overall_fail);
    printf("Result           : %s\n", pass ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
