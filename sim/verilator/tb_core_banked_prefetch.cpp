// ============================================================
//  tb_core_banked_prefetch.cpp — end-to-end test for
//  flash_attn_core_banked_prefetch (banked core WITH KV double-buffering).
//
//  Identical flow to tb_core_banked.cpp: preloads Q/K/V banked scratchpads via
//  scalar writes (no DMA, kv_tiles_ready tied high), runs attention, compares to
//  the golden expected.hex, and reports the cycle count so the prefetch speedup
//  vs core_banked can be read off directly.
//
//  Usage: ./sim_core_banked_prefetch [--N <n>] [--D <d>] [--data <dir>]
//         [--causal] [--stress-config-lock]
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "Vflash_attn_core_banked_prefetch.h"
#include "verilated.h"
#include "tb_fixture_checks.h"

static void tick(Vflash_attn_core_banked_prefetch* dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}


int main(int argc, char** argv) {
    int N = TB_SEQ_LEN, D = TB_HEAD_DIM; std::string data = "../../data/N64";
    bool causal = false;
    bool stress_config_lock = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i + 1 < argc) data = argv[++i];
        else if (!strcmp(argv[i], "--causal")) causal = true;
        else if (!strcmp(argv[i], "--stress-config-lock")) stress_config_lock = true;
    }
    if (!check_tb_geometry(N, D, TB_SEQ_LEN, TB_HEAD_DIM)) return 1;
    const int MAT = N * D;

    std::vector<int8_t> q, k, v; std::vector<int32_t> exp;
    if (!load_hex_vector((data + "/q_input.hex").c_str(), q, MAT)) return 1;
    if (!load_hex_vector((data + "/k_input.hex").c_str(), k, MAT)) return 1;
    if (!load_hex_vector((data + "/v_input.hex").c_str(), v, MAT)) return 1;
    if (!load_hex_vector((data + "/expected.hex").c_str(), exp, MAT)) return 1;
    uint16_t sq = 0x100, sk = 0x100, sv = 0x100;
    if (!load_required_scales(data + "/scales.txt", sq, sk, sv)) return 1;

    Verilated::commandArgs(argc, argv);
    auto* dut = new Vflash_attn_core_banked_prefetch;

    printf("=== flash_attn_core_banked_prefetch: N=%d D=%d (banked + KV double-buffer) ===\n", N, D);

    // Reset
    dut->rst_n = 0; dut->start = 0; dut->mode = 0; dut->kv_len = 0; dut->causal = causal;
    dut->q_we = dut->k_we = dut->v_we = 0;
    // Vector DMA port unused in this core-level test (scalar preload path);
    // dma_v_we=0 disables it, dma_v_data (128-bit VlWide) left as don't-care.
    dut->dma_v_we = 0; dut->dma_v_dst = 0; dut->dma_v_addr = 0;
    dut->out_raddr = 0;
    dut->scale_q = sq; dut->scale_k = sk; dut->scale_v = sv;
    dut->kv_tiles_ready = 0xFFFF;      // everything preloaded
    tick(dut); tick(dut);
    dut->rst_n = 1; tick(dut);

    // Preload scratchpads (scalar byte writes, row-major)
    for (int i = 0; i < MAT; i++) {
        dut->q_we = 1; dut->q_waddr = i; dut->q_wdata = (uint8_t)q[i];
        dut->k_we = 1; dut->k_waddr = i; dut->k_wdata = (uint8_t)k[i];
        dut->v_we = 1; dut->v_waddr = i; dut->v_wdata = (uint8_t)v[i];
        tick(dut);
    }
    dut->q_we = dut->k_we = dut->v_we = 0; tick(dut);

    // Run
    int max_cycle = N * N * 20 * (D / 16) + 50000;
    int cycles = 0; bool done = false;
    dut->start = 1; tick(dut); dut->start = 0;
    while (cycles < max_cycle) {
        // Deliberately corrupt every live transaction-level input and inject
        // busy-time start pulses.  Accepted-start locking must isolate the run.
        if (stress_config_lock && cycles == 1) {
            dut->scale_q = sq ^ 0x0055;
            dut->scale_k = sk ^ 0x0033;
            dut->scale_v = sv ^ 0x0077;
            dut->causal = !causal;
            dut->mode = 1;
            dut->kv_len = 16;
        }
        dut->start = stress_config_lock &&
                     (cycles == 50 || cycles == 500 || cycles == 1500);
        tick(dut); cycles++;
        dut->start = 0;
        if (dut->done) { done = true; break; }
    }
    if (!done) { printf("TIMEOUT after %d cycles\n", max_cycle); delete dut; return 1; }

    // Read output
    std::vector<int32_t> hw(MAT);
    for (int i = 0; i < MAT; i++) {
        dut->out_raddr = i; tick(dut); tick(dut);
        hw[i] = (int32_t)dut->out_rdata;
    }

    int exact_fail = 0;
    int numerical_fail = 0;
    double max_rel = 0.0, max_abs = 0.0;
    for (int i = 0; i < MAT; i++) {
        double h = hw[i] / 256.0, e = exp[i] / 256.0;
        double ae = fabs(h - e), ref = fabs(e);
        double re = (ref > 1e-3) ? ae / ref : 0.0;
        if (ae > max_abs) max_abs = ae;
        if (re > max_rel) max_rel = re;
        if ((ref > 1e-3 && re > 0.05) || (ref <= 1e-3 && ae > 0.05))
            numerical_fail++;
        if (hw[i] != exp[i]) {
            if (exact_fail < 5)
                printf("  exact mismatch [%d] hw=0x%08X expected=0x%08X\n",
                       i, (uint32_t)hw[i], (uint32_t)exp[i]);
            exact_fail++;
        }
    }

    bool pass = (exact_fail == 0);
    printf("  done at cycle %d\n", cycles);
    printf("  exact mismatches   : %d/%d\n", exact_fail, MAT);
    printf("  max absolute error : %.6f\n", max_abs);
    printf("  max relative error : %.4f%%   entries>5%%: %d/%d\n",
           max_rel * 100, numerical_fail, MAT);
    // Reference: banked core WITHOUT prefetch (core_banked_N64) = 11929 cycles.
    if (N == 64 && D == 16)
        printf("  reference (banked, no prefetch, core_banked_N64): 11929 cycles\n");

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : flash_attn_core_banked_prefetch\n");
    printf("Scenarios covered: banked_kv_double_buffer_end_to_end_N%d_d%d%s%s\n",
           N, D, causal ? "_causal" : "",
           stress_config_lock ? "_config_lock_stress" : "");
    printf("Test cases run   : 1\n");
    printf("Mismatches       : %d\n", exact_fail);
    printf("Result           : %s\n", pass ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    delete dut;
    return pass ? 0 : 1;
}
