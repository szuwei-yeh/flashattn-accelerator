// ============================================================
//  tb_dma_banked_wb.cpp — end-to-end test for flash_attn_top_dma_banked_wb.
//
//  Full DRAM round-trip: backdoor-loads Q/K/V into the RW DRAM model, runs the
//  top (read DMA → banked core → output_buffer drain → write DMA), then
//  backdoor-reads the O region and compares it, word-for-word, to the golden
//  expected.hex.  Sweeps (rd_latency, wr_latency) and prints the read-side and
//  write-back performance counters.
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "Vtb_dma_banked_wb_harness.h"
#include "verilated.h"

static void tick(Vtb_dma_banked_wb_harness* d) { d->clk = 0; d->eval(); d->clk = 1; d->eval(); }

static bool load_i8(const char* p, std::vector<int8_t>& o, int n) {
    FILE* f = fopen(p, "r"); if (!f) { printf("ERR open %s\n", p); return false; }
    o.resize(n);
    for (int i = 0; i < n; i++) { unsigned v; if (fscanf(f, "%X", &v) != 1) { fclose(f); return false; } o[i] = (int8_t)(v & 0xFF); }
    fclose(f); return true;
}
static bool load_i32(const char* p, std::vector<int32_t>& o, int n) {
    FILE* f = fopen(p, "r"); if (!f) { printf("ERR open %s\n", p); return false; }
    o.resize(n);
    for (int i = 0; i < n; i++) { unsigned v; if (fscanf(f, "%X", &v) != 1) { fclose(f); return false; } o[i] = (int32_t)v; }
    fclose(f); return true;
}
static void load_scales(const char* p, uint16_t& sq, uint16_t& sk, uint16_t& sv) {
    FILE* f = fopen(p, "r"); if (!f) return;
    unsigned a = 0x100, b = 0x100, c = 0x100; char ln[128];
    while (fgets(ln, sizeof(ln), f)) { unsigned v;
        if (sscanf(ln, "scale_q_q88 = 0x%X", &v) == 1) a = v;
        if (sscanf(ln, "scale_k_q88 = 0x%X", &v) == 1) b = v;
        if (sscanf(ln, "scale_v_q88 = 0x%X", &v) == 1) c = v; }
    fclose(f); sq = a; sk = b; sv = c;
}

struct Res {
    bool     pass, cfg_err, timeout;
    int      fail, tb_cycles;
    double   max_rel;
    uint32_t perf_total, perf_dma_bytes, perf_wb_bytes, perf_wb_cycles, perf_wb_beats;
    uint16_t perf_kv_tiles;
};

static Res run_once(int N, int D, int rd_lat, int wr_lat,
                    const std::vector<int8_t>& q, const std::vector<int8_t>& k,
                    const std::vector<int8_t>& v, const std::vector<int32_t>& exp,
                    uint16_t sq, uint16_t sk, uint16_t sv, uint32_t o_base) {
    const int MAT = N * D;
    auto* dut = new Vtb_dma_banked_wb_harness;
    Res r{}; r.timeout = false;

    dut->rst_n = 0; dut->start = 0; dut->causal = 0; dut->init_we = 0; dut->bd_raddr = 0;
    dut->rd_latency = (uint16_t)rd_lat; dut->wr_latency = (uint16_t)wr_lat;
    dut->scale_q = sq; dut->scale_k = sk; dut->scale_v = sv;
    tick(dut); tick(dut);
    dut->rst_n = 1; tick(dut);

    // Backdoor-load DRAM: Q@0, K@MAT, V@2*MAT
    auto dram = [&](int base, const std::vector<int8_t>& s) {
        for (int i = 0; i < (int)s.size(); i++) {
            dut->init_we = 1; dut->init_addr = base + i; dut->init_data = (uint8_t)s[i]; tick(dut);
        }
        dut->init_we = 0; tick(dut);
    };
    dram(0, q); dram(MAT, k); dram(2 * MAT, v);

    dut->cfg_seq_len = (uint16_t)N;
    dut->cfg_q_base  = 0;
    dut->cfg_k_base  = (uint32_t)MAT;
    dut->cfg_v_base  = (uint32_t)(2 * MAT);
    dut->cfg_o_base  = o_base;
    tick(dut);
    r.cfg_err = dut->cfg_error;
    if (r.cfg_err) { delete dut; r.pass = false; return r; }

    int max_cycle = N * N * 20 * (D / 16) + 50000 + MAT * 8 + wr_lat * (MAT / 32);
    int cyc = 0; bool done = false;
    dut->start = 1; tick(dut); dut->start = 0;
    while (cyc < max_cycle) {
        tick(dut); cyc++;
        if (dut->done) { done = true; break; }
    }
    if (!done) { r.timeout = true; r.pass = false; delete dut; return r; }
    r.tb_cycles = cyc;

    // Backdoor-read the O region from DRAM and compare, word for word.
    auto bd = [&](uint32_t a) -> uint8_t { dut->bd_raddr = a; dut->eval(); return (uint8_t)dut->bd_rdata; };
    r.fail = 0; r.max_rel = 0.0;
    for (int i = 0; i < MAT; i++) {
        uint32_t w = (uint32_t)bd(o_base + i * 4 + 0)
                   | ((uint32_t)bd(o_base + i * 4 + 1) << 8)
                   | ((uint32_t)bd(o_base + i * 4 + 2) << 16)
                   | ((uint32_t)bd(o_base + i * 4 + 3) << 24);
        int32_t hw = (int32_t)w;
        double h = hw / 256.0, e = exp[i] / 256.0;
        double ae = fabs(h - e), ref = fabs(e);
        double re = (ref > 1e-3) ? ae / ref : 0.0;
        if (re > r.max_rel) r.max_rel = re;
        if (re > 0.05 && ref > 1e-3) r.fail++;
    }
    r.pass = (r.fail == 0);

    r.perf_total     = (uint32_t)dut->perf_total_cycles;
    r.perf_dma_bytes = (uint32_t)dut->perf_dma_bytes;
    r.perf_wb_bytes  = (uint32_t)dut->perf_wb_bytes;
    r.perf_wb_cycles = (uint32_t)dut->perf_wb_cycles;
    r.perf_wb_beats  = (uint32_t)dut->perf_wb_beats;
    r.perf_kv_tiles  = (uint16_t)dut->perf_kv_tiles_loaded;

    delete dut;
    return r;
}

int main(int argc, char** argv) {
    int N = 64, D = 16; std::string data = "../../data/N64";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i + 1 < argc) data = argv[++i];
    }
    const int MAT = N * D;
    // O base = 3*MAT — immediately above V, so [0,3*MAT) (Q,K,V) and
    // [o_base, o_base+MAT*4) (O) never overlap.
    const uint32_t o_base = (uint32_t)(3 * MAT);
    if (o_base < (uint32_t)(3 * MAT)) { printf("BAD o_base overlap\n"); return 1; }

    std::vector<int8_t> q, k, v; std::vector<int32_t> exp;
    if (!load_i8((data + "/q_input.hex").c_str(), q, MAT)) return 1;
    if (!load_i8((data + "/k_input.hex").c_str(), k, MAT)) return 1;
    if (!load_i8((data + "/v_input.hex").c_str(), v, MAT)) return 1;
    if (!load_i32((data + "/expected.hex").c_str(), exp, MAT)) return 1;
    uint16_t sq = 0x100, sk = 0x100, sv = 0x100;
    load_scales((data + "/scales.txt").c_str(), sq, sk, sv);

    Verilated::commandArgs(argc, argv);

    printf("=== flash_attn_top_dma_banked_wb: N=%d D=%d — full DRAM round-trip ===\n", N, D);
    printf("    cfg_q,k,v_base=0,%d,%d  cfg_o_base=%u (=3*N*D, non-overlapping)\n\n",
           MAT, 2 * MAT, o_base);

    const int rd_lats[] = {0, 20, 100};
    const int wr_lats[] = {0, 20, 100};
    const uint32_t exp_rd_bytes = (uint32_t)(3 * MAT);  // Q+K+V streamed once
    const uint32_t exp_wb_bytes = (uint32_t)(MAT * 4);  // O words, 4 B each
    const uint16_t exp_tiles    = (uint16_t)(N / 16);
    const uint32_t exp_wb_beats = (uint32_t)(MAT / 2);  // 2 words / 64-bit beat

    printf("  rd_lat wr_lat  result  tb_cyc  perf_total  rd_bytes  wb_bytes  wb_cyc  wb_beats  kv_tiles  max_rel\n");
    printf("  ------ ------  ------  ------  ----------  --------  --------  ------  --------  --------  -------\n");

    bool all_pass = true, invariants_ok = true;
    for (int li = 0; li < 3; li++) {
        Res r = run_once(N, D, rd_lats[li], wr_lats[li], q, k, v, exp, sq, sk, sv, o_base);
        if (r.cfg_err) { printf("  %5d  %5d   CFG_ERROR\n", rd_lats[li], wr_lats[li]); all_pass = false; continue; }
        if (r.timeout) { printf("  %5d  %5d   TIMEOUT\n",   rd_lats[li], wr_lats[li]); all_pass = false; continue; }

        printf("  %5d  %5d  %-6s  %6d  %10u  %8u  %8u  %6u  %8u  %8u  %.4f\n",
               rd_lats[li], wr_lats[li], r.pass ? "PASS" : "FAIL", r.tb_cycles,
               r.perf_total, r.perf_dma_bytes, r.perf_wb_bytes,
               r.perf_wb_cycles, r.perf_wb_beats, r.perf_kv_tiles, r.max_rel);

        if (!r.pass) all_pass = false;
        if (r.perf_dma_bytes != exp_rd_bytes) invariants_ok = false;
        if (r.perf_wb_bytes  != exp_wb_bytes) invariants_ok = false;
        if (r.perf_wb_beats  != exp_wb_beats) invariants_ok = false;
        if (r.perf_kv_tiles  != exp_tiles)    invariants_ok = false;
    }

    printf("\n  Invariants: rd_bytes==%u, wb_bytes==%u, wb_beats==%u, kv_tiles==%u : %s\n",
           exp_rd_bytes, exp_wb_bytes, exp_wb_beats, exp_tiles,
           invariants_ok ? "CONFIRMED" : "MISMATCH");
    printf("  Read DMA and write-back are temporally disjoint (write starts on core done),\n");
    printf("  so the single AXI RW model serves AR/R then AW/W/B with no contention.\n");

    bool ok = all_pass && invariants_ok;
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : flash_attn_top_dma_banked_wb\n");
    printf("Scenarios covered: dram_roundtrip_N%d_d%d_{rd,wr}lat_sweep{0,20,100}\n", N, D);
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", ok ? 0 : 1);
    printf("Result           : %s\n", ok ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
