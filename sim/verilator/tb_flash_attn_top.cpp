// tb_flash_attn_top.cpp — End-to-end Verilator testbench for flash_attn_top
//
// Usage:
//   ./sim_top [--N <seq_len>] [--data <data_dir>]
//
// Default: --N 16  --data ../../data
//
// Build (from sim/verilator/):
//   make tb_top          # N=16
//   make tb_top_N64      # N=64
//   make tb_top_N128     # N=128
//   make tb_top_N256     # N=256
//   make tb_top_all      # all four

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "Vflash_attn_top.h"
#include "verilated.h"

static void tick(Vflash_attn_top *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

static void reset(Vflash_attn_top *dut) {
    dut->rst_n     = 0;
    dut->start     = 0;
    dut->q_we_ext  = 0;
    dut->kv_we_ext = 0;
    dut->o_re_ext  = 0;
    dut->scale_q   = 0x0100;
    dut->scale_k   = 0x0100;
    dut->scale_v   = 0x0100;
    tick(dut); tick(dut);
    dut->rst_n = 1;
    tick(dut);
}

static bool load_int8_hex(const char *path, std::vector<int8_t> &out, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("ERROR: cannot open %s\n", path); return false; }
    out.resize(n);
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%X", &v) != 1) {
            printf("ERROR: short read at %s[%d]\n", path, i);
            fclose(f); return false;
        }
        out[i] = (int8_t)(v & 0xFF);
    }
    fclose(f);
    return true;
}

static bool load_int32_hex(const char *path, std::vector<int32_t> &out, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("ERROR: cannot open %s\n", path); return false; }
    out.resize(n);
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%X", &v) != 1) {
            printf("ERROR: short read at %s[%d]\n", path, i);
            fclose(f); return false;
        }
        out[i] = (int32_t)v;
    }
    fclose(f);
    return true;
}

static bool load_scales(const char *path,
                         uint16_t &scale_q, uint16_t &scale_k,
                         uint16_t &scale_v) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("WARN: cannot open %s\n", path); return false; }
    unsigned sq = 0x0100, sk = 0x0100, sv = 0x0100;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        unsigned v;
        if (sscanf(line, "scale_q_q88 = 0x%X", &v) == 1) sq = v;
        if (sscanf(line, "scale_k_q88 = 0x%X", &v) == 1) sk = v;
        if (sscanf(line, "scale_v_q88 = 0x%X", &v) == 1) sv = v;
    }
    fclose(f);
    scale_q = (uint16_t)sq;
    scale_k = (uint16_t)sk;
    scale_v = (uint16_t)sv;
    return true;
}

static void write_q_sram(Vflash_attn_top *dut,
                          const std::vector<int8_t> &data) {
    for (int i = 0; i < (int)data.size(); i++) {
        dut->q_we_ext    = 1;
        dut->q_waddr_ext = (uint16_t)i;
        dut->q_wdata_ext = (uint8_t)data[i];
        tick(dut);
    }
    dut->q_we_ext = 0;
    tick(dut);
}

static void write_kv_sram(Vflash_attn_top *dut,
                           const std::vector<int8_t> &kdata,
                           const std::vector<int8_t> &vdata) {
    for (int i = 0; i < (int)kdata.size(); i++) {
        dut->kv_we_ext   = 1;
        dut->k_waddr_ext = (uint16_t)i;
        dut->k_wdata_ext = (uint8_t)kdata[i];
        dut->v_waddr_ext = (uint16_t)i;
        dut->v_wdata_ext = (uint8_t)vdata[i];
        tick(dut);
    }
    dut->kv_we_ext = 0;
    tick(dut);
}

int main(int argc, char **argv) {
    // ── Parse custom args BEFORE passing to Verilator ────────────────
    int  N        = 16;
    int  D        = 16;
    std::string data_dir = "../../data";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--N") == 0 && i+1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) {
            data_dir = argv[++i];
        }
    }

    // MAX_CYCLE scales with N² (empirical: N=16 finishes at ~917 cycles)
    int max_cycle    = (N * N * 15 > 100000) ? N * N * 15 : 100000;
    int print_every  = (N * N / 256 > 10) ? N * N / 256 : 10;

    std::string q_hex   = data_dir + "/q_input.hex";
    std::string k_hex   = data_dir + "/k_input.hex";
    std::string v_hex   = data_dir + "/v_input.hex";
    std::string exp_hex = data_dir + "/expected.hex";
    std::string scales  = data_dir + "/scales.txt";

    VerilatedContext *ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    Vflash_attn_top *dut = new Vflash_attn_top{ctx};

    std::vector<int8_t>  q_data, k_data, v_data;
    std::vector<int32_t> expected;

    printf("=== flash_attn_top test: N=%d D=%d ===\n", N, D);
    printf("Loading test vectors from %s ...\n", data_dir.c_str());
    if (!load_int8_hex(q_hex.c_str(),   q_data,   N * D)) return 1;
    if (!load_int8_hex(k_hex.c_str(),   k_data,   N * D)) return 1;
    if (!load_int8_hex(v_hex.c_str(),   v_data,   N * D)) return 1;
    if (!load_int32_hex(exp_hex.c_str(), expected, N * D)) return 1;
    printf("  Loaded %d entries each\n", N * D);

    uint16_t scale_q = 0x0100, scale_k = 0x0100, scale_v = 0x0100;
    load_scales(scales.c_str(), scale_q, scale_k, scale_v);
    printf("  scale_q=0x%04X  scale_k=0x%04X  scale_v=0x%04X\n",
           scale_q, scale_k, scale_v);

    reset(dut);
    dut->scale_q = scale_q;
    dut->scale_k = scale_k;
    dut->scale_v = scale_v;

    printf("Writing Q SRAM (%d bytes)...\n", N * D);
    write_q_sram(dut, q_data);

    printf("Writing K/V SRAM (%d bytes each)...\n", N * D);
    write_kv_sram(dut, k_data, v_data);

    printf("Starting DUT... (max_cycle=%d)\n", max_cycle);
    dut->start = 1;
    tick(dut);
    dut->start = 0;

    int  cycles   = 0;
    bool finished = false;

    while (cycles < max_cycle) {
        tick(dut);
        cycles++;
        if (dut->done) {
            finished = true;
            break;
        }
        if (cycles % print_every == 0)
            printf("cycle=%d fsm=%d sfx_valid=%d sfx_out=%d sfx_state=%d pv=%d acc0=%d accum=%d saddr=%d\n",
                   cycles,
                   (int)dut->dbg_state,
                   (int)dut->dbg_sfx_tile_valid,
                   (int)dut->dbg_sfx_out_valid,
                   (int)dut->dbg_softmax_state,
                   (int)dut->dbg_is_pv_phase,
                   (int)dut->dbg_acc0,
                   (int)dut->dbg_accum_en,
                   (int)dut->dbg_sram_addr);
    }

    if (!finished) {
        printf("TIMEOUT after %d cycles\n", max_cycle);
        dut->final(); delete dut; delete ctx;
        return 1;
    }
    printf("  done=1 at cycle %d  acc0=%d is_pv=%d\n",
           cycles, (int)dut->dbg_acc0, (int)dut->dbg_is_pv_phase);

    printf("Reading output buffer (%d entries)...\n", N * D);
    std::vector<int32_t> hw_out(N * D);
    dut->o_re_ext = 1;
    for (int i = 0; i < N * D; i++) {
        dut->o_raddr_ext = (uint16_t)i;
        tick(dut); tick(dut);
        hw_out[i] = dut->o_rdata_ext;
    }
    dut->o_re_ext = 0;

    printf("Comparing...\n");
    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int    fail_count  = 0;

    for (int i = 0; i < N * D; i++) {
        double hw_f  = (double)hw_out[i]  / 256.0;
        double exp_f = (double)expected[i] / 256.0;
        double abs_e = fabs(hw_f - exp_f);
        double ref   = fabs(exp_f);
        double rel_e = (ref > 1e-3) ? abs_e / ref : 0.0;

        if (abs_e > max_abs_err) max_abs_err = abs_e;
        if (rel_e > max_rel_err) max_rel_err = rel_e;

        if (rel_e > 0.05 && ref > 1e-3) {
            if (fail_count < 5)
                printf("  [%3d] hw=%.4f exp=%.4f rel=%.2f%%\n",
                       i, hw_f, exp_f, rel_e * 100.0);
            fail_count++;
        }
    }

    printf("\n=== Results (N=%d) ===\n", N);
    printf("  Max absolute error : %.6f\n", max_abs_err);
    printf("  Max relative error : %.4f%%\n", max_rel_err * 100.0);
    printf("  Entries > 5%%      : %d / %d\n", fail_count, N * D);

    bool pass = (fail_count == 0);
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : flash_attn_top\n");
    printf("Scenarios covered: end_to_end_N%d\n", N);
    printf("Test cases run   : 1\n");
    printf("Mismatches       : %d\n", fail_count);
    printf("Result           : %s\n", pass ? "PASS" : "FAIL");

    dut->final();
    delete dut;
    delete ctx;
    return pass ? 0 : 1;
}
