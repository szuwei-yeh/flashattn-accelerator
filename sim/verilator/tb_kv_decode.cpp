// tb_kv_decode.cpp — End-to-end testbench for KV Cache decode mode (Week 6)
//
// DUT: flash_attn_core compiled with SEQ_LEN=32, HEAD_DIM=16
//
// Three test phases:
//   Phase 1 — Prefill (mode=0, N=32):
//     Load Q/K/V flat SRAMs, start with mode=0, verify all 32×16 output words.
//
//   Phase 2 — Decode (mode=1, kv_len=32):
//     Fresh DUT. Load K/V from prefill data, load Q_dec at row 0 only.
//     Write 32 K/V tokens into kv_cache. Start with mode=1, kv_len=32.
//     Verify output row 0 (positions 0..15) against expected_decode.hex.
//
//   Phase 3 — KV cache sanity:
//     Check kc_cache_len==32 (set during Phase 2) and read back 2 tokens.
//
// Build (from sim/verilator/):
//   make tb_kv_decode

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "Vflash_attn_core.h"
#include "verilated.h"

static const int N_PREFILL = 32;
static const int HEAD_DIM  = 16;
static const int N_WORDS   = (HEAD_DIM * 8 + 31) / 32;   // 4 for 128-bit

// ------------------------------------------------------------------ helpers

static void tick(Vflash_attn_core *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

static void do_reset(Vflash_attn_core *dut) {
    dut->rst_n        = 0;
    dut->start        = 0;
    dut->mode         = 0;
    dut->kv_len       = 0;
    dut->q_we         = 0;
    dut->k_we         = 0;
    dut->v_we         = 0;
    dut->q_waddr      = 0;
    dut->k_waddr      = 0;
    dut->v_waddr      = 0;
    dut->q_wdata      = 0;
    dut->k_wdata      = 0;
    dut->v_wdata      = 0;
    dut->out_raddr    = 0;
    dut->scale_q      = 0x0100;
    dut->scale_k      = 0x0100;
    dut->scale_v      = 0x0100;
    dut->kc_write_en  = 0;
    dut->kc_write_ptr = 0;
    dut->kc_read_addr = 0;
    for (int w = 0; w < N_WORDS; w++) {
        dut->kc_k_flat[w] = 0;
        dut->kc_v_flat[w] = 0;
    }
    tick(dut); tick(dut);
    dut->rst_n = 1;
    tick(dut);
}

// ---- file loaders -----------------------------------------------

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

// ---- SRAM write helpers -----------------------------------------

static void write_q_sram(Vflash_attn_core *dut,
                          const std::vector<int8_t> &data, int n_bytes) {
    for (int i = 0; i < n_bytes; i++) {
        dut->q_we    = 1;
        dut->q_waddr = (uint16_t)i;
        dut->q_wdata = (uint8_t)data[i];
        tick(dut);
    }
    dut->q_we = 0;
    tick(dut);
}

static void write_kv_sram(Vflash_attn_core *dut,
                           const std::vector<int8_t> &kdata,
                           const std::vector<int8_t> &vdata, int n_bytes) {
    for (int i = 0; i < n_bytes; i++) {
        dut->k_we    = 1;
        dut->k_waddr = (uint16_t)i;
        dut->k_wdata = (uint8_t)kdata[i];
        dut->v_we    = 1;
        dut->v_waddr = (uint16_t)i;
        dut->v_wdata = (uint8_t)vdata[i];
        tick(dut);
    }
    dut->k_we = 0;
    dut->v_we = 0;
    tick(dut);
}

// ---- Pack HEAD_DIM INT8 bytes into a 128-bit flat WData word ----

static void pack_kv_flat(WData *dst, const int8_t *src) {
    for (int w = 0; w < N_WORDS; w++) dst[w] = 0;
    for (int j = 0; j < HEAD_DIM; j++)
        dst[j / 4] |= ((WData)((uint8_t)src[j])) << ((j % 4) * 8);
}

// ---- Write N tokens to kv_cache --------------------------------

static void populate_kv_cache(Vflash_attn_core *dut,
                                const std::vector<int8_t> &kdata,
                                const std::vector<int8_t> &vdata,
                                int n_tokens) {
    for (int t = 0; t < n_tokens; t++) {
        dut->kc_write_en  = 1;
        dut->kc_write_ptr = (uint8_t)t;
        pack_kv_flat(&dut->kc_k_flat[0], kdata.data() + t * HEAD_DIM);
        pack_kv_flat(&dut->kc_v_flat[0], vdata.data() + t * HEAD_DIM);
        tick(dut);
    }
    dut->kc_write_en = 0;
    tick(dut);
}

// ---- Run DUT and wait for done ----------------------------------

static bool run_and_wait(Vflash_attn_core *dut, int max_cycles, const char *tag) {
    dut->start = 1; tick(dut); dut->start = 0;
    for (int c = 0; c < max_cycles; c++) {
        tick(dut);
        if (dut->done) {
            printf("  [%s] done at cycle %d\n", tag, c + 1);
            return true;
        }
    }
    printf("  [%s] TIMEOUT after %d cycles\n", tag, max_cycles);
    return false;
}

// ---- Read output buffer row-major ------------------------------

static void read_output(Vflash_attn_core *dut,
                         std::vector<int32_t> &hw_out, int n_entries) {
    hw_out.resize(n_entries);
    for (int i = 0; i < n_entries; i++) {
        dut->out_raddr = (uint16_t)i;
        tick(dut); tick(dut);   // 1-cycle SRAM latency + 1 slack
        hw_out[i] = (int32_t)dut->out_rdata;
    }
}

// ---- Compare and report -----------------------------------------

static int compare(const std::vector<int32_t> &hw_out,
                   const std::vector<int32_t> &expected,
                   int n, const char *tag) {
    int failures = 0;
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double hw_f  = (double)hw_out[i]  / 256.0;
        double ex_f  = (double)expected[i] / 256.0;
        double abs_e = fabs(hw_f - ex_f);
        if (abs_e > max_abs) max_abs = abs_e;
        if (abs_e > 0.5 && fabs(ex_f) > 1e-3) {   // allow ±0.5 tolerance
            if (failures < 5)
                printf("  [%s] idx=%d hw=%.3f exp=%.3f diff=%.3f\n",
                       tag, i, hw_f, ex_f, abs_e);
            failures++;
        }
    }
    printf("  [%s] max_abs_err=%.4f  failures=%d/%d  %s\n",
           tag, max_abs, failures, n, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

// ================================================================
//  main
// ================================================================

int main(int argc, char **argv) {
    std::string data_dir = "../../data/kv_decode";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            data_dir = argv[++i];
    }

    // ---- Load test vectors from disk ----------------------------
    std::vector<int8_t>  q_pf, k_pf, v_pf, q_dec;
    std::vector<int32_t> exp_pf, exp_dec;
    uint16_t scale_q, scale_k, scale_v;

    auto p = [&](const char *f){ return data_dir + "/" + f; };

    printf("=== tb_kv_decode: N=%d  HEAD_DIM=%d ===\n", N_PREFILL, HEAD_DIM);
    printf("Loading from %s ...\n", data_dir.c_str());

    if (!load_int8_hex( p("q_prefill.hex").c_str(), q_pf,  N_PREFILL * HEAD_DIM)) return 1;
    if (!load_int8_hex( p("k_prefill.hex").c_str(), k_pf,  N_PREFILL * HEAD_DIM)) return 1;
    if (!load_int8_hex( p("v_prefill.hex").c_str(), v_pf,  N_PREFILL * HEAD_DIM)) return 1;
    if (!load_int8_hex( p("q_decode.hex").c_str(),  q_dec, HEAD_DIM))             return 1;
    if (!load_int32_hex(p("expected_prefill.hex").c_str(), exp_pf, N_PREFILL * HEAD_DIM)) return 1;
    if (!load_int32_hex(p("expected_decode.hex").c_str(),  exp_dec, HEAD_DIM))             return 1;
    if (!load_scales(   p("scales.txt").c_str(), scale_q, scale_k, scale_v))      return 1;

    printf("  sq=0x%04X  sk=0x%04X  sv=0x%04X\n", scale_q, scale_k, scale_v);

    int total_failures = 0;

    // ==============================================================
    // Phase 1: Prefill (mode=0, N=32)
    // ==============================================================
    printf("\n--- Phase 1: Prefill (mode=0, N=%d) ---\n", N_PREFILL);
    {
        VerilatedContext *ctx = new VerilatedContext;
        Vflash_attn_core *dut = new Vflash_attn_core{ctx};

        do_reset(dut);
        dut->scale_q = scale_q;
        dut->scale_k = scale_k;
        dut->scale_v = scale_v;
        dut->mode    = 0;
        dut->kv_len  = 0;

        write_q_sram( dut, q_pf, N_PREFILL * HEAD_DIM);
        write_kv_sram(dut, k_pf, v_pf, N_PREFILL * HEAD_DIM);

        if (!run_and_wait(dut, 300000, "prefill")) {
            total_failures++;
        } else {
            std::vector<int32_t> hw_out;
            read_output(dut, hw_out, N_PREFILL * HEAD_DIM);
            total_failures += compare(hw_out, exp_pf, N_PREFILL * HEAD_DIM, "prefill");
        }

        dut->final(); delete dut; delete ctx;
    }

    // ==============================================================
    // Phase 2: Decode (mode=1, kv_len=32)
    //   - Fresh DUT (output_buffer starts at 0)
    //   - K/V flat SRAMs loaded with prefill K/V
    //   - Q flat SRAM: only row 0 (Q_dec), positions 16..255 are 0
    //   - 32 tokens written to kv_cache
    //   - Check output positions 0..15 (row 0)
    // ==============================================================
    printf("\n--- Phase 2: Decode (mode=1, kv_len=%d) ---\n", N_PREFILL);
    {
        VerilatedContext *ctx = new VerilatedContext;
        Vflash_attn_core *dut = new Vflash_attn_core{ctx};

        do_reset(dut);
        dut->scale_q = scale_q;
        dut->scale_k = scale_k;
        dut->scale_v = scale_v;
        dut->mode    = 1;
        dut->kv_len  = (uint16_t)N_PREFILL;

        // K/V flat SRAMs: full prefill content
        write_kv_sram(dut, k_pf, v_pf, N_PREFILL * HEAD_DIM);

        // Q flat SRAM: only the 16 bytes of q_dec at row 0 (addresses 0..15)
        write_q_sram(dut, q_dec, HEAD_DIM);   // writes positions 0..HEAD_DIM-1

        // Populate kv_cache with prefill K/V
        populate_kv_cache(dut, k_pf, v_pf, N_PREFILL);

        // Check kc_cache_len
        bool kc_ok = ((unsigned)dut->kc_cache_len == (unsigned)N_PREFILL);
        printf("  kc_cache_len = %u  expected %d  %s\n",
               (unsigned)dut->kc_cache_len, N_PREFILL, kc_ok ? "PASS" : "FAIL");
        if (!kc_ok) total_failures++;

        if (!run_and_wait(dut, 200000, "decode")) {
            total_failures++;
        } else {
            // Read only row 0 output (positions 0..HEAD_DIM-1)
            std::vector<int32_t> hw_out;
            read_output(dut, hw_out, HEAD_DIM);
            total_failures += compare(hw_out, exp_dec, HEAD_DIM, "decode row0");
        }

        dut->final(); delete dut; delete ctx;
    }

    // ==============================================================
    // Phase 3: KV cache read-back sanity
    //   - Fresh DUT, write 4 tokens, read them back via kc_read_addr
    // ==============================================================
    printf("\n--- Phase 3: kv_cache read-back sanity ---\n");
    {
        VerilatedContext *ctx = new VerilatedContext;
        Vflash_attn_core *dut = new Vflash_attn_core{ctx};
        do_reset(dut);

        // Write 4 known tokens
        int8_t k_tok[4][HEAD_DIM], v_tok[4][HEAD_DIM];
        for (int t = 0; t < 4; t++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                k_tok[t][j] = (int8_t)((t * HEAD_DIM + j + 1) & 0x7F);
                v_tok[t][j] = (int8_t)((t * HEAD_DIM + j + 64) & 0x7F);
            }
        }
        for (int t = 0; t < 4; t++) {
            dut->kc_write_en  = 1;
            dut->kc_write_ptr = (uint8_t)t;
            pack_kv_flat(&dut->kc_k_flat[0], k_tok[t]);
            pack_kv_flat(&dut->kc_v_flat[0], v_tok[t]);
            tick(dut);
        }
        dut->kc_write_en = 0;

        bool len_ok = ((unsigned)dut->kc_cache_len == 4u);
        printf("  cache_len=4: %s\n", len_ok ? "PASS" : "FAIL");
        if (!len_ok) total_failures++;

        // Read token 2 back (1-cycle latency)
        dut->kc_read_addr = 2;
        tick(dut);
        bool rb_ok = true;
        for (int j = 0; j < HEAD_DIM; j++) {
            uint8_t got = (uint8_t)((dut->kc_k_out[j / 4] >> ((j % 4) * 8)) & 0xFF);
            if (got != (uint8_t)k_tok[2][j]) {
                printf("  k_tok[2][%d]: got 0x%02X exp 0x%02X\n",
                       j, got, (uint8_t)k_tok[2][j]);
                rb_ok = false;
            }
        }
        printf("  readback kc_k_out[token=2]: %s\n", rb_ok ? "PASS" : "FAIL");
        if (!rb_ok) total_failures++;

        dut->final(); delete dut; delete ctx;
    }

    // ==============================================================
    printf("\n=== RESULT: %s ===\n", total_failures == 0 ? "PASS" : "FAIL");

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : flash_attn_core\n");
    printf("Scenarios covered: prefill_N32, decode_mode, kvcache_readback\n");
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", total_failures);
    printf("Result           : %s\n", total_failures == 0 ? "PASS" : "FAIL");

    return total_failures == 0 ? 0 : 1;
}
