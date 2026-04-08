// ============================================================
//  tb_flash_attn_top_axi.cpp — End-to-end testbench for flash_attn_top_axi
//
//  Loads Q/K/V from data/N64_axi/, streams via AXI4-Stream slave interface,
//  waits for done, receives output via AXI4-Stream master, then compares
//  each head's output against the per-head expected hex files.
//
//  Usage:
//    ./tb_top_axi_N64 [--N <seq_len>] [--data <data_dir>]
//  Default:
//    --N 64  --data ../../data/N64_axi
// ============================================================

#include "Vflash_attn_top_axi.h"
#include "verilated.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static Vflash_attn_top_axi* dut;
static uint64_t sim_time  = 0;
static int      max_cycle = 5000000;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    sim_time++;
    if ((int)sim_time > max_cycle) {
        printf("TIMEOUT after %llu cycles\n", (unsigned long long)sim_time);
        exit(1);
    }
}

static void reset() {
    dut->rst_n         = 0;
    dut->s_axis_tvalid = 0;
    dut->s_axis_tdata  = 0;
    dut->s_axis_tlast  = 0;
    dut->m_axis_tready = 0;
    dut->scale_q       = 0x0100;
    dut->scale_k       = 0x0100;
    dut->scale_v       = 0x0100;
    for (int i = 0; i < 5; i++) tick();
    dut->rst_n = 1;
    tick();
}

// ── File loading helpers ───────────────────────────────────────────────

static bool load_int8_hex(const std::string& path, std::vector<uint8_t>& out, int n) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { printf("ERROR: cannot open %s\n", path.c_str()); return false; }
    out.resize(n);
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%X", &v) != 1) {
            printf("ERROR: short read at %s[%d]\n", path.c_str(), i);
            fclose(f); return false;
        }
        out[i] = (uint8_t)(v & 0xFF);
    }
    fclose(f);
    return true;
}

static bool load_int32_hex(const std::string& path, std::vector<int32_t>& out, int n) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { printf("WARN: cannot open %s\n", path.c_str()); return false; }
    out.resize(n);
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, "%X", &v) != 1) {
            printf("ERROR: short read at %s[%d]\n", path.c_str(), i);
            fclose(f); return false;
        }
        out[i] = (int32_t)v;
    }
    fclose(f);
    return true;
}

static bool load_scales(const std::string& path,
                         uint16_t& scale_q, uint16_t& scale_k, uint16_t& scale_v) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { printf("WARN: cannot open scales %s\n", path.c_str()); return false; }
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

// ── AXI stream helper ─────────────────────────────────────────────────

// Stream one matrix (N×HD bytes), asserting tlast on the last byte.
// Waits for tready before advancing.
static void stream_matrix(const std::vector<uint8_t>& mat) {
    for (int i = 0; i < (int)mat.size(); i++) {
        dut->s_axis_tdata  = mat[i];
        dut->s_axis_tvalid = 1;
        dut->s_axis_tlast  = (i == (int)mat.size() - 1) ? 1 : 0;
        dut->eval();
        // Wait for tready
        int wait = 0;
        while (!dut->s_axis_tready) {
            tick();
            dut->s_axis_tvalid = 1;
            dut->s_axis_tdata  = mat[i];
            dut->s_axis_tlast  = (i == (int)mat.size() - 1) ? 1 : 0;
            dut->eval();
            if (++wait > 1000) { printf("TIMEOUT waiting for tready\n"); exit(1); }
        }
        tick();
    }
    dut->s_axis_tvalid = 0;
    dut->s_axis_tlast  = 0;
}

// ── Main ──────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int         N        = 64;
    int         HD       = 64;   // total head dimension
    int         HEADS    = 4;
    int         PHD      = 16;   // per-head dimension
    std::string data_dir = "../../data/N64_axi";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N")    && i+1 < argc) N        = atoi(argv[++i]);
        if (!strcmp(argv[i], "--data") && i+1 < argc) data_dir = argv[++i];
    }

    max_cycle = N * N * 80 > 500000 ? N * N * 80 : 500000;

    VerilatedContext* ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);
    dut = new Vflash_attn_top_axi{ctx};

    printf("=== flash_attn_top_axi test: N=%d HD=%d heads=%d ===\n", N, HD, HEADS);
    printf("Data dir: %s\n", data_dir.c_str());

    // ── Load test vectors ────────────────────────────────────────────
    std::vector<uint8_t> Q_bytes, K_bytes, V_bytes;
    int total = N * HD;

    if (!load_int8_hex(data_dir + "/q_input.hex", Q_bytes, total)) return 1;
    if (!load_int8_hex(data_dir + "/k_input.hex", K_bytes, total)) return 1;
    if (!load_int8_hex(data_dir + "/v_input.hex", V_bytes, total)) return 1;
    printf("Loaded %d bytes per matrix\n", total);

    uint16_t scale_q = 0x0100, scale_k = 0x0100, scale_v = 0x0100;
    load_scales(data_dir + "/scales.txt", scale_q, scale_k, scale_v);
    printf("scale_q=0x%04X  scale_k=0x%04X  scale_v=0x%04X\n",
           scale_q, scale_k, scale_v);

    // ── Reset and configure ─────────────────────────────────────────
    reset();
    dut->scale_q = (int16_t)scale_q;
    dut->scale_k = (int16_t)scale_k;
    dut->scale_v = (int16_t)scale_v;

    // ── Stream Q, K, V ──────────────────────────────────────────────
    printf("Streaming Q (%d bytes)...\n", total);
    stream_matrix(Q_bytes);

    printf("Streaming K (%d bytes)...\n", total);
    stream_matrix(K_bytes);

    printf("Streaming V (%d bytes)...\n", total);
    stream_matrix(V_bytes);

    // ── Wait for done ────────────────────────────────────────────────
    printf("Waiting for done...\n");
    while (!dut->done) tick();
    printf("  done=1 at cycle %llu\n", (unsigned long long)sim_time);

    // ── Receive output via AXI master ────────────────────────────────
    int expected_beats = N * HEADS * PHD;
    std::vector<int32_t> rtl_out;
    rtl_out.reserve(expected_beats);

    dut->m_axis_tready = 1;
    printf("Receiving %d output beats...\n", expected_beats);

    while ((int)rtl_out.size() < expected_beats) {
        dut->eval();
        if (dut->m_axis_tvalid && dut->m_axis_tready) {
            rtl_out.push_back((int32_t)dut->m_axis_tdata);
        }
        tick();
    }
    printf("  Received %d beats\n", (int)rtl_out.size());

    // ── Compare with per-head expected files ─────────────────────────
    // RTL output layout: [N rows × (HEADS × PHD) cols], row-major
    //   rtl_out[row*(HEADS*PHD) + head*PHD + col]
    bool all_pass = true;

    for (int h = 0; h < HEADS; h++) {
        std::string exp_path = data_dir + "/expected_h" +
                               std::to_string(h) + ".hex";
        std::vector<int32_t> expected;
        if (!load_int32_hex(exp_path, expected, N * PHD)) {
            printf("  Head %d: expected file not found, skipping\n", h);
            continue;
        }

        int max_err = 0, total_err = 0, errors = 0;
        for (int i = 0; i < N * PHD; i++) {
            int row = i / PHD;
            int col = i % PHD;
            int rtl_idx = row * (HEADS * PHD) + h * PHD + col;
            int32_t rv = rtl_out[rtl_idx];
            int32_t ev = expected[i];
            int err = abs(rv - ev);
            if (err > max_err) max_err = err;
            total_err += err;
            if (err > 0) errors++;
        }
        bool head_pass = (max_err == 0);
        printf("  Head %d: max_err=%d  total_err=%d  wrong_entries=%d  %s\n",
               h, max_err, total_err, errors,
               head_pass ? "PASS" : "FAIL");
        if (!head_pass) all_pass = false;
    }

    printf("\ndone at cycle %llu  %s\n",
           (unsigned long long)sim_time,
           all_pass ? "ALL PASS" : "SOME FAIL");

    delete dut;
    delete ctx;
    return all_pass ? 0 : 1;
}
