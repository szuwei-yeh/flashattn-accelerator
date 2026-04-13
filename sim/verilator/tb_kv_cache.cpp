// tb_kv_cache.cpp — Unit testbench for kv_cache.sv
//
// Tests:
//   1. Write 5 tokens, read back, verify K and V match exactly
//   2. Write MAX_SEQ_LEN tokens, verify cache_len reaches MAX_SEQ_LEN
//   3. rst_n pulse resets cache_len to 0

#include "Vkv_cache.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

static const int HEAD_DIM    = 16;
static const int MAX_SEQ_LEN = 256;
static const int N_WORDS     = (HEAD_DIM * 8 + 31) / 32;  // = 4 for 128-bit

// ---- clock tick ------------------------------------------------
static void tick(Vkv_cache *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

// ---- reset helper ----------------------------------------------
static void do_reset(Vkv_cache *dut) {
    dut->rst_n     = 0;
    dut->write_en  = 0;
    dut->write_ptr = 0;
    dut->read_addr = 0;
    for (int w = 0; w < N_WORDS; w++) {
        dut->k_new_flat[w] = 0;
        dut->v_new_flat[w] = 0;
    }
    tick(dut); tick(dut);
    dut->rst_n = 1;
    tick(dut);
}

// ---- pack HEAD_DIM bytes into a flat wide signal ---------------
static void pack_vec(WData* dst, const uint8_t* src) {
    for (int w = 0; w < N_WORDS; w++) dst[w] = 0;
    for (int j = 0; j < HEAD_DIM; j++)
        dst[j / 4] |= ((WData)src[j]) << ((j % 4) * 8);
}

// ---- check HEAD_DIM bytes against a flat wide signal -----------
static bool check_vec(const WData* got, const uint8_t* expected, int token) {
    bool ok = true;
    for (int j = 0; j < HEAD_DIM; j++) {
        uint8_t g = (uint8_t)((got[j / 4] >> ((j % 4) * 8)) & 0xFF);
        if (g != expected[j]) {
            printf("  token[%d] byte[%d]: got 0x%02X expected 0x%02X\n",
                   token, j, g, expected[j]);
            ok = false;
        }
    }
    return ok;
}

// ================================================================
int main(int argc, char **argv) {
    VerilatedContext *ctx = new VerilatedContext;
    Vkv_cache *dut = new Vkv_cache{ctx};

    int failures = 0;

    // ==============================================================
    // Test 1: Write 5 tokens, read back, verify K and V
    // ==============================================================
    do_reset(dut);

    uint8_t k_data[5][HEAD_DIM], v_data[5][HEAD_DIM];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            k_data[i][j] = (uint8_t)((i * HEAD_DIM + j + 1) & 0xFF);
            v_data[i][j] = (uint8_t)((i * HEAD_DIM + j + 100) & 0xFF);
        }
    }

    // Write 5 tokens sequentially
    for (int i = 0; i < 5; i++) {
        dut->write_en  = 1;
        dut->write_ptr = (uint8_t)i;
        pack_vec(&dut->k_new_flat[0], k_data[i]);
        pack_vec(&dut->v_new_flat[0], v_data[i]);
        tick(dut);
    }
    dut->write_en = 0;

    // Check cache_len == 5
    bool pass1_len = (dut->cache_len == 5);
    printf("Test 1a cache_len after 5 writes: got %u expected 5  %s\n",
           (unsigned)dut->cache_len, pass1_len ? "PASS" : "FAIL");
    if (!pass1_len) failures++;

    // Read back each token (SRAM has 1-cycle latency: tick then check)
    bool pass1_data = true;
    for (int i = 0; i < 5; i++) {
        dut->read_addr = (uint8_t)i;
        tick(dut);  // rdata latches on this posedge
        bool k_ok = check_vec(&dut->k_out_flat[0], k_data[i], i);
        bool v_ok = check_vec(&dut->v_out_flat[0], v_data[i], i);
        if (!k_ok || !v_ok) {
            printf("  Token %d readback: K=%s V=%s\n",
                   i, k_ok ? "OK" : "FAIL", v_ok ? "OK" : "FAIL");
            pass1_data = false;
        }
    }
    printf("Test 1b (write 5 tokens, read back): %s\n",
           pass1_data ? "PASS" : "FAIL");
    if (!pass1_data) failures++;

    // ==============================================================
    // Test 2: Write MAX_SEQ_LEN tokens, verify cache_len at top
    // ==============================================================
    do_reset(dut);

    for (int i = 0; i < MAX_SEQ_LEN; i++) {
        uint8_t k_vec[HEAD_DIM], v_vec[HEAD_DIM];
        for (int j = 0; j < HEAD_DIM; j++) {
            k_vec[j] = (uint8_t)((i + j) & 0xFF);
            v_vec[j] = (uint8_t)((i + j + 128) & 0xFF);
        }
        dut->write_en  = 1;
        dut->write_ptr = (uint8_t)i;
        pack_vec(&dut->k_new_flat[0], k_vec);
        pack_vec(&dut->v_new_flat[0], v_vec);
        tick(dut);
    }
    dut->write_en = 0;

    bool pass2 = (dut->cache_len == (uint32_t)MAX_SEQ_LEN);
    printf("Test 2  (fill MAX_SEQ_LEN=%d): cache_len=%u  %s\n",
           MAX_SEQ_LEN, (unsigned)dut->cache_len, pass2 ? "PASS" : "FAIL");
    if (!pass2) failures++;

    // Extra: one more write beyond MAX_SEQ_LEN must NOT increment cache_len
    {
        uint8_t dummy[HEAD_DIM] = {};
        dut->write_en  = 1;
        dut->write_ptr = 0;   // any address
        pack_vec(&dut->k_new_flat[0], dummy);
        pack_vec(&dut->v_new_flat[0], dummy);
        tick(dut);
        dut->write_en = 0;
        bool pass2b = (dut->cache_len == (uint32_t)MAX_SEQ_LEN);
        printf("Test 2b (no overflow beyond MAX_SEQ_LEN): cache_len=%u  %s\n",
               (unsigned)dut->cache_len, pass2b ? "PASS" : "FAIL");
        if (!pass2b) failures++;
    }

    // ==============================================================
    // Test 3: rst_n resets cache_len to 0
    // ==============================================================
    // cache_len is currently MAX_SEQ_LEN
    do_reset(dut);

    bool pass3 = (dut->cache_len == 0);
    printf("Test 3  (rst_n clears cache_len): cache_len=%u  %s\n",
           (unsigned)dut->cache_len, pass3 ? "PASS" : "FAIL");
    if (!pass3) failures++;

    // ---- summary ------------------------------------------------
    printf("\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : kv_cache\n");
    printf("Scenarios covered: write_read_5tokens, fill_max_seq_len, no_overflow, rst_n_reset\n");
    printf("Test cases run   : 5\n");
    printf("Mismatches       : %d\n", failures);
    printf("Result           : %s\n", failures == 0 ? "PASS" : "FAIL");

    dut->final();
    delete dut;
    delete ctx;
    return failures == 0 ? 0 : 1;
}
