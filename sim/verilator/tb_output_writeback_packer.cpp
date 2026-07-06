// ============================================================
//  tb_output_writeback_packer.cpp — Stage WB-2 microbenchmark.
//
//  Preloads deterministic 32-bit words into an output-buffer-like SRAM, drains
//  them through output_writeback_packer → dma_write_engine → axi_mem_model_rw,
//  and checks the DRAM image byte-for-byte against the expected word order.
//
//  Sizes exercised: N=64 d=16 (1024 words / 4096 B) and N=64 d=64 (4096 words /
//  16384 B), at a non-zero output base, swept over wr_latency {0,20,100}.
//  Also verifies: guard bytes untouched, bytes written = num_words*4, `done`
//  fires once, and that the packer holds a beat under write-engine back-pressure.
// ============================================================
#include "Vtb_output_writeback_packer.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vtb_output_writeback_packer* dut = nullptr;

// Observations for the current run
static long g_beats;        // AXI W beats accepted
static long g_bp_cycles;    // cycles src_valid & !src_ready (packer holding)
static int  g_pkr_done;
static int  g_dma_done;

static void tick() {
    dut->clk = 0; dut->eval();
    // sample pre-edge (registered state stable)
    bool w_fire = dut->dbg_w_fire;
    bool src_v  = dut->dbg_src_valid;
    bool src_r  = dut->dbg_src_ready;
    bool pkr_dn = dut->pkr_done;
    bool dma_dn = dut->dma_done;
    main_time++;
    dut->clk = 1; dut->eval(); main_time++;
    if (w_fire) g_beats++;
    if (src_v && !src_r) g_bp_cycles++;
    if (pkr_dn) g_pkr_done++;
    if (dma_dn) g_dma_done++;
}

static void reset() {
    dut->rst_n = 0; dut->wr_latency = 0;
    dut->obuf_we = 0; dut->obuf_waddr = 0; dut->obuf_wdata = 0;
    dut->start = 0; dut->num_words = 0;
    dut->desc_valid = 0; dut->desc_addr = 0; dut->desc_len_bytes = 0;
    dut->init_we = 0; dut->init_addr = 0; dut->init_data = 0; dut->bd_raddr = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static void obuf_write(uint32_t addr, uint32_t data) {
    dut->obuf_we = 1; dut->obuf_waddr = addr; dut->obuf_wdata = data; tick();
    dut->obuf_we = 0;
}

static void dram_write(uint32_t addr, uint8_t data) {
    dut->init_we = 1; dut->init_addr = addr; dut->init_data = data; tick();
    dut->init_we = 0;
}

static uint8_t bd_read(uint32_t addr) {
    dut->bd_raddr = addr; dut->eval();
    return (uint8_t)dut->bd_rdata;
}

// Deterministic expected DRAM byte at global offset g within the output region.
static inline uint8_t exp_byte(uint32_t g, uint32_t base) {
    return (uint8_t)((g * 7 + 3 + (base & 0xFF)) & 0xFF);
}

static int g_errors = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("    FAIL: "); printf(__VA_ARGS__); printf("\n"); g_errors++; } } while (0)

static void run_case(const char* name, uint32_t num_words, uint32_t base, uint16_t wr_lat) {
    const uint32_t len = num_words * 4;

    // 1. Preload the source SRAM: word i packs bytes exp_byte(i*4+b).
    for (uint32_t i = 0; i < num_words; i++) {
        uint32_t w = 0;
        for (int b = 0; b < 4; b++) w |= (uint32_t)exp_byte(i * 4 + b, base) << (8 * b);
        obuf_write(i, w);
    }

    // 2. Sentinel-preload the DRAM window + 16-byte guards with 0xAA.
    uint32_t g0 = (base >= 16) ? base - 16 : 0;
    uint32_t g1 = base + len + 16;
    for (uint32_t a = g0; a < g1; a++) dram_write(a, 0xAA);

    // 3. Reset observation counters, then issue descriptor + start the packer.
    g_beats = 0; g_bp_cycles = 0; g_pkr_done = 0; g_dma_done = 0;
    dut->wr_latency = wr_lat;
    dut->desc_addr = base; dut->desc_len_bytes = len;
    dut->num_words = (uint16_t)num_words;
    dut->desc_valid = 1; dut->start = 1;
    int guard = 0;
    while (true) {
        dut->clk = 0; dut->eval();
        bool ready = dut->desc_ready;
        main_time++;
        dut->clk = 1; dut->eval(); main_time++;   // accept desc + latch start here
        if (ready || ++guard > 50) break;
    }
    dut->desc_valid = 0; dut->start = 0;

    // 4. Run until the write engine's B for the final burst, plus a few extra.
    int cyc = 0, extra = 0;
    while (cyc < 2000000) {
        tick(); cyc++;
        if (g_dma_done > 0) { if (++extra >= 5) break; }
    }

    // 5a. byte-exact placement of the whole output region
    bool bytes_ok = true;
    for (uint32_t g = 0; g < len; g++) {
        uint8_t got = bd_read(base + g), exp = exp_byte(g, base);
        if (got != exp) {
            CHECK(false, "%s wr_lat=%u: byte@+%u got %02x exp %02x", name, wr_lat, g, got, exp);
            bytes_ok = false; break;
        }
    }
    // 5b. guards untouched
    for (uint32_t a = g0; a < base; a++)
        CHECK(bd_read(a) == 0xAA, "%s wr_lat=%u: front guard@%u overwritten", name, wr_lat, a);
    for (uint32_t a = base + len; a < g1; a++)
        CHECK(bd_read(a) == 0xAA, "%s wr_lat=%u: back guard@%u overwritten", name, wr_lat, a);

    // 5c. bytes written = num_words*4  (beats * BYTES_PER_BEAT)
    long exp_beats = (long)num_words / 2;
    CHECK(g_beats == exp_beats, "%s wr_lat=%u: W beats got %ld exp %ld", name, wr_lat, g_beats, exp_beats);
    CHECK(g_beats * 8 == (long)len, "%s wr_lat=%u: bytes got %ld exp %u", name, wr_lat, g_beats * 8, len);

    // 5d. done fires exactly once (packer and write engine)
    CHECK(g_pkr_done == 1, "%s wr_lat=%u: pkr_done pulsed %d (exp 1)", name, wr_lat, g_pkr_done);
    CHECK(g_dma_done == 1, "%s wr_lat=%u: dma_done pulsed %d (exp 1)", name, wr_lat, g_dma_done);

    // 5e. back-pressure was exercised (packer held a beat while engine not ready).
    //     Guaranteed once write latency stretches the between-burst B wait.
    if (wr_lat > 0)
        CHECK(g_bp_cycles > 0, "%s wr_lat=%u: no back-pressure observed (bp=%ld)", name, wr_lat, g_bp_cycles);

    if (bytes_ok)
        printf("  [%-9s words=%4u base=0x%05x len=%5u beats=%4ld wr_lat=%3u] %s  (%d cyc, bp=%ld)\n",
               name, num_words, base, len, g_beats, wr_lat,
               (g_errors == 0 ? "PASS" : "has-fail"), cyc, g_bp_cycles);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_output_writeback_packer;
    reset();

    struct Case { const char* name; uint32_t words; uint32_t base; };
    Case cases[] = {
        {"d16_N64", 1024, 0x04000},   // N=64 d=16 → 1024 words = 4096 B
        {"d64_N64", 4096, 0x08000},   // N=64 d=64 → 4096 words = 16384 B
    };
    const uint16_t lats[] = {0, 20, 100};

    printf("=== output_writeback_packer + dma_write_engine + axi_mem_model_rw (WB-2) ===\n");
    for (auto& c : cases)
        for (uint16_t lat : lats)
            run_case(c.name, c.words, c.base, lat);

    int ncases = (int)(sizeof(cases)/sizeof(cases[0])) * 3;
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : output_writeback_packer (WB-2 drain datapath)\n");
    printf("Scenarios covered: d16_N64/d64_N64 x wr_latency{0,20,100}, non-zero base\n");
    printf("Test cases run   : %d\n", ncases);
    printf("Mismatches       : %d\n", g_errors);
    printf("Result           : %s\n", g_errors == 0 ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", g_errors == 0 ? "PASS" : "FAIL");

    delete dut;
    return g_errors ? 1 : 0;
}
