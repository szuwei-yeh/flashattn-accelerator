// ============================================================
//  tb_dma_write_unit.cpp — Unit test for dma_write_engine + axi_mem_model_rw.
//
//  Drives a deterministic source-stream pattern through the AXI4 write master
//  into the behavioral DRAM model, then backdoor-reads the store and checks:
//    - every byte landed byte-exact at the descriptor's DRAM base;
//    - sentinel guard bytes just outside the window are untouched;
//    - WLAST fired exactly on the last beat of each <= MAX_BURST burst;
//    - the AW burst count matches ceil(beats / MAX_BURST);
//    - `done` pulses exactly once.
//  Cases: (a) short single-burst, (b) multi-burst > MAX_BURST at a non-zero
//  base, (c) tiny 2-beat — each swept over wr_latency {0, 20, 100}.
// ============================================================
#include "Vtb_dma_write_unit.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vtb_dma_write_unit* dut = nullptr;

static const int MAX_BURST = 16;
static const int BYTES_PER_BEAT = 8;

// Source stream state + captured observations for the current descriptor
static std::vector<uint64_t> g_src;
static size_t g_src_idx;
static int    g_done_pulses;
static int    g_beat_count;
static int    g_aw_count;
static std::vector<int> g_wlast_at;   // 0-based beat indices where WLAST fired

static void tick() {
    bool have = g_src_idx < g_src.size();
    dut->src_valid = have ? 1 : 0;
    dut->src_data  = have ? g_src[g_src_idx] : 0;
    dut->clk = 0; dut->eval();
    // sample pre-edge (registered state stable; combinational outputs settled)
    bool consume = dut->src_valid && dut->src_ready;
    bool w_fire  = dut->dbg_w_fire;
    bool w_last  = dut->dbg_w_last;
    bool aw_fire = dut->dbg_aw_fire;
    bool dn      = dut->done;
    main_time++;
    dut->clk = 1; dut->eval(); main_time++;   // rising edge
    if (w_fire) { if (w_last) g_wlast_at.push_back(g_beat_count); g_beat_count++; }
    if (aw_fire) g_aw_count++;
    if (dn) g_done_pulses++;
    if (consume) g_src_idx++;
}

static void reset() {
    dut->rst_n = 0; dut->wr_latency = 0;
    dut->init_we = 0; dut->init_addr = 0; dut->init_data = 0; dut->bd_raddr = 0;
    dut->desc_valid = 0; dut->desc_addr = 0; dut->desc_len_bytes = 0;
    dut->src_valid = 0; dut->src_data = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static void dram_write(uint32_t addr, uint8_t data) {
    dut->init_we = 1; dut->init_addr = addr; dut->init_data = data; tick();
    dut->init_we = 0;
}

static uint8_t bd_read(uint32_t addr) {
    dut->bd_raddr = addr; dut->eval();
    return (uint8_t)dut->bd_rdata;
}

static std::vector<uint64_t> pack_beats(const std::vector<uint8_t>& img) {
    std::vector<uint64_t> beats(img.size() / BYTES_PER_BEAT);
    for (size_t k = 0; k < beats.size(); k++) {
        uint64_t w = 0;
        for (int b = 0; b < BYTES_PER_BEAT; b++)
            w |= (uint64_t)img[k * BYTES_PER_BEAT + b] << (8 * b);
        beats[k] = w;
    }
    return beats;
}

// Issue one descriptor and run to completion. Returns cycle count.
static int run_desc(uint32_t base, uint32_t len, const std::vector<uint64_t>& beats,
                    uint16_t wr_lat) {
    g_src = beats; g_src_idx = 0;
    g_done_pulses = 0; g_beat_count = 0; g_aw_count = 0; g_wlast_at.clear();
    dut->wr_latency = wr_lat;

    // Issue descriptor: hold valid until an edge where desc_ready is high.
    dut->desc_valid = 1; dut->desc_addr = base; dut->desc_len_bytes = len;
    int guard = 0;
    while (true) {
        dut->src_valid = 0;
        dut->clk = 0; dut->eval();
        bool ready = dut->desc_ready;
        main_time++;
        dut->clk = 1; dut->eval(); main_time++;   // if ready, descriptor accepted here
        if (ready || ++guard > 100) break;
    }
    dut->desc_valid = 0;

    // Run until done, then a few extra cycles to confirm it doesn't re-pulse.
    int cyc = 0, extra = 0;
    while (cyc < 500000) {
        tick(); cyc++;
        if (g_done_pulses > 0) { if (++extra >= 5) break; }
    }
    return cyc;
}

static int g_errors = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("    FAIL: "); printf(__VA_ARGS__); printf("\n"); g_errors++; } } while (0)

static void run_case(const char* name, uint32_t base, uint32_t len, uint16_t wr_lat) {
    const int NB = (int)(len / BYTES_PER_BEAT);

    // Deterministic byte image; base folded in so cases have distinct data.
    std::vector<uint8_t> img(len);
    for (uint32_t i = 0; i < len; i++)
        img[i] = (uint8_t)((i * 7 + 3 + (base & 0xFF)) & 0xFF);

    // Sentinel-preload the window + 16-byte guards on each side with 0xAA.
    uint32_t g0 = (base >= 16) ? base - 16 : 0;
    uint32_t g1 = base + len + 16;
    for (uint32_t a = g0; a < g1; a++) dram_write(a, 0xAA);

    int cyc = run_desc(base, len, pack_beats(img), wr_lat);

    // 1. byte-exact placement
    bool bytes_ok = true;
    for (uint32_t i = 0; i < len; i++) {
        uint8_t got = bd_read(base + i);
        if (got != img[i]) {
            CHECK(false, "%s wr_lat=%u: byte@%u got %02x exp %02x", name, wr_lat, i, got, img[i]);
            bytes_ok = false; break;
        }
    }
    // 2. guards untouched
    for (uint32_t a = g0; a < base; a++)
        CHECK(bd_read(a) == 0xAA, "%s wr_lat=%u: front guard@%u overwritten", name, wr_lat, a);
    for (uint32_t a = base + len; a < g1; a++)
        CHECK(bd_read(a) == 0xAA, "%s wr_lat=%u: back guard@%u overwritten", name, wr_lat, a);

    // 3. WLAST at each burst boundary
    std::vector<int> expect_wlast;
    for (int s = 0; s < NB; s += MAX_BURST) {
        int sz = std::min(MAX_BURST, NB - s);
        expect_wlast.push_back(s + sz - 1);
    }
    CHECK(g_wlast_at == expect_wlast, "%s wr_lat=%u: WLAST positions mismatch (got %zu, exp %zu)",
          name, wr_lat, g_wlast_at.size(), expect_wlast.size());

    // 4. AW burst count = ceil(NB / MAX_BURST)
    int exp_bursts = (NB + MAX_BURST - 1) / MAX_BURST;
    CHECK(g_aw_count == exp_bursts, "%s wr_lat=%u: AW count got %d exp %d", name, wr_lat, g_aw_count, exp_bursts);

    // 5. all beats consumed
    CHECK((int)g_src_idx == NB, "%s wr_lat=%u: consumed %zu beats exp %d", name, wr_lat, g_src_idx, NB);

    // 6. done exactly once
    CHECK(g_done_pulses == 1, "%s wr_lat=%u: done pulsed %d times (exp 1)", name, wr_lat, g_done_pulses);

    if (bytes_ok)
        printf("  [%-14s base=0x%05x len=%4u beats=%2d bursts=%d wr_lat=%3u] %s  (%d cyc)\n",
               name, base, len, NB, exp_bursts, wr_lat,
               (g_errors == 0 ? "PASS" : "has-fail"), cyc);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_dma_write_unit;
    reset();

    struct Case { const char* name; uint32_t base; uint32_t len; };
    // (a) short single-burst; (b) multi-burst > MAX_BURST @ non-zero base; (c) tiny.
    Case cases[] = {
        {"short_1burst",  0x00000, 64},    // 8 beats, 1 burst
        {"multi_3burst",  0x01000, 320},   // 40 beats, 3 bursts, non-zero base
        {"tiny_2beat",    0x02000, 16},    // 2 beats, 1 burst
    };
    const uint16_t lats[] = {0, 20, 100};

    printf("=== dma_write_engine + axi_mem_model_rw unit test ===\n");
    for (auto& c : cases)
        for (uint16_t lat : lats)
            run_case(c.name, c.base, c.len, lat);

    printf("\n=== Coverage Summary ===\n");
    printf("Module           : dma_write_engine + axi_mem_model_rw\n");
    printf("Scenarios covered: wb_short/multiburst/tiny x wr_latency{0,20,100}\n");
    printf("Test cases run   : %d\n", (int)(sizeof(cases)/sizeof(cases[0])) * 3);
    printf("Mismatches       : %d\n", g_errors);
    printf("Result           : %s\n", g_errors == 0 ? "PASS" : "FAIL");
    printf("\nRESULT: %s\n", g_errors == 0 ? "PASS" : "FAIL");

    delete dut;
    return g_errors ? 1 : 0;
}
