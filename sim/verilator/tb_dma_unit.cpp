// ============================================================
//  tb_dma_unit.cpp — Unit test for dma_engine + axi_mem_model.
//  Loads a known pattern into DRAM, issues a descriptor, captures the
//  scratchpad write stream, and checks every byte landed at the right
//  address.  Sweeps a few read latencies.
// ============================================================
#include "Vtb_dma_unit.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vtb_dma_unit* dut = nullptr;

// Captured scratchpad writes for the current descriptor
struct WriteRec { uint8_t dst; uint16_t addr; uint8_t data; };
static std::vector<WriteRec> g_writes;

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
    // sample write stream on the rising edge we just produced
    if (dut->w_we) {
        g_writes.push_back({(uint8_t)dut->w_dst, (uint16_t)dut->w_addr, (uint8_t)dut->w_data});
    }
}

static void reset() {
    dut->rst_n = 0; dut->desc_valid = 0; dut->init_we = 0;
    dut->desc_addr = 0; dut->desc_dst_addr = 0; dut->desc_len_bytes = 0; dut->desc_dst = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static void dram_write(uint32_t addr, uint8_t data) {
    dut->init_we = 1; dut->init_addr = addr; dut->init_data = data; tick();
    dut->init_we = 0;
}

// Run one descriptor, return cycles until done
static int run_desc(uint32_t src, uint16_t dst_addr, uint32_t len, uint8_t dst) {
    g_writes.clear();
    dut->desc_addr = src; dut->desc_dst_addr = dst_addr;
    dut->desc_len_bytes = len; dut->desc_dst = dst;
    dut->desc_valid = 1;
    // hold valid until accepted
    int guard = 0;
    while (!dut->desc_ready && guard++ < 100) tick();
    tick();                 // accepted this edge
    dut->desc_valid = 0;
    int cyc = 0;
    while (!dut->done && cyc < 200000) { tick(); cyc++; }
    return cyc;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_dma_unit;

    int errors = 0;
    reset();

    // Build a DRAM image: K matrix base = 0x1000, 256 bytes (one 16x16 tile)
    const uint32_t K_BASE = 0x1000;
    const uint32_t LEN = 256;
    std::vector<uint8_t> img(LEN);
    for (uint32_t i = 0; i < LEN; i++) {
        img[i] = (uint8_t)((i * 7 + 3) & 0xFF);
        dram_write(K_BASE + i, img[i]);
    }

    int latencies[] = {0, 20, 100};
    for (int li = 0; li < 3; li++) {
        dut->rd_latency = latencies[li];
        // dst_addr 0, dst=1 (K)
        int cyc = run_desc(K_BASE, 0, LEN, 1);

        // Check: exactly LEN writes, in order, to addresses 0..LEN-1, dst=1
        bool ok = (g_writes.size() == LEN);
        if (!ok) {
            printf("  [lat=%3d] FAIL: %zu writes (expected %u)\n",
                   latencies[li], g_writes.size(), LEN);
            errors++;
            continue;
        }
        for (uint32_t i = 0; i < LEN; i++) {
            if (g_writes[i].dst != 1 || g_writes[i].addr != i || g_writes[i].data != img[i]) {
                printf("  [lat=%3d] FAIL @%u: dst=%u addr=%u data=%02x (exp dst=1 addr=%u data=%02x)\n",
                       latencies[li], i, g_writes[i].dst, g_writes[i].addr, g_writes[i].data,
                       i, img[i]);
                errors++; ok = false; break;
            }
        }
        if (ok) printf("  [lat=%3d] PASS: %u bytes landed correctly in %d cycles\n",
                       latencies[li], LEN, cyc);
    }

    printf("\n=== DMA unit test: %s ===\n", errors == 0 ? "PASS" : "FAIL");
    delete dut;
    return errors ? 1 : 0;
}
