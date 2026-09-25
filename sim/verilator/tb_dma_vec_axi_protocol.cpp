// AXI read DMA scoreboard: stalls, boundary splitting, sticky fatal errors.
#include "Vdma_engine_vec.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* why) {
    if (!ok) { std::printf("FAIL %s\n", why); ++failures; }
}
static uint64_t pattern(int beat) { return UINT64_C(0x1020304050607000) + beat; }
struct Write { uint32_t addr; uint64_t lo, hi; };
static std::vector<Write> writes;
static void tick(Vdma_engine_vec& d) {
    d.clk = 0; d.eval();
    if (d.w_en) writes.push_back({d.w_addr,
        uint64_t(d.w_vdata[0]) | (uint64_t(d.w_vdata[1]) << 32),
        uint64_t(d.w_vdata[2]) | (uint64_t(d.w_vdata[3]) << 32)});
    d.clk = 1; d.eval();
}
static void reset(Vdma_engine_vec& d) {
    d.rst_n = 0; d.desc_valid = 0; d.m_arready = 0;
    d.m_rvalid = 0; d.m_rlast = 0; d.m_rresp = 0;
    tick(d); tick(d); d.rst_n = 1; tick(d); writes.clear();
}
// error_kind: 0 good, 1 early RLAST, 2 missing RLAST, 3 SLVERR,
// 4 DECERR, 5 EXOKAY (nonexclusive read), 6 late RLAST.
static void transfer(Vdma_engine_vec& d, uint32_t base, int bytes, int error_kind) {
    reset(d);
    d.desc_addr = base; d.desc_dst_addr = 0; d.desc_len_bytes = bytes;
    d.desc_dst = 1; d.desc_valid = 1; tick(d); d.desc_valid = 0;
    int beat = 0, bursts = 0;
    bool failed = false, completed = false;
    while (beat < bytes / 8 && !failed && !completed) {
        check(d.m_arvalid, "AR request missing");
        const uint32_t addr = d.m_araddr;
        const int count = d.m_arlen + 1;
        check(addr == base + unsigned(beat * 8), "AR address order");
        check((addr & 4095) + unsigned(count * 8) <= 4096, "burst crosses 4 KiB");
        check(count <= 16 && d.m_arsize == 3 && d.m_arburst == 1, "AR attributes");
        // Deterministic varying backpressure on every address request.
        d.m_arready = 0;
        for (int stall = 0; stall < 1 + bursts % 5; ++stall) {
            tick(d);
            check(d.m_arvalid && d.m_araddr == addr && d.m_arlen + 1 == count,
                  "AR payload changed under backpressure");
            check(!d.done && !d.error, "completion while AR stalled");
        }
        d.m_arready = 1; tick(d); d.m_arready = 0; ++bursts;
        for (int b = 0; b < count; ++b) {
            d.m_rvalid = 0;
            const auto old_writes = writes.size();
            for (int gap = 0; gap < beat % 4; ++gap) tick(d);
            check(writes.size() == old_writes && !d.done, "work advanced during R gap");
            const bool last = b == count - 1;
            d.m_rvalid = 1; d.m_rdata = pattern(beat); d.m_rresp = 0; d.m_rlast = last;
            if (error_kind == 1 && beat == 1) d.m_rlast = 1;
            if ((error_kind == 2 || error_kind == 6) && last) d.m_rlast = 0;
            if (beat == 1 && error_kind >= 3 && error_kind <= 5)
                d.m_rresp = error_kind == 3 ? 2 : error_kind == 4 ? 3 : 1;
            const bool bad = d.m_rresp != 0 || bool(d.m_rlast) != last;
            check(d.m_rready, "unexpected R backpressure");
            tick(d); ++beat;
            if (bad) {
                check(d.error && !d.done, "bad beat did not produce fatal status");
                check(writes.size() == old_writes, "bad beat wrote a stripe");
                failed = true; break;
            }
            if (d.done) {
                check(beat == bytes / 8, "premature successful done");
                completed = true;
            }
        }
        d.m_rvalid = 0;
    }
    if (error_kind) {
        check(failed, "injected error not observed");
        const auto size = writes.size();
        // Includes a late RLAST, extra start/descriptor, and continued RVALID.
        d.m_rvalid = 1; d.m_rlast = 1; d.m_rresp = 0; d.desc_valid = 1;
        for (int i = 0; i < 8; ++i) {
            tick(d);
            check(d.error && !d.done && !d.desc_ready && !d.m_rready && !d.m_arvalid,
                  "fatal state did not hold until reset");
        }
        check(writes.size() == size, "writes after fatal error");
    } else {
        check(completed && !d.error, "normal transfer did not complete");
        check(int(writes.size()) == bytes / 16, "stripe count");
        for (int i = 0; i < int(writes.size()); ++i)
            check(writes[i].addr == unsigned(i * 16) && writes[i].lo == pattern(2*i) &&
                  writes[i].hi == pattern(2*i+1), "stripe order or data");
    }
    std::printf("case base=0x%X bytes=%d error=%d bursts=%d checked\n",base,bytes,error_kind,bursts);
}
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vdma_engine_vec d;
    transfer(d, 0x1000, 32, 0);
    transfer(d, 0x0FF0, 512, 0);
    transfer(d, 0x0FF8, 256, 0); // stripe spans two AXI bursts: keep its low beat
    transfer(d, 0x1000, 4096, 0);
    for (int e = 1; e <= 6; ++e) transfer(d, 0x1000, 32, e);
    // Invalid descriptor rejection must not put an underflowed ARLEN on AXI.
    for (int kind = 0; kind < 4; ++kind) {
        reset(d); d.desc_addr = kind == 1 ? 3 : 0x1000;
        d.desc_dst_addr = kind == 2 ? 1 : 0;
        d.desc_len_bytes = kind == 0 ? 0 : kind == 3 ? 24 : 32;
        d.desc_valid = 1; tick(d); d.desc_valid = 0;
        check(d.error && !d.m_arvalid && !d.done, "invalid descriptor accepted");
    }
    // Abort in the middle of a burst. Common reset discards its half stripe.
    reset(d); d.desc_addr = 0x1000; d.desc_dst_addr = 0; d.desc_len_bytes = 32;
    d.desc_valid = 1; tick(d); d.desc_valid = 0;
    d.m_arready = 1; tick(d); d.m_arready = 0;
    d.m_rvalid = 1; d.m_rdata = pattern(0); d.m_rlast = 0; tick(d);
    reset(d);
    check(d.desc_ready && !d.error && !d.done && writes.empty(), "mid-burst reset state");
    transfer(d, 0x1000, 32, 0); // reset recovery on the same object
    std::printf("RESULT: %s (%d failures)\n",failures ? "FAIL" : "PASS",failures);
    return failures ? 1 : 0;
}
