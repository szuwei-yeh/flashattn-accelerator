// Directed characterization of dma_engine_vec AXI read-channel corner cases.
// This is intentionally not an AXI VIP: C++ drives ARREADY and the R channel
// directly so RLAST/RRESP can be perturbed independently of the beat count.
#include "Vdma_engine_vec.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static Vdma_engine_vec* dut = nullptr;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

struct StripeWrite {
    uint16_t addr;
    uint8_t dst;
    uint64_t lo;
    uint64_t hi;
};

static std::vector<StripeWrite> writes;

static uint64_t wide_word64(const VlWide<4>& value, int word64) {
    const int lo_word = word64 * 2;
    return static_cast<uint64_t>(value[lo_word]) |
           (static_cast<uint64_t>(value[lo_word + 1]) << 32);
}

// Capture combinational handshake/write signals immediately before the edge
// that consumes them.  Sampling w_en after the edge would observe the updated
// beat_in_stripe state rather than the write presented to the sequential sink.
static bool tick_and_report_r_handshake() {
    dut->clk = 0;
    dut->eval();
    const bool r_fire = dut->m_rvalid && dut->m_rready;
    if (dut->w_en) {
        writes.push_back({
            static_cast<uint16_t>(dut->w_addr),
            static_cast<uint8_t>(dut->w_dst),
            wide_word64(dut->w_vdata, 0),
            wide_word64(dut->w_vdata, 1),
        });
    }
    dut->clk = 1;
    dut->eval();
    main_time++;
    return r_fire;
}

static void reset_dut() {
    writes.clear();
    dut->rst_n = 0;
    dut->desc_valid = 0;
    dut->desc_addr = 0;
    dut->desc_dst_addr = 0;
    dut->desc_len_bytes = 0;
    dut->desc_dst = 0;
    dut->m_arready = 0;
    dut->m_rdata = 0;
    dut->m_rresp = 0;
    dut->m_rlast = 0;
    dut->m_rvalid = 0;
    tick_and_report_r_handshake();
    tick_and_report_r_handshake();
    dut->rst_n = 1;
    tick_and_report_r_handshake();
}

struct CaseSpec {
    const char* name;
    int early_rlast_beat;  // 0-based; -1 means none
    bool final_rlast;
    bool send_late_rlast;
    int error_beat;        // 0-based; -1 means all OKAY
    uint8_t error_resp;
};

struct CaseResult {
    bool pass;
    int done_after_beat;
    bool late_beat_accepted;
    size_t stripe_writes;
    std::string observation;
};

static uint64_t beat_data(int beat) {
    return UINT64_C(0x1020304050607000) + static_cast<uint64_t>(beat);
}

static CaseResult run_case(const CaseSpec& spec) {
    constexpr uint32_t SRC = 0x1000;
    constexpr uint32_t LEN = 32;  // four 64-bit beats / two 128-bit stripes
    constexpr int BEATS = 4;

    reset_dut();

    dut->desc_addr = SRC;
    dut->desc_dst_addr = 0;
    dut->desc_len_bytes = LEN;
    dut->desc_dst = 1;
    dut->desc_valid = 1;
    tick_and_report_r_handshake();
    dut->desc_valid = 0;

    bool ar_ok = dut->m_arvalid && dut->m_araddr == SRC &&
                 dut->m_arlen == BEATS - 1 && dut->m_arsize == 3 &&
                 dut->m_arburst == 1;
    dut->m_arready = 1;
    tick_and_report_r_handshake();
    dut->m_arready = 0;

    int done_after = -1;
    bool all_expected_beats_accepted = true;
    for (int beat = 0; beat < BEATS; ++beat) {
        dut->m_rvalid = 1;
        dut->m_rdata = beat_data(beat);
        dut->m_rresp = (beat == spec.error_beat) ? spec.error_resp : 0;
        dut->m_rlast = (beat == spec.early_rlast_beat) ||
                       (spec.final_rlast && beat == BEATS - 1);
        if (!tick_and_report_r_handshake())
            all_expected_beats_accepted = false;
        if (dut->done && done_after < 0)
            done_after = beat + 1;
    }

    dut->m_rvalid = 0;
    dut->m_rlast = 0;
    dut->m_rresp = 0;

    bool late_accepted = false;
    if (spec.send_late_rlast) {
        dut->m_rvalid = 1;
        dut->m_rdata = beat_data(BEATS);
        dut->m_rlast = 1;
        late_accepted = tick_and_report_r_handshake();
        dut->m_rvalid = 0;
        dut->m_rlast = 0;
    } else {
        tick_and_report_r_handshake();
    }

    bool write_ok = writes.size() == 2;
    if (write_ok) {
        write_ok = writes[0].addr == 0 && writes[0].dst == 1 &&
                   writes[0].lo == beat_data(0) && writes[0].hi == beat_data(1) &&
                   writes[1].addr == 16 && writes[1].dst == 1 &&
                   writes[1].lo == beat_data(2) && writes[1].hi == beat_data(3);
    }

    const bool common_ok = ar_ok && all_expected_beats_accepted &&
                           done_after == BEATS && write_ok;
    bool pass = common_ok;
    std::string observation;

    if (spec.early_rlast_beat >= 0) {
        observation = "early RLAST ignored; completion remained internal-count based";
    } else if (!spec.final_rlast && spec.send_late_rlast) {
        pass = pass && !late_accepted;
        observation = "missing RLAST did not block completion; later beat was not accepted";
    } else if (!spec.final_rlast) {
        observation = "missing RLAST did not block completion";
    } else if (spec.error_beat >= 0) {
        observation = (spec.error_resp == 2)
            ? "SLVERR ignored; data/write/completion behavior unchanged"
            : "DECERR ignored; data/write/completion behavior unchanged";
    } else {
        observation = "normal four-beat burst completed on the expected final beat";
    }

    return {pass, done_after, late_accepted, writes.size(), observation};
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vdma_engine_vec;

    const CaseSpec cases[] = {
        {"Correct RLAST",      -1, true,  false, -1, 0},
        {"Early RLAST",         1, false, false, -1, 0},
        {"Missing RLAST",      -1, false, false, -1, 0},
        {"Late RLAST",         -1, false, true,  -1, 0},
        {"RRESP = SLVERR",     -1, true,  false,  1, 2},
        {"RRESP = DECERR",     -1, true,  false,  1, 3},
    };

    int failures = 0;
    std::printf("=== dma_engine_vec AXI read protocol characterization ===\n");
    std::printf("Descriptor: 32 bytes, 4 expected beats, 2 vector writes\n\n");
    for (const auto& spec : cases) {
        CaseResult result = run_case(spec);
        std::printf("%-18s %s — done after beat %d, writes=%zu; %s\n",
                    spec.name, result.pass ? "PASS" : "FAIL",
                    result.done_after_beat, result.stripe_writes,
                    result.observation.c_str());
        if (!result.pass)
            failures++;
    }

    std::printf("\nObserved interface: no read-error output/status is exposed by dma_engine_vec.\n");
    std::printf("RESULT: %s (%d/%zu failed)\n", failures ? "FAIL" : "PASS",
                failures, sizeof(cases) / sizeof(cases[0]));

    dut->final();
    delete dut;
    return failures ? 1 : 0;
}
