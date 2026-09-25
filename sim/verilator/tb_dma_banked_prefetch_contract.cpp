// Directed transaction-contract test for flash_attn_top_dma_banked_prefetch.
// Covers rejected runtime-short configurations, accepted-start configuration
// locking, ignored busy-time starts, causal locking, and single-shot enforcement.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "Vtb_dma_banked_prefetch_harness.h"
#include "verilated.h"

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        failures++;
    }
}

void tick(Vtb_dma_banked_prefetch_harness* dut) {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

bool load_i8(const std::string& path, std::vector<int8_t>& out, int count) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    out.resize(count);
    for (int i = 0; i < count; i++) {
        unsigned value;
        if (std::fscanf(f, "%X", &value) != 1) {
            std::fclose(f);
            return false;
        }
        out[i] = static_cast<int8_t>(value & 0xff);
    }
    std::fclose(f);
    return true;
}

bool load_i32(const std::string& path, std::vector<int32_t>& out, int count) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    out.resize(count);
    for (int i = 0; i < count; i++) {
        unsigned value;
        if (std::fscanf(f, "%X", &value) != 1) {
            std::fclose(f);
            return false;
        }
        out[i] = static_cast<int32_t>(value);
    }
    std::fclose(f);
    return true;
}

void load_scales(const std::string& path, uint16_t& sq, uint16_t& sk,
                 uint16_t& sv) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    char line[128];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned value;
        if (std::sscanf(line, "scale_q_q88 = 0x%X", &value) == 1) sq = value;
        if (std::sscanf(line, "scale_k_q88 = 0x%X", &value) == 1) sk = value;
        if (std::sscanf(line, "scale_v_q88 = 0x%X", &value) == 1) sv = value;
    }
    std::fclose(f);
}

void pulse_start(Vtb_dma_banked_prefetch_harness* dut) {
    dut->start = 1;
    tick(dut);
    dut->start = 0;
}

void drive_valid_config(Vtb_dma_banked_prefetch_harness* dut, int n, int mat,
                        uint16_t sq, uint16_t sk, uint16_t sv, bool causal) {
    dut->cfg_seq_len = n;
    dut->cfg_q_base = 0;
    dut->cfg_k_base = mat;
    dut->cfg_v_base = 2 * mat;
    dut->scale_q = sq;
    dut->scale_k = sk;
    dut->scale_v = sv;
    dut->causal = causal;
    tick(dut);
    check(!dut->cfg_error, "canonical aligned configuration must be valid");
}

bool wait_done_with_hostile_inputs(Vtb_dma_banked_prefetch_harness* dut,
                                   int max_cycles, bool accepted_causal) {
    // Change every live transaction-level input immediately after acceptance.
    dut->cfg_seq_len = 48;
    dut->cfg_q_base = 0x1010;
    dut->cfg_k_base = 0x2020;
    dut->cfg_v_base = 0x3030;
    dut->scale_q ^= 0x0055;
    dut->scale_k ^= 0x0033;
    dut->scale_v ^= 0x0077;
    dut->causal = !accepted_causal;

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        // These raw start pulses occur while the scheduler/core are busy and
        // must neither restart nor clear state in the active transaction.
        dut->start = (cycle == 20 || cycle == 500 || cycle == 2000);
        tick(dut);
        dut->start = 0;
        if (dut->done) return true;
    }
    return false;
}

int compare_exact(Vtb_dma_banked_prefetch_harness* dut,
                  const std::vector<int32_t>& expected) {
    int mismatches = 0;
    for (int i = 0; i < static_cast<int>(expected.size()); i++) {
        dut->out_raddr = i;
        tick(dut);
        tick(dut);
        const int32_t actual = static_cast<int32_t>(dut->out_rdata);
        if (actual != expected[i]) {
            if (mismatches < 5)
                std::printf("  exact mismatch [%d] actual=0x%08X expected=0x%08X\n",
                            i, static_cast<uint32_t>(actual),
                            static_cast<uint32_t>(expected[i]));
            mismatches++;
        }
    }
    return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    constexpr int N = 64;
    constexpr int D = 16;
    constexpr int MAT = N * D;
    const std::string data = "../../data/N64";

    std::vector<int8_t> q, k, v;
    std::vector<int32_t> expected;
    if (!load_i8(data + "/q_input.hex", q, MAT) ||
        !load_i8(data + "/k_input.hex", k, MAT) ||
        !load_i8(data + "/v_input.hex", v, MAT) ||
        !load_i32(data + "/expected.hex", expected, MAT)) {
        std::printf("FAIL: unable to load canonical vectors\n");
        return 1;
    }
    uint16_t sq = 0x100, sk = 0x100, sv = 0x100;
    load_scales(data + "/scales.txt", sq, sk, sv);

    auto* dut = new Vtb_dma_banked_prefetch_harness;
    dut->inject_rresp = 0; dut->inject_bad_rlast = 0;
    dut->rst_n = 0;
    dut->start = 0;
    dut->causal = 0;
    dut->scale_q = sq;
    dut->scale_k = sk;
    dut->scale_v = sv;
    dut->cfg_seq_len = N;
    dut->cfg_q_base = 0;
    dut->cfg_k_base = MAT;
    dut->cfg_v_base = 2 * MAT;
    dut->rd_latency = 0;
    dut->init_we = 0;
    dut->init_addr = 0;
    dut->init_data = 0;
    dut->out_raddr = 0;
    tick(dut);
    tick(dut);
    dut->rst_n = 1;
    tick(dut);

    auto write_dram = [&](int base, const std::vector<int8_t>& src) {
        for (int i = 0; i < static_cast<int>(src.size()); i++) {
            dut->init_we = 1;
            dut->init_addr = base + i;
            dut->init_data = static_cast<uint8_t>(src[i]);
            tick(dut);
        }
        dut->init_we = 0;
        tick(dut);
    };
    write_dram(0, q);
    write_dram(MAT, k);
    write_dram(2 * MAT, v);

    // Invalid configs must be rejected without starting counters/DMA.  A later
    // valid run proves the rejection did not leave the machine wedged.
    dut->cfg_seq_len = 0;
    dut->eval();
    check(dut->cfg_error, "cfg_seq_len=0 must be rejected");
    pulse_start(dut);
    for (int i = 0; i < 16; i++) tick(dut);
    check(dut->perf_total_cycles == 0 && dut->dbg_kv_tiles_ready == 0,
          "rejected zero-length start must not launch work");

    dut->cfg_seq_len = 48;  // aligned but shorter than compile-time SEQ_LEN
    dut->eval();
    check(dut->cfg_error, "runtime-short cfg_seq_len must be rejected");
    pulse_start(dut);
    for (int i = 0; i < 16; i++) tick(dut);
    check(dut->perf_total_cycles == 0 && dut->dbg_kv_tiles_ready == 0,
          "rejected runtime-short start must remain idle, not hang");

    dut->cfg_seq_len = 63;  // neither equal to SEQ_LEN nor tile-aligned
    dut->eval();
    check(dut->cfg_error, "misaligned cfg_seq_len must be rejected");
    pulse_start(dut);
    for (int i = 0; i < 16; i++) tick(dut);
    check(dut->perf_total_cycles == 0 && dut->dbg_kv_tiles_ready == 0,
          "rejected misaligned sequence start must not launch work");

    dut->cfg_seq_len = N;
    dut->cfg_q_base = 8;
    dut->eval();
    check(dut->cfg_error, "non-16-byte-aligned AXI base must be rejected");
    pulse_start(dut);
    for (int i = 0; i < 16; i++) tick(dut);
    check(dut->perf_total_cycles == 0 && dut->dbg_kv_tiles_ready == 0,
          "rejected unaligned-base start must not launch work");

    // Transaction 1: accept non-causal config, then corrupt all pins and inject
    // busy starts.  The result must remain exactly the accepted transaction.
    drive_valid_config(dut, N, MAT, sq, sk, sv, false);
    pulse_start(dut);
    check(wait_done_with_hostile_inputs(dut, 200000, false),
          "transaction 1 timed out after config mutation/busy starts");
    const int first_mismatches = compare_exact(dut, expected);
    check(first_mismatches == 0,
          "transaction 1 changed after live scale/causal/base/busy-start mutation");
    check(dut->perf_dma_bytes == 3 * MAT,
          "transaction 1 must transfer exactly Q+K+V bytes");

    // Back-to-back analysis found that the output SRAM has no transaction clear.
    // Verify the intentionally enforced single-shot-until-reset contract: a
    // second start is ignored and cannot alter counters or the first result.
    const uint32_t held_total = dut->perf_total_cycles;
    const uint32_t held_bytes = dut->perf_dma_bytes;
    drive_valid_config(dut, N, MAT, sq, sk, sv, true);
    pulse_start(dut);
    for (int i = 0; i < 64; i++) tick(dut);
    check(dut->perf_total_cycles == held_total &&
              dut->perf_dma_bytes == held_bytes,
          "post-done start must be ignored until reset");
    check(compare_exact(dut, expected) == 0,
          "rejected post-done start must not alter the completed result");

    // Reset and rerun the same DUT; SRAM retains the previous transaction.
    dut->rst_n = 0;
    tick(dut); tick(dut);
    dut->rst_n = 1;
    drive_valid_config(dut, N, MAT, sq, sk, sv, false);
    pulse_start(dut);
    check(wait_done_with_hostile_inputs(dut, 200000, false), "reset-restart timed out");
    check(!dut->dma_error && compare_exact(dut, expected) == 0,
          "reset-restart consumed stale output SRAM");
    std::printf("PASS same-instance reset-restart with retained output SRAM\n");

    // Abort after eight real SRAM write edges, partway through the first fused
    // output update. Common reset must cancel pending work without clearing SRAM.
    dut->rst_n = 0; tick(dut); tick(dut); dut->rst_n = 1;
    drive_valid_config(dut, N, MAT, sq, sk, sv, false);
    pulse_start(dut);
    int output_writes = 0;
    for (int cycle = 0; cycle < 200000 && output_writes < 8 && !dut->done; ++cycle) {
        dut->clk = 0; dut->eval();
        const bool write_at_edge = dut->dbg_output_we;
        tick(dut);
        if (write_at_edge) ++output_writes;
    }
    check(output_writes == 8 && !dut->done && !dut->dma_error,
          "mid-output reset trigger was not reached during a healthy transaction");
    dut->rst_n = 0; tick(dut); tick(dut);
    check(!dut->done && !dut->dma_error && !dut->dbg_output_we &&
          dut->dbg_kv_tiles_ready == 0 && dut->perf_total_cycles == 0,
          "common reset did not clear pending work and transaction state");
    dut->rst_n = 1;
    // Change V for the restarted transaction: zero output is independent of the
    // old partial result, so replaying stale data cannot masquerade as recovery.
    write_dram(2 * MAT, std::vector<int8_t>(MAT, 0));
    drive_valid_config(dut, N, MAT, sq, sk, sv, false);
    pulse_start(dut);
    check(wait_done_with_hostile_inputs(dut, 200000, false),
          "restart after mid-output reset timed out");
    check(!dut->dma_error && compare_exact(dut, std::vector<int32_t>(MAT, 0)) == 0,
          "mid-output reset recovery leaked stale or partial output");
    check(dut->perf_dma_bytes == 3 * MAT && dut->dbg_kv_tiles_ready == N / 16,
          "mid-output reset recovery did not reload a complete transaction");
    write_dram(2 * MAT, v); // restore canonical data for subsequent error cases
    if (failures == 0)
        std::printf("PASS mid-output reset after %d writes; changed-V restart exact\n", output_writes);

    // Check error propagation both before compute and while later K/V streams.
    for (int kind = 0; kind < 3; ++kind) {
        dut->rst_n = 0; tick(dut); tick(dut); dut->rst_n = 1;
        dut->inject_rresp = 0; dut->inject_bad_rlast = 0;
        drive_valid_config(dut, N, MAT, sq, sk, sv, false);
        pulse_start(dut);
        if (kind == 2) {
            int wait = 0;
            while (dut->dbg_kv_tiles_ready == 0 && wait++ < 10000) tick(dut);
            check(dut->dbg_kv_tiles_ready == 1, "first K/V did not become resident");
        }
        const auto ready_before = dut->dbg_kv_tiles_ready;
        if (kind == 1) dut->inject_bad_rlast = 1;
        else dut->inject_rresp = 2;
        int wait = 0;
        while (!dut->dma_error && wait++ < 10000) tick(dut);
        check(dut->dma_error, "DMA fault not exposed to host");
        for (int i = 0; i < 1000; ++i) {
            tick(dut);
            check(!dut->done && dut->dma_error, "fault reported as successful completion");
        }
        check(dut->dbg_kv_tiles_ready == ready_before, "failed tile promoted resident");
        dut->inject_rresp = 0; dut->inject_bad_rlast = 0;
    }
    // Recovery requires the shared reset of the DMA and modeled AXI slave.
    dut->rst_n = 0; tick(dut); tick(dut); dut->rst_n = 1;
    drive_valid_config(dut, N, MAT, sq, sk, sv, false);
    pulse_start(dut);
    check(wait_done_with_hostile_inputs(dut, 200000, false), "post-error recovery timed out");
    check(!dut->dma_error && compare_exact(dut, expected) == 0, "post-error recovery mismatch");
    std::printf("PASS host error propagation, residency suppression, common-reset recovery\n");

    if (failures == 0) {
        std::printf("PASS zero/runtime-short/misaligned cfg and unaligned-base rejection\n");
        std::printf("PASS accepted-start scale/causal/DMA-config locking\n");
        std::printf("PASS busy-time start/config isolation\n");
        std::printf("PASS documented single-shot-until-reset enforcement\n");
        std::printf("RESULT: PASS\n");
    } else {
        std::printf("RESULT: FAIL (%d checks)\n", failures);
    }

    delete dut;
    return failures == 0 ? 0 : 1;
}
