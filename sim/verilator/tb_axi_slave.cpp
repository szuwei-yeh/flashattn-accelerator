// ============================================================
//  tb_axi_slave.cpp — Unit test for axi4_stream_slave
//
//  Tests:
//    Test 1: Verify byte routing for first row (head_sel, waddr, wdata)
//    Test 2: Verify mat_sel transitions Q → K → V → load_done
// ============================================================

#include "Vaxi4_stream_slave.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static Vaxi4_stream_slave* dut;
static uint64_t sim_time = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    sim_time++;
}

static void reset() {
    dut->rst_n   = 0;
    dut->s_tvalid = 0;
    dut->s_tdata  = 0;
    dut->s_tlast  = 0;
    tick(); tick();
    dut->rst_n = 1;
    tick();
}

// Capture one write event
struct WriteEv {
    uint8_t  data;
    uint16_t waddr;
    uint8_t  head;
    uint8_t  mat;   // 0=Q, 1=K, 2=V
};

static std::vector<WriteEv> captured;

// Drive one byte and capture the write event (if we==1).
// Properly handles AXI backpressure: waits for s_tready before capturing.
// This correctly handles the S_IDLE→S_RECV_Q transition on the first byte.
static void drive_byte(uint8_t data, bool last) {
    dut->s_tdata  = data;
    dut->s_tvalid = 1;
    dut->s_tlast  = last ? 1 : 0;

    // Wait for tready (AXI: transfer only when both tvalid and tready are high)
    dut->eval();
    int wait_cnt = 0;
    while (!dut->s_tready) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
        sim_time++;
        dut->eval();
        if (++wait_cnt > 10000) { printf("drive_byte: tready TIMEOUT\n"); exit(1); }
    }

    // tready is high — capture the write event
    if (dut->we) {
        captured.push_back({
            data,
            (uint16_t)dut->waddr,
            (uint8_t)dut->head_sel,
            (uint8_t)dut->mat_sel
        });
    }

    // Complete the transfer with one clock edge
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    dut->s_tvalid = 0;
    sim_time++;
}

// Send N bytes of a complete matrix (tlast on last byte)
static void send_matrix(int total_bytes, uint8_t fill) {
    for (int i = 0; i < total_bytes; i++) {
        drive_byte(fill, i == total_bytes - 1);
    }
}

int main() {
    dut = new Vaxi4_stream_slave;

    // Default parameters: HEAD_DIM=64, NUM_HEADS=4, PER_HEAD_DIM=16, SEQ_LEN=64
    // Total bytes per matrix = 64 * 64 = 4096

    bool all_pass = true;

    // ----------------------------------------------------------------
    // Test 1: First row routing (64 bytes = one full row)
    //   bytes 0-15  → head=0, mat=Q, waddr=0..15
    //   bytes 16-31 → head=1, mat=Q, waddr=0..15
    //   bytes 32-47 → head=2, mat=Q, waddr=0..15
    //   bytes 48-63 → head=3, mat=Q, waddr=0..15
    // ----------------------------------------------------------------
    printf("[Test 1] First row byte routing (64 bytes)...\n");
    reset();
    captured.clear();

    for (int col = 0; col < 64; col++) {
        drive_byte((uint8_t)col, false);
    }

    bool t1_pass = true;
    if ((int)captured.size() < 64) {
        printf("  ERROR: only %d captured events (expected 64)\n", (int)captured.size());
        t1_pass = false;
    } else {
        // Check bytes 0-15: head=0, mat=0, waddr=0..15, data=col
        for (int col = 0; col < 64; col++) {
            int exp_head  = col / 16;
            int exp_waddr = col % 16;           // per-head column
            // waddr = {row[7:0]=0, phd_col[3:0]}
            int exp_waddr_full = (0 << 4) | exp_waddr;  // row=0

            auto& ev = captured[col];
            if (ev.head != exp_head || ev.mat != 0 ||
                ev.waddr != exp_waddr_full || ev.data != (uint8_t)col) {
                printf("  FAIL byte %d: got head=%d mat=%d waddr=%d data=%d"
                       " (exp head=%d mat=0 waddr=%d data=%d)\n",
                       col, ev.head, ev.mat, ev.waddr, ev.data,
                       exp_head, exp_waddr_full, col);
                t1_pass = false;
            }
        }
    }
    printf("  Test 1: %s\n", t1_pass ? "PASS" : "FAIL");
    if (!t1_pass) all_pass = false;

    // ----------------------------------------------------------------
    // Test 2: mat_sel transitions Q → K → V → load_done
    // Send full Q matrix, first K byte, full K, full V; check transitions.
    // ----------------------------------------------------------------
    printf("[Test 2] mat_sel transitions and load_done...\n");
    reset();
    captured.clear();

    const int N = 64, HD = 64;
    const int total = N * HD;  // 4096

    // Send Q (4096 bytes, tlast on last byte)
    send_matrix(total, 0xAA);

    // After Q's tlast: slave should now be in S_RECV_K
    // Drive first K byte and verify mat_sel=1
    drive_byte(0xBB, false);
    bool t2a = false;
    if (!captured.empty() && captured.back().mat == 1) {
        t2a = true;
        printf("  mat after Q tlast: K(1) — PASS\n");
    } else {
        int got_mat = captured.empty() ? -1 : captured.back().mat;
        printf("  mat after Q tlast: got %d — FAIL\n", got_mat);
    }

    // Send rest of K
    for (int i = 1; i < total; i++) {
        drive_byte(0xBB, i == total - 1);
    }

    // After K's tlast: slave in S_RECV_V
    // Drive first V byte and verify mat_sel=2
    drive_byte(0xCC, false);
    bool t2b = false;
    if (!captured.empty() && captured.back().mat == 2) {
        t2b = true;
        printf("  mat after K tlast: V(2) — PASS\n");
    } else {
        int got_mat = captured.empty() ? -1 : captured.back().mat;
        printf("  mat after K tlast: got %d — FAIL\n", got_mat);
    }

    // Send rest of V
    for (int i = 1; i < total; i++) {
        drive_byte(0xCC, i == total - 1);
    }

    // After V's tlast: slave in S_DONE, load_done=1
    tick(); tick();
    dut->eval();
    bool t2c = (dut->load_done == 1);
    printf("  load_done after V tlast: %s\n", t2c ? "PASS" : "FAIL");

    bool t2_pass = t2a && t2b && t2c;
    printf("  Test 2: %s\n", t2_pass ? "PASS" : "FAIL");
    if (!t2_pass) all_pass = false;

    // ----------------------------------------------------------------
    // Test 3: waddr for second row
    //   Row 1 bytes: byte_cnt 64..127
    //   For byte_cnt=64: col=64&63=0, head=0, phd=0, row=64>>6=1
    //   waddr = {1, 0} = 16
    // ----------------------------------------------------------------
    printf("[Test 3] Second row waddr check...\n");
    reset();
    captured.clear();

    // Send one full row (row 0) without tlast
    for (int col = 0; col < 64; col++) {
        drive_byte((uint8_t)(0x40 + col), false);
    }
    // Send first byte of row 1
    drive_byte(0x80, false);

    bool t3_pass = false;
    if (!captured.empty()) {
        auto& last_ev = captured.back();
        // Row 1, col 0 of head 0: waddr = {row=1, phd_col=0} = 16
        if (last_ev.head == 0 && last_ev.waddr == 16 && last_ev.mat == 0) {
            t3_pass = true;
        } else {
            printf("  FAIL: got head=%d waddr=%d mat=%d (exp head=0 waddr=16 mat=0)\n",
                   last_ev.head, last_ev.waddr, last_ev.mat);
        }
    } else {
        printf("  FAIL: no write event captured\n");
    }
    printf("  Test 3: %s\n", t3_pass ? "PASS" : "FAIL");
    if (!t3_pass) all_pass = false;

    printf("\n%s\n", all_pass ? "ALL PASS" : "SOME FAIL");

    int mismatches = (t1_pass ? 0 : 1) + (t2_pass ? 0 : 1) + (t3_pass ? 0 : 1);
    printf("\n=== Coverage Summary ===\n");
    printf("Module           : axi4_stream_slave\n");
    printf("Scenarios covered: byte_routing, mat_sel_transition, second_row_waddr\n");
    printf("Test cases run   : 3\n");
    printf("Mismatches       : %d\n", mismatches);
    printf("Result           : %s\n", all_pass ? "PASS" : "FAIL");

    delete dut;
    return all_pass ? 0 : 1;
}
