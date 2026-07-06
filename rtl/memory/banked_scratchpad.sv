// ============================================================
//  banked_scratchpad.sv — TPU-style banked on-chip scratchpad
//
//  NUM_BANKS independent byte-wide SRAM banks with a low-order interleave:
//      bank_id   = addr[LANE_LOG2-1:0]      (addr & (NUM_BANKS-1))
//      bank_addr = addr >> LANE_LOG2
//  so NUM_BANKS *contiguous* bytes land in NUM_BANKS distinct banks — i.e. a
//  16-byte tile row is a single conflict-free "stripe".  The byte at address A
//  always lives in bank A[3:0] at row A>>4 regardless of access width, so
//  scalar (1 byte/cycle) and vector (NUM_BANKS bytes/cycle) accesses are fully
//  coherent with each other.
//
//  One write port (scalar OR aligned vector) and one read port (scalar OR
//  aligned vector).  Each bank is 1R1W, so a vector write and a vector read
//  may proceed in the SAME cycle (2*NUM_BANKS bytes/cycle aggregate).
//
//  Vector accesses assume the base address is NUM_BANKS-aligned; the low
//  LANE_LOG2 bits are ignored (treated as 0).
// ============================================================
`timescale 1ns/1ps

module banked_scratchpad #(
    parameter int NUM_BANKS  = 16,
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 4096,               // total elements (Σ banks)
    localparam int LANE_LOG2   = $clog2(NUM_BANKS),
    localparam int BANK_DEPTH  = DEPTH / NUM_BANKS,
    localparam int BANK_ADDR_W = $clog2(BANK_DEPTH),
    localparam int ADDR_W      = $clog2(DEPTH),
    localparam int VEC_W       = NUM_BANKS * DATA_WIDTH
)(
    input  logic clk,

    // ── Write port (scalar OR vector) ────────────────────────────────
    input  logic                  w_en,
    input  logic                  w_vec,    // 1 = NUM_BANKS-byte stripe, 0 = 1 byte
    input  logic [ADDR_W-1:0]     w_addr,
    input  logic [DATA_WIDTH-1:0] w_sdata,  // scalar write data
    input  logic [VEC_W-1:0]      w_vdata,  // vector write data (lane b = byte b)

    // ── Read port (scalar OR vector), 1-cycle latency ────────────────
    input  logic                  r_en,
    input  logic                  r_vec,
    input  logic [ADDR_W-1:0]     r_addr,
    output logic [DATA_WIDTH-1:0] r_sdata,
    output logic [VEC_W-1:0]      r_vdata
);

    // Per-bank addresses (drop the lane bits)
    logic [BANK_ADDR_W-1:0] w_bank_addr, r_bank_addr;
    assign w_bank_addr = w_addr[ADDR_W-1:LANE_LOG2];
    assign r_bank_addr = r_addr[ADDR_W-1:LANE_LOG2];

    // Scalar lane selects
    logic [LANE_LOG2-1:0] w_lane, r_lane;
    assign w_lane = w_addr[LANE_LOG2-1:0];
    assign r_lane = r_addr[LANE_LOG2-1:0];

    // Per-bank read data
    logic [DATA_WIDTH-1:0] bank_rdata [NUM_BANKS-1:0];

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b++) begin : g_bank
            logic                  bank_we;
            logic [DATA_WIDTH-1:0] bank_wdata;

            // Write: vector → every bank; scalar → only the addressed lane.
            assign bank_we    = w_en & (w_vec | (w_lane == LANE_LOG2'(b)));
            assign bank_wdata = w_vec ? w_vdata[b*DATA_WIDTH +: DATA_WIDTH] : w_sdata;

            // Read: vector → every bank; scalar → only the addressed lane (saves
            // read power, and r_sdata picks that lane after the 1-cycle latency).
            logic bank_re;
            assign bank_re = r_en & (r_vec | (r_lane == LANE_LOG2'(b)));

            sram_1r1w #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(BANK_DEPTH)) u_bank (
                .clk   (clk),
                .we    (bank_we),
                .waddr (w_bank_addr),
                .wdata (bank_wdata),
                .re    (bank_re),
                .raddr (r_bank_addr),
                .rdata (bank_rdata[b])
            );

            assign r_vdata[b*DATA_WIDTH +: DATA_WIDTH] = bank_rdata[b];
        end
    endgenerate

    // Scalar read: the lane select must be delayed one cycle to match the
    // bank read latency.
    logic [LANE_LOG2-1:0] r_lane_d;
    always_ff @(posedge clk) begin
        if (r_en) r_lane_d <= r_lane;
    end
    assign r_sdata = bank_rdata[r_lane_d];

endmodule
