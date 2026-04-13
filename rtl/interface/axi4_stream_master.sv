// ============================================================
//  axi4_stream_master.sv — AXI4-Stream master for output streaming
//
//  Reads output SRAMs from NUM_HEADS attention cores and streams
//  the results via AXI4-Stream (32-bit beats, one INT32 per beat).
//
//  Output order per matrix row:
//    head0 [PER_HEAD_DIM INT32s], head1, head2, head3
//  Total beats = SEQ_LEN * NUM_HEADS * PER_HEAD_DIM
//
//  3-stage micro-pipeline to account for 1-cycle SRAM read latency:
//    S_ADDR   → drive SRAM read address
//    S_WAIT   → SRAM data valid, latch it
//    S_STREAM → present data on AXI master, wait for tready
//
//  SRAM address per element (elem_cnt decomposition):
//    col    = elem_cnt[PHD_LOG2-1:0]
//    head   = elem_cnt[PHD_LOG2+NH_LOG2-1:PHD_LOG2]
//    row    = elem_cnt[PHD_LOG2+NH_LOG2 +: clog2(SEQ_LEN)]
//    sram_a = row * PER_HEAD_DIM + col  (bit-shift since power-of-2)
// ============================================================
`timescale 1ns/1ps

module axi4_stream_master #(
    parameter int SEQ_LEN      = 64,
    parameter int NUM_HEADS    = 4,
    parameter int PER_HEAD_DIM = 16
)(
    input  logic clk,
    input  logic rst_n,

    input  logic start,   // 1-cycle pulse: all cores done, begin output

    // Flat read ports to NUM_HEADS cores' output buffers
    output logic [NUM_HEADS*12-1:0] out_raddr_flat,   // [h*12 +: 12] = head h raddr
    input  logic [NUM_HEADS*32-1:0] out_rdata_flat,   // [h*32 +: 32] = head h rdata

    // AXI4-Stream master (32-bit beats, one INT32 per beat)
    output logic [31:0] m_tdata,
    output logic        m_tvalid,
    input  logic        m_tready,
    output logic        m_tlast
);

    localparam int TOTAL    = SEQ_LEN * NUM_HEADS * PER_HEAD_DIM;
    localparam int PHD_LOG2 = $clog2(PER_HEAD_DIM);      // 4
    localparam int NH_LOG2  = $clog2(NUM_HEADS);          // 2
    localparam int ROW_LOG2 = $clog2(SEQ_LEN);            // e.g. 6 for SEQ_LEN=64
    localparam int ECNT_W   = $clog2(TOTAL + 1);

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE, S_ADDR, S_WAIT, S_STREAM
    } state_t;

    state_t state;

    // -------------------------------------------------------------------
    // Element counter and address decomposition
    // -------------------------------------------------------------------
    logic [ECNT_W-1:0]     elem_cnt;

    logic [PHD_LOG2-1:0]   col_e;
    logic [NH_LOG2-1:0]    head_e;
    logic [ROW_LOG2-1:0]   row_e;

    assign col_e  = elem_cnt[PHD_LOG2-1:0];
    assign head_e = elem_cnt[PHD_LOG2 +: NH_LOG2];
    assign row_e  = elem_cnt[PHD_LOG2+NH_LOG2 +: ROW_LOG2];

    // SRAM address = row * PER_HEAD_DIM + col (power-of-2 → bit shift)
    logic [11:0] sram_addr_e;
    assign sram_addr_e = (12'(row_e) << PHD_LOG2) | 12'(col_e);

    // -------------------------------------------------------------------
    // Drive read addresses: active head gets current address, others 0
    // -------------------------------------------------------------------
    genvar hh;
    generate
        for (hh = 0; hh < NUM_HEADS; hh++) begin : raddr_drive
            assign out_raddr_flat[hh*12 +: 12] =
                ((state != S_IDLE) && (head_e == NH_LOG2'(hh)))
                ? sram_addr_e : 12'h0;
        end
    endgenerate

    // -------------------------------------------------------------------
    // Latch read data in S_WAIT (SRAM registered output is valid here)
    // -------------------------------------------------------------------
    logic [31:0] rdata_latch;

    always_ff @(posedge clk) begin
        if (state == S_WAIT) begin
            // Use a case statement to avoid variable part-select in clocked block
            unique case (head_e)
                2'd0: rdata_latch <= out_rdata_flat[ 0 +: 32];
                2'd1: rdata_latch <= out_rdata_flat[32 +: 32];
                2'd2: rdata_latch <= out_rdata_flat[64 +: 32];
                2'd3: rdata_latch <= out_rdata_flat[96 +: 32];
            endcase
        end
    end

    // -------------------------------------------------------------------
    // AXI-Stream outputs
    // -------------------------------------------------------------------
    localparam logic [ECNT_W-1:0] LAST_ELEM = ECNT_W'(TOTAL - 1);

    assign m_tdata  = rdata_latch;
    assign m_tvalid = (state == S_STREAM);
    assign m_tlast  = (state == S_STREAM) && (elem_cnt == LAST_ELEM);

    // -------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            elem_cnt <= '0;
        end else begin
            unique case (state)

                S_IDLE: begin
                    elem_cnt <= '0;
                    if (start) state <= S_ADDR;
                end

                S_ADDR: state <= S_WAIT;      // address driven this cycle

                S_WAIT: state <= S_STREAM;    // rdata latched, stream next cycle

                S_STREAM: begin
                    if (m_tready) begin
                        if (elem_cnt == LAST_ELEM) begin
                            state    <= S_IDLE;
                            elem_cnt <= '0;
                        end else begin
                            elem_cnt <= elem_cnt + 1;
                            state    <= S_ADDR;
                        end
                    end
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
