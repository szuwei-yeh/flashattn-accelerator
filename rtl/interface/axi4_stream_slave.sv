// ============================================================
//  axi4_stream_slave.sv — AXI4-Stream slave for Q/K/V loading
//
//  Receives Q, K, V matrices over AXI4-Stream (1 byte per beat,
//  INT8) and routes each byte to the correct head's SRAM.
//
//  Data layout per matrix:
//    SEQ_LEN rows × HEAD_DIM cols, row-major.
//    Within each row: bytes[0..PHD-1]=head0, bytes[PHD..2*PHD-1]=head1, etc.
//
//  Transfer sequence (three consecutive AXI-Stream packets):
//    Q (tlast marks end of Q matrix)
//    K (tlast marks end of K matrix)
//    V (tlast marks end of V matrix → FSM enters S_DONE, load_done asserts)
//
//  Address decomposition (all bit-slices, no division):
//    col      = byte_cnt[HEAD_LOG2-1:0]
//    head_idx = col[HEAD_LOG2-1:PHD_LOG2]
//    phd_col  = col[PHD_LOG2-1:0]
//    row      = byte_cnt[HEAD_LOG2+:8]
//    waddr    = {row[7:0], phd_col}  (= row*PER_HEAD_DIM + phd_col)
//
//  All dimension parameters must be powers of 2.
// ============================================================
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
module axi4_stream_slave #(
    parameter int SEQ_LEN      = 64,
    parameter int HEAD_DIM     = 64,
    parameter int NUM_HEADS    = 4,
    parameter int PER_HEAD_DIM = HEAD_DIM / NUM_HEADS  // 16
)
/* verilator lint_on UNUSEDPARAM */
(
    input  logic clk,
    input  logic rst_n,

    // AXI4-Stream slave
    input  logic [7:0]  s_tdata,
    input  logic        s_tvalid,
    output logic        s_tready,
    input  logic        s_tlast,

    // Write output — decoded by flash_attn_top into per-core write ports
    output logic        we,
    output logic [11:0] waddr,     // address within per-head SRAM
    output logic [7:0]  wdata,
    output logic [1:0]  head_sel,  // 0-3
    output logic [1:0]  mat_sel,   // 0=Q  1=K  2=V

    output logic        load_done  // high while in S_DONE (asserted after V tlast)
);

    // -------------------------------------------------------------------
    // Local parameters
    // -------------------------------------------------------------------
    localparam int HEAD_LOG2 = $clog2(HEAD_DIM);     // e.g. 6 for HEAD_DIM=64
    localparam int PHD_LOG2  = $clog2(PER_HEAD_DIM); // e.g. 4 for PHD=16
    localparam int NH_LOG2   = $clog2(NUM_HEADS);     // e.g. 2 for 4 heads

    // byte counter: max = SEQ_LEN*HEAD_DIM → need enough bits
    // For SEQ_LEN=256, HEAD_DIM=64: max=16384 → 14 bits
    localparam int CNT_W = 14;

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE, S_RECV_Q, S_RECV_K, S_RECV_V, S_DONE
    } state_t;

    state_t state;

    // -------------------------------------------------------------------
    // Byte counter (resets at each matrix boundary)
    // -------------------------------------------------------------------
    logic [CNT_W-1:0] byte_cnt;

    // -------------------------------------------------------------------
    // Address decomposition — pure bit-slicing, no division
    // -------------------------------------------------------------------
    logic [HEAD_LOG2-1:0] col_w;
    logic [NH_LOG2-1:0]   head_w;
    logic [PHD_LOG2-1:0]  phd_w;
    logic [7:0]           row_w;   // supports up to SEQ_LEN=256

    assign col_w  = byte_cnt[HEAD_LOG2-1:0];
    assign head_w = col_w[HEAD_LOG2-1 -: NH_LOG2];   // e.g. col_w[5:4]
    assign phd_w  = col_w[PHD_LOG2-1:0];             // e.g. col_w[3:0]
    assign row_w  = byte_cnt[HEAD_LOG2 +: 8];        // e.g. byte_cnt[13:6]

    // -------------------------------------------------------------------
    // Combinational outputs
    // -------------------------------------------------------------------
    logic accepting;
    assign accepting = (state == S_RECV_Q || state == S_RECV_K || state == S_RECV_V);

    always_comb begin
        s_tready  = accepting;
        we        = accepting & s_tvalid;
        waddr     = {row_w, phd_w};          // 12 bits: {8-bit row, 4-bit col}
        wdata     = s_tdata;
        head_sel  = head_w;
        load_done = (state == S_DONE);

        unique case (state)
            S_RECV_Q:  mat_sel = 2'd0;
            S_RECV_K:  mat_sel = 2'd1;
            default:   mat_sel = 2'd2;       // S_RECV_V and others
        endcase
    end

    // -------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            byte_cnt <= '0;
        end else begin
            unique case (state)

                S_IDLE: begin
                    byte_cnt <= '0;
                    if (s_tvalid) state <= S_RECV_Q;
                end

                S_RECV_Q, S_RECV_K, S_RECV_V: begin
                    if (s_tvalid & accepting) begin
                        if (s_tlast) begin
                            byte_cnt <= '0;
                            unique case (state)
                                S_RECV_Q: state <= S_RECV_K;
                                S_RECV_K: state <= S_RECV_V;
                                S_RECV_V: state <= S_DONE;
                                default:  state <= S_DONE;
                            endcase
                        end else begin
                            byte_cnt <= byte_cnt + 1;
                        end
                    end
                end

                S_DONE: ;  // stays here until rst_n

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
