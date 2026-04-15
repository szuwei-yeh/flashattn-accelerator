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
    parameter int NUM_KV_HEADS = NUM_HEADS,              // GQA: KV heads ≤ NUM_HEADS
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
    localparam int HEAD_LOG2    = $clog2(HEAD_DIM);      // e.g. 6 for HEAD_DIM=64
    localparam int PHD_LOG2     = $clog2(PER_HEAD_DIM);  // e.g. 4 for PHD=16
    localparam int NH_LOG2      = $clog2(NUM_HEADS);      // e.g. 2 for 4 Q-heads
    localparam int ROW_LOG2     = $clog2(SEQ_LEN);       // e.g. 6 for SEQ_LEN=64
    // GQA: KV matrices are NUM_KV_HEADS heads wide instead of NUM_HEADS
    localparam int KV_HEAD_DIM  = NUM_KV_HEADS * PER_HEAD_DIM;
    localparam int KV_HEAD_LOG2 = $clog2(KV_HEAD_DIM);
    // Safe: ensure ≥1 bit even for MQA (NUM_KV_HEADS=1 → $clog2=0)
    localparam int NH_KV_LOG2   = (NUM_KV_HEADS > 1) ? $clog2(NUM_KV_HEADS) : 1;

    // byte counter: sized for Q (largest packet = SEQ_LEN*HEAD_DIM bytes)
    localparam int CNT_W = HEAD_LOG2 + ROW_LOG2;

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
    //
    //  Q phase  (HEAD_DIM-wide row):  head_sel cycles 0..NUM_HEADS-1
    //  K/V phase (KV_HEAD_DIM-wide row): head_sel cycles 0..NUM_KV_HEADS-1
    //  For MHA (NUM_KV_HEADS==NUM_HEADS) both paths are identical.
    // -------------------------------------------------------------------
    logic is_kv_phase;
    assign is_kv_phase = (state == S_RECV_K || state == S_RECV_V);

    // Q decomposition
    logic [HEAD_LOG2-1:0]    col_q;
    logic [NH_LOG2-1:0]      head_q;
    logic [PHD_LOG2-1:0]     phd_q;
    logic [ROW_LOG2-1:0]     row_q;
    assign col_q  = byte_cnt[HEAD_LOG2-1:0];
    assign head_q = col_q[HEAD_LOG2-1 -: NH_LOG2];
    assign phd_q  = col_q[PHD_LOG2-1:0];
    assign row_q  = byte_cnt[HEAD_LOG2 +: ROW_LOG2];

    // KV decomposition (KV_HEAD_DIM-wide row)
    logic [KV_HEAD_LOG2-1:0] col_kv;
    logic [NH_KV_LOG2-1:0]   head_kv;
    logic [PHD_LOG2-1:0]     phd_kv;
    logic [ROW_LOG2-1:0]     row_kv;
    assign col_kv  = byte_cnt[KV_HEAD_LOG2-1:0];
    assign head_kv = col_kv[KV_HEAD_LOG2-1 -: NH_KV_LOG2];
    assign phd_kv  = col_kv[PHD_LOG2-1:0];
    assign row_kv  = byte_cnt[KV_HEAD_LOG2 +: ROW_LOG2];

    // Muxed address fields (GQA: switch to KV decomp during K/V phases)
    logic [NH_LOG2-1:0]  head_w;
    logic [PHD_LOG2-1:0] phd_w;
    logic [ROW_LOG2-1:0] row_w;
    assign head_w = is_kv_phase ? NH_LOG2'(head_kv) : head_q;
    assign phd_w  = is_kv_phase ? phd_kv            : phd_q;
    assign row_w  = is_kv_phase ? row_kv            : row_q;

    // -------------------------------------------------------------------
    // Combinational outputs
    // -------------------------------------------------------------------
    logic accepting;
    assign accepting = (state == S_RECV_Q || state == S_RECV_K || state == S_RECV_V);

    always_comb begin
        s_tready  = accepting;
        we        = accepting & s_tvalid;
        waddr     = (12'(row_w) << PHD_LOG2) | 12'(phd_w); // row*PER_HEAD_DIM + col
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
