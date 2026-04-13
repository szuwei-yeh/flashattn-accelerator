// ============================================================
//  flash_attn_top.sv — FlashAttention Accelerator Top Level (v3)
//
//  v3 changes (multi-tile N=64/128/256 support):
//  - SRAM_DEPTH = 4096 (max SEQ_LEN×HEAD_DIM = 256×16)
//  - Q/K/V: flat sram_1r1w (no ping-pong), read at global offsets:
//      Q:  q_global_offset + sram_addr = tile_row×HEAD_DIM + local
//      KV: k_global_offset + sram_addr = tile_col×HEAD_DIM + local
//  - Output buffer: global write addr = tile_row×HEAD_DIM + local
//  - tile_controller S_UPDATE_SOFTMAX always uses tile_start=tile_last=1:
//    each KV tile runs its own independent softmax; PV accumulates
//    across all KV tiles in the output buffer.
// ============================================================
`timescale 1ns/1ps

module flash_attn_top #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 16,
    parameter int SEQ_LEN   = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    input  logic        q_we_ext,
    input  logic [11:0] q_waddr_ext,
    input  logic [7:0]  q_wdata_ext,

    input  logic        kv_we_ext,
    input  logic [11:0] k_waddr_ext,
    input  logic [7:0]  k_wdata_ext,
    input  logic [11:0] v_waddr_ext,
    input  logic [7:0]  v_wdata_ext,

    input  logic        o_re_ext,
    input  logic [11:0] o_raddr_ext,
    output logic signed [31:0] o_rdata_ext,

    output logic [3:0]         dbg_state,
    output logic               dbg_sfx_tile_valid,
    output logic               dbg_sfx_out_valid,
    output logic [2:0]         dbg_softmax_state,
    output logic               dbg_is_pv_phase,
    output logic signed [31:0] dbg_acc0,
    output logic               dbg_accum_en,
    output logic [7:0]         dbg_sram_addr
);

    localparam int SIZE       = TILE_SIZE;
    localparam int FLAT       = SIZE * SIZE;   // elements per tile = 256
    localparam int SRAM_DEPTH = 4096;          // max SEQ_LEN×HEAD_DIM (256×16)

    // =========================================================
    // 1. Tile Controller + Addr Gen
    // =========================================================
    logic [15:0] tile_row, tile_col;
    logic        cnt_en, cnt_clr, cnt_done;
    logic        kv_swap_banks, q_we, kv_we;
    logic        array_start, array_done, array_busy;
    logic        softmax_tile_start, softmax_tile_valid, softmax_tile_last;
    logic        softmax_out_valid;
    logic        accum_en, norm_en;
    logic [11:0] sram_addr_wide;
    logic [7:0]  sram_addr;           // local tile index 0..FLAT-1

    tile_controller #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM), .SEQ_LEN(SEQ_LEN)
    ) u_fsm (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .mode(1'b0), .kv_len(16'b0),   // prefill-only: mode=0, kv_len unused
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr), .cnt_done(cnt_done),
        .kv_swap_banks(kv_swap_banks), .q_we(q_we), .kv_we(kv_we),
        .array_start(array_start), .array_done(array_done),
        .softmax_tile_start(softmax_tile_start),
        .softmax_tile_valid(softmax_tile_valid),
        .softmax_tile_last(softmax_tile_last),
        .softmax_out_valid(softmax_out_valid),
        .accum_en(accum_en), .norm_en(norm_en),
        .dbg_state(dbg_state)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] q_global_offset, k_global_offset;
    logic [31:0] dummy_v_off;
    /* verilator lint_on UNUSEDSIGNAL */

    addr_gen #(.MAX_SRAM_DEPTH(4096)) u_addr_gen (
        .clk(clk), .rst_n(rst_n),
        .seq_len(16'(SEQ_LEN)), .head_dim(16'(HEAD_DIM)), .tile_size(16'(TILE_SIZE)),
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr),
        .sram_addr(sram_addr_wide), .cnt_done(cnt_done),
        .q_global_offset(q_global_offset),
        .k_global_offset(k_global_offset),
        .v_global_offset(dummy_v_off)
    );
    assign sram_addr = sram_addr_wide[7:0];

    // Global SRAM read addresses (12-bit, sufficient for SRAM_DEPTH=4096)
    logic [11:0] q_global_rd_addr;   // Q SRAM read = tile_row×HEAD_DIM + local
    logic [11:0] kv_global_rd_addr;  // K/V SRAM read = tile_col×HEAD_DIM + local
    logic [11:0] out_global_addr;    // output buffer = same as Q global addr
    assign q_global_rd_addr  = q_global_offset[11:0] + {4'b0, sram_addr};
    assign kv_global_rd_addr = k_global_offset[11:0] + {4'b0, sram_addr};
    assign out_global_addr   = q_global_offset[11:0] + {4'b0, sram_addr};

    // =========================================================
    // 2. Byte-addressable flat SRAMs (DATA_WIDTH=8, DEPTH=4096)
    //    No ping-pong: testbench writes all Q/K/V before start.
    //    Reads use global tile offsets for correct multi-tile indexing.
    // =========================================================
    logic [7:0] q_rdata, k_rdata, v_rdata;

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_q_buf (
        .clk(clk),
        .we(q_we_ext),  .waddr(q_waddr_ext), .wdata(q_wdata_ext),
        .re(1'b1),      .raddr(q_global_rd_addr), .rdata(q_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_k_buf (
        .clk(clk),
        .we(kv_we_ext), .waddr(k_waddr_ext), .wdata(k_wdata_ext),
        .re(1'b1),      .raddr(kv_global_rd_addr), .rdata(k_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_v_buf (
        .clk(clk),
        .we(kv_we_ext), .waddr(v_waddr_ext), .wdata(v_wdata_ext),
        .re(1'b1),      .raddr(kv_global_rd_addr), .rdata(v_rdata)
    );

    // =========================================================
    // 3. Tile Registers (filled during LOAD, read during MATMUL)
    // =========================================================
    logic signed [7:0] Q_reg [FLAT-1:0];
    logic signed [7:0] K_reg [FLAT-1:0];
    logic signed [7:0] V_reg [FLAT-1:0];

    // Delay 1 cycle to compensate SRAM registered read latency
    logic [7:0] reg_wr_addr;
    logic       q_we_d, kv_we_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_wr_addr <= '0;
            q_we_d      <= 1'b0;
            kv_we_d     <= 1'b0;
        end else begin
            reg_wr_addr <= sram_addr;   // local index (0..255), 1-cycle delayed
            q_we_d      <= q_we;
            kv_we_d     <= kv_we;
        end
    end

    // K is stored transposed: K_reg[col*SIZE+row] = K_tile[row][col]
    // Local addr = row*SIZE + col → transposed = col*SIZE + row
    // For SIZE=16: swap the two 4-bit nibbles of the 8-bit address
    localparam int LOG2_SIZE = $clog2(SIZE);
    logic [7:0] k_trans_wr_addr;
    assign k_trans_wr_addr = {reg_wr_addr[LOG2_SIZE-1:0],
                               reg_wr_addr[2*LOG2_SIZE-1:LOG2_SIZE]};

    always_ff @(posedge clk) begin
        if (q_we_d)  Q_reg[reg_wr_addr]       <= signed'(q_rdata);
        if (kv_we_d) begin
            K_reg[k_trans_wr_addr] <= signed'(k_rdata);   // store K transposed
            V_reg[reg_wr_addr]     <= signed'(v_rdata);
        end
    end

    // =========================================================
    // 4. Systolic Array + PV Phase Mux
    // =========================================================
    logic is_pv_phase;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                         is_pv_phase <= 1'b0;
        else if (softmax_out_valid)         is_pv_phase <= 1'b1;
        else if (array_done && is_pv_phase) is_pv_phase <= 1'b0;
    end

    logic signed [7:0]  p_matrix_int8 [FLAT-1:0];
    logic signed [7:0]  array_a_in    [FLAT-1:0];
    logic signed [7:0]  array_b_in    [FLAT-1:0];
    logic signed [31:0] array_acc     [FLAT-1:0];

    for (genvar gi = 0; gi < FLAT; gi++) begin : gen_mux
        assign array_a_in[gi] = is_pv_phase ? p_matrix_int8[gi] : Q_reg[gi];
        assign array_b_in[gi] = is_pv_phase ? V_reg[gi]         : K_reg[gi];
    end

    array_controller #(.SIZE(SIZE)) u_array_ctrl (
        .clk(clk), .rst_n(rst_n),
        .a_flat(array_a_in), .b_flat(array_b_in),
        .start(array_start), .busy(array_busy), .done(array_done),
        .acc(array_acc)
    );

    // =========================================================
    // 5. Dequantizers (FLAT parallel)
    // =========================================================
    logic signed [15:0] dequant_out   [FLAT-1:0];
    logic               dequant_valid [FLAT-1:0];

    for (genvar gi = 0; gi < FLAT; gi++) begin : gen_dequant
        dequantizer #(.OUT_WIDTH(16), .FRAC_BITS(8)) u_deq (
            .clk(clk), .rst_n(rst_n),
            .valid_in(array_done && !is_pv_phase),
            .data_in(array_acc[gi]),
            .scale_q(scale_q), .scale_k(scale_k),
            .valid_out(dequant_valid[gi]),
            .data_out(dequant_out[gi])
        );
    end

    // =========================================================
    // 6. Online Softmax (SIZE rows parallel)
    // =========================================================
    logic [SIZE*16-1:0] softmax_flat_out [SIZE-1:0];
    logic [SIZE-1:0]    softmax_valid_arr;

    logic softmax_tile_valid_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) softmax_tile_valid_d <= 1'b0;
        else        softmax_tile_valid_d <= softmax_tile_valid;
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic [2:0] sfx_dbg_state [SIZE-1:0];
    /* verilator lint_on UNUSEDSIGNAL */

    for (genvar r = 0; r < SIZE; r++) begin : gen_softmax
        logic [SIZE*16-1:0] row_scores;
        for (genvar c = 0; c < SIZE; c++) begin : gen_pack
            assign row_scores[c*16 +: 16]    = dequant_out[r*SIZE + c] >>> 2; // ×1/sqrt(16)
            assign p_matrix_int8[r*SIZE + c] = signed'(softmax_flat_out[r][c*16 +: 8]);
        end

        online_softmax #(.DIM(SIZE)) u_softmax (
            .clk(clk), .rst_n(rst_n),
            .tile_start(softmax_tile_start),
            .tile_valid(softmax_tile_valid_d),
            .tile_last(softmax_tile_last),
            .scores_flat(row_scores),
            .out_valid(softmax_valid_arr[r]),
            .softmax_flat(softmax_flat_out[r]),
            .dbg_state(sfx_dbg_state[r])
        );
    end

    assign softmax_out_valid  = softmax_valid_arr[0];
    assign dbg_sfx_tile_valid = softmax_tile_valid_d;
    assign dbg_sfx_out_valid  = softmax_out_valid;
    assign dbg_softmax_state  = sfx_dbg_state[0];
    assign dbg_is_pv_phase    = is_pv_phase;
    assign dbg_acc0           = array_acc[0];
    assign dbg_accum_en       = accum_en;
    assign dbg_sram_addr      = sram_addr;

    // =========================================================
    // 7. Output Buffer  (SRAM_DEPTH=4096, global write address)
    // =========================================================
    // accum_data_in = (P_int8 @ V_int8)[sram_addr] × scale_v / 256
    // Accumulated at global addr = tile_row×HEAD_DIM + sram_addr.
    // Multiple KV tiles for the same Q tile all write to the SAME
    // global range, so the output_buffer correctly sums their contributions.
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [47:0] pv_scaled_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [31:0] accum_data_in;
    assign pv_scaled_wide = 48'(signed'(array_acc[sram_addr])) * 48'(signed'(scale_v));
    assign accum_data_in  = $signed(pv_scaled_wide[39:8]);   // >>8 = ÷256

    output_buffer #(.DATA_WIDTH(32), .DEPTH(SRAM_DEPTH)) u_out_buf (
        .clk(clk), .rst_n(rst_n),
        .accum_en(accum_en), .addr(out_global_addr),
        .data_in(accum_data_in),
        .re_ext(o_re_ext), .raddr_ext(o_raddr_ext),
        .rdata_ext(o_rdata_ext)
    );

    // ── Suppress unused warnings ──────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused;
    assign _unused = &{
        q_we, kv_we, norm_en, array_busy,
        kv_swap_banks,           // no ping-pong in v3
        dummy_v_off,
        sram_addr_wide[11:8],
        dequant_valid, softmax_valid_arr[SIZE-1:1],
        1'b0
    };
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
