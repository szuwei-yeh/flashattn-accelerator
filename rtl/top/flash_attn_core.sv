// ============================================================
//  flash_attn_core.sv — FlashAttention Single-Head Core (Week 5)
//
//  Derived from flash_attn_top.sv (Week 4 v3).  Interface changes:
//  - Module renamed from flash_attn_top to flash_attn_core
//  - Separate q_we / k_we / v_we write enables (no more kv_we_ext)
//  - out_raddr / out_rdata replaces o_re_ext / o_raddr_ext / o_rdata_ext
//  - SRAM_DEPTH promoted to a parameter
//  - Debug output ports removed (signals suppressed internally)
//  - done_latch latches done=1 and drives re_ext on output_buffer,
//    allowing AXI master to read output after computation completes
// ============================================================
`timescale 1ns/1ps

module flash_attn_core #(
    parameter int TILE_SIZE  = 16,
    parameter int HEAD_DIM   = 16,
    parameter int SEQ_LEN    = 16,
    parameter int SRAM_DEPTH = 4096
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    input  logic        mode,       // 0 = prefill, 1 = decode
    input  logic [15:0] kv_len,     // runtime KV length for decode mode

    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // Q SRAM write port
    input  logic        q_we,
    input  logic [11:0] q_waddr,
    input  logic [7:0]  q_wdata,

    // K SRAM write port
    input  logic        k_we,
    input  logic [11:0] k_waddr,
    input  logic [7:0]  k_wdata,

    // V SRAM write port
    input  logic        v_we,
    input  logic [11:0] v_waddr,
    input  logic [7:0]  v_wdata,

    // Output buffer external read port (driven by AXI master after done)
    input  logic [11:0]          out_raddr,
    output logic signed [31:0]   out_rdata,

    // KV cache interface (exposed for testbench / AXI loader)
    input  logic                    kc_write_en,
    input  logic [7:0]              kc_write_ptr,
    input  logic [HEAD_DIM*8-1:0]   kc_k_flat,
    input  logic [HEAD_DIM*8-1:0]   kc_v_flat,
    input  logic [7:0]              kc_read_addr,
    output logic [HEAD_DIM*8-1:0]   kc_k_out,
    output logic [HEAD_DIM*8-1:0]   kc_v_out,
    output logic [8:0]              kc_cache_len
);

    localparam int SIZE = TILE_SIZE;
    localparam int FLAT = SIZE * SIZE;   // elements per tile = 256

    // =========================================================
    // 1. Tile Controller + Addr Gen
    // =========================================================
    logic [15:0] tile_row, tile_col;
    logic        cnt_en, cnt_clr, cnt_done;
    logic        kv_swap_banks, fsm_q_we, fsm_kv_we;
    logic        array_start, array_done, array_busy;
    logic        softmax_tile_start, softmax_tile_valid, softmax_tile_last;
    logic        softmax_out_valid;
    logic        accum_en, norm_en;
    logic [11:0] sram_addr_wide;
    logic [7:0]  sram_addr;           // local tile index 0..FLAT-1

    /* verilator lint_off UNUSEDSIGNAL */
    logic [3:0] _dbg_state;
    /* verilator lint_on UNUSEDSIGNAL */

    tile_controller #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM), .SEQ_LEN(SEQ_LEN)
    ) u_fsm (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .mode(mode), .kv_len(kv_len),
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr), .cnt_done(cnt_done),
        .kv_swap_banks(kv_swap_banks), .q_we(fsm_q_we), .kv_we(fsm_kv_we),
        .array_start(array_start), .array_done(array_done),
        .softmax_tile_start(softmax_tile_start),
        .softmax_tile_valid(softmax_tile_valid),
        .softmax_tile_last(softmax_tile_last),
        .softmax_out_valid(softmax_out_valid),
        .accum_en(accum_en), .norm_en(norm_en),
        .dbg_state(_dbg_state)
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

    // Global SRAM read addresses (12-bit)
    logic [11:0] q_global_rd_addr;
    logic [11:0] kv_global_rd_addr;
    logic [11:0] out_global_addr;
    assign q_global_rd_addr  = q_global_offset[11:0] + {4'b0, sram_addr};
    assign kv_global_rd_addr = k_global_offset[11:0] + {4'b0, sram_addr};
    assign out_global_addr   = q_global_offset[11:0] + {4'b0, sram_addr};

    // =========================================================
    // 2. Byte-addressable flat SRAMs (DATA_WIDTH=8, DEPTH=SRAM_DEPTH)
    //    Testbench / AXI slave writes Q/K/V before start.
    //    Reads use global tile offsets for correct multi-tile indexing.
    // =========================================================
    logic [7:0] q_rdata, k_rdata, v_rdata;

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_q_buf (
        .clk(clk),
        .we(q_we),  .waddr(q_waddr), .wdata(q_wdata),
        .re(1'b1),  .raddr(q_global_rd_addr), .rdata(q_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_k_buf (
        .clk(clk),
        .we(k_we),  .waddr(k_waddr), .wdata(k_wdata),
        .re(1'b1),  .raddr(kv_global_rd_addr), .rdata(k_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_v_buf (
        .clk(clk),
        .we(v_we),  .waddr(v_waddr), .wdata(v_wdata),
        .re(1'b1),  .raddr(kv_global_rd_addr), .rdata(v_rdata)
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
            q_we_d      <= fsm_q_we;
            kv_we_d     <= fsm_kv_we;
        end
    end

    // K is stored transposed: swap the two 4-bit nibbles of the 8-bit address
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

    assign softmax_out_valid = softmax_valid_arr[0];

    // =========================================================
    // 7. Output Buffer  (SRAM_DEPTH=4096, global write address)
    // =========================================================
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [47:0] pv_scaled_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [31:0] accum_data_in;
    assign pv_scaled_wide = 48'(signed'(array_acc[sram_addr])) * 48'(signed'(scale_v));
    assign accum_data_in  = $signed(pv_scaled_wide[39:8]);   // >>8 = ÷256

    // done_latch: set high when computation completes, enabling AXI master reads.
    // Cleared on the next start pulse.
    logic done_latch;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)     done_latch <= 1'b0;
        else if (done)  done_latch <= 1'b1;
        else if (start) done_latch <= 1'b0;
    end

    output_buffer #(.DATA_WIDTH(32), .DEPTH(SRAM_DEPTH)) u_out_buf (
        .clk(clk), .rst_n(rst_n),
        .accum_en(accum_en), .addr(out_global_addr),
        .data_in(accum_data_in),
        .re_ext(done_latch), .raddr_ext(out_raddr),
        .rdata_ext(out_rdata)
    );

    // =========================================================
    // 8. KV Cache (persistent K/V history for decode mode)
    // =========================================================
    kv_cache #(.MAX_SEQ_LEN(256), .HEAD_DIM(HEAD_DIM)) u_kv_cache (
        .clk       (clk),
        .rst_n     (rst_n),
        .write_en  (kc_write_en),
        .write_ptr (kc_write_ptr),
        .k_new_flat(kc_k_flat),
        .v_new_flat(kc_v_flat),
        .read_addr (kc_read_addr),
        .k_out_flat(kc_k_out),
        .v_out_flat(kc_v_out),
        .cache_len (kc_cache_len)
    );

    // ── Suppress unused warnings ──────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused;
    assign _unused = &{
        fsm_q_we, fsm_kv_we, norm_en, array_busy,
        kv_swap_banks,           // no ping-pong in v3
        dummy_v_off,
        sram_addr_wide[11:8],
        dequant_valid, softmax_valid_arr[SIZE-1:1],
        1'b0
    };
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
