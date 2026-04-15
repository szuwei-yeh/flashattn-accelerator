// ============================================================
//  flash_attn_top.sv — FlashAttention Accelerator Top Level (v5)
//
//  v5 changes (HEAD_DIM=64 inner-dimension tiling):
//  - KV_FLAT = TILE_SIZE * HEAD_DIM (register size scales with HEAD_DIM).
//  - NUM_CHUNKS = HEAD_DIM / TILE_SIZE inner-dimension passes per KV tile.
//  - K stored row-major (no nibble-swap); K^T extracted at read time.
//  - Data-slicing mux: Q/K/V reads use k_chunk offset into HEAD_DIM.
//  - array_no_clear: QK chunks 1+ skip PE CLEAR to accumulate.
//  - short_cnt_mode: RESCALE/ACCUMULATE/NORMALIZE count TILE_SIZE² only.
//  - is_pv_phase cleared via pv_done (fires after ALL PV chunks complete).
//  - Prefetch counter widened to KV_FLAT elements.
//  - SRAM_DEPTH kept at 4096 (fits N*d ≤ 4096: N=256×d=16 or N=64×d=64).
//  - Backward compatible: HEAD_DIM=16 (NUM_CHUNKS=1) identical to v4.
// ============================================================
`timescale 1ns/1ps

module flash_attn_top #(
    parameter  int TILE_SIZE   = 16,
    parameter  int HEAD_DIM    = 16,
    parameter  int SEQ_LEN     = 16,
    localparam int SRAM_DEPTH  = SEQ_LEN * HEAD_DIM,   // auto-size: N*d entries
    localparam int SRAM_ADDR_W = $clog2(SRAM_DEPTH)    // e.g. 12 for 4096, 14 for 16384
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    input  logic                   q_we_ext,
    input  logic [SRAM_ADDR_W-1:0] q_waddr_ext,
    input  logic [7:0]             q_wdata_ext,

    input  logic                   kv_we_ext,
    input  logic [SRAM_ADDR_W-1:0] k_waddr_ext,
    input  logic [7:0]             k_wdata_ext,
    input  logic [SRAM_ADDR_W-1:0] v_waddr_ext,
    input  logic [7:0]             v_wdata_ext,

    input  logic                   o_re_ext,
    input  logic [SRAM_ADDR_W-1:0] o_raddr_ext,
    output logic signed [31:0]     o_rdata_ext,

    input  logic               causal,

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
    localparam int FLAT       = SIZE * SIZE;               // systolic array size: always 256
    localparam int KV_FLAT    = SIZE * HEAD_DIM;           // elements per Q/K/V tile register
    localparam int NUM_CHUNKS = HEAD_DIM / TILE_SIZE;      // inner-dim passes
    localparam int LOG2_T     = $clog2(TILE_SIZE);         // = 4 for SIZE=16
    localparam int CHUNK_W    = (NUM_CHUNKS > 1) ? $clog2(NUM_CHUNKS) : 1;
    localparam int KV_IDX_W   = $clog2(KV_FLAT);          // bits to index KV_FLAT elements
    // 1/√HEAD_DIM implemented as arithmetic right-shift by SCALE_SHIFT bits.
    // HEAD_DIM=16: √16=4=2², shift=2.  HEAD_DIM=64: √64=8=2³, shift=3.
    localparam int SCALE_SHIFT = $clog2(HEAD_DIM) / 2;   // e.g. 4/2=2, 6/2=3

    // =========================================================
    // 1. Tile Controller + Addr Gen
    // =========================================================
    logic [15:0] tile_row, tile_col;
    logic        cnt_en, cnt_clr, cnt_done;
    logic        kv_swap_banks, q_we, kv_we;
    logic        array_start, array_done, array_busy;
    logic        array_no_clear;
    logic        softmax_tile_start, softmax_tile_valid, softmax_tile_last;
    logic        softmax_out_valid;
    logic        exp_out_valid_top;
    logic        accum_en, rescale_en, norm_en;
    logic        short_cnt_mode;
    logic [CHUNK_W-1:0] k_chunk;
    logic        pv_done;
    logic [SRAM_ADDR_W-1:0] sram_addr_raw;     // addr_gen output (SRAM_ADDR_W bits)
    logic [7:0]  sram_addr;                    // lower 8 bits: row*T+col in short_cnt_mode

    // KV prefetch
    logic        kv_prefetch_en;
    logic [15:0] kv_prefetch_col;
    logic        kv_prefetch_rdy;

    tile_controller #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM), .SEQ_LEN(SEQ_LEN)
    ) u_fsm (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .mode(1'b0), .kv_len(16'b0),
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr), .cnt_done(cnt_done),
        .kv_swap_banks(kv_swap_banks), .q_we(q_we), .kv_we(kv_we),
        .array_start(array_start), .array_no_clear(array_no_clear),
        .array_done(array_done),
        .softmax_tile_start(softmax_tile_start),
        .softmax_tile_valid(softmax_tile_valid),
        .softmax_tile_last(softmax_tile_last),
        .exp_out_valid(exp_out_valid_top),
        .softmax_out_valid(softmax_out_valid),
        .accum_en(accum_en), .rescale_en(rescale_en), .norm_en(norm_en),
        .short_cnt_mode(short_cnt_mode),
        .k_chunk(k_chunk),
        .pv_done(pv_done),
        .causal(causal),
        .kv_prefetch_en(kv_prefetch_en),
        .kv_prefetch_col(kv_prefetch_col),
        .kv_prefetch_rdy(kv_prefetch_rdy),
        .dbg_state(dbg_state)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] q_global_offset, k_global_offset;
    logic [31:0] dummy_v_off;
    /* verilator lint_on UNUSEDSIGNAL */

    addr_gen #(.MAX_SRAM_DEPTH(SRAM_DEPTH)) u_addr_gen (
        .clk(clk), .rst_n(rst_n),
        .seq_len(16'(SEQ_LEN)), .head_dim(16'(HEAD_DIM)), .tile_size(16'(TILE_SIZE)),
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr),
        .short_cnt_mode(short_cnt_mode),
        .sram_addr(sram_addr_raw), .cnt_done(cnt_done),
        .q_global_offset(q_global_offset),
        .k_global_offset(k_global_offset),
        .v_global_offset(dummy_v_off)
    );
    assign sram_addr = sram_addr_raw[7:0];

    // ── Global SRAM read addresses ────────────────────────────────────
    logic [SRAM_ADDR_W-1:0] q_global_rd_addr;
    logic [SRAM_ADDR_W-1:0] kv_global_rd_addr;

    // Output address within Q-tile: row*HEAD_DIM + k_chunk*T + col
    //   row = sram_addr[2*LOG2_T-1:LOG2_T], col = sram_addr[LOG2_T-1:0]
    //   (valid during short_cnt_mode where sram_addr counts 0..FLAT-1)
    logic [SRAM_ADDR_W-1:0] out_global_addr;
    assign q_global_rd_addr = q_global_offset[SRAM_ADDR_W-1:0] + sram_addr_raw;
    assign out_global_addr  = q_global_offset[SRAM_ADDR_W-1:0]
                            + SRAM_ADDR_W'(sram_addr[2*LOG2_T-1:LOG2_T]) * SRAM_ADDR_W'(HEAD_DIM)
                            + SRAM_ADDR_W'(k_chunk) * SRAM_ADDR_W'(TILE_SIZE)
                            + SRAM_ADDR_W'(sram_addr[LOG2_T-1:0]);

    // ── Prefetch address ──────────────────────────────────────────────
    logic [15:0] pf_tile_col;
    // Counter counts 0..KV_FLAT-1; for power-of-2 KV_FLAT, $clog2(KV_FLAT) bits suffice
    localparam int PF_CNT_W = KV_IDX_W;
    logic [PF_CNT_W-1:0] pf_cnt;
    logic        pf_running;
    logic        pf_we_d;
    logic [PF_CNT_W-1:0] pf_cnt_d;
    logic        kv_prefetch_done;
    logic        kv_prefetched;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] pf_k_global_offset;
    /* verilator lint_on UNUSEDSIGNAL */
    assign pf_k_global_offset = 32'(pf_tile_col) * 32'(HEAD_DIM);

    logic [SRAM_ADDR_W-1:0] pf_kv_global_rd_addr;
    assign pf_kv_global_rd_addr = pf_k_global_offset[SRAM_ADDR_W-1:0] + SRAM_ADDR_W'(pf_cnt);

    // Mux: redirect SRAM read to prefetch tile during prefetch
    logic [SRAM_ADDR_W-1:0] kv_global_rd_addr_base;
    assign kv_global_rd_addr_base = k_global_offset[SRAM_ADDR_W-1:0] + sram_addr_raw;
    assign kv_global_rd_addr = pf_running ? pf_kv_global_rd_addr : kv_global_rd_addr_base;

    // =========================================================
    // 2. Byte-addressable flat SRAMs (DATA_WIDTH=8, DEPTH=SEQ_LEN*HEAD_DIM)
    // =========================================================
    logic [7:0] q_rdata, k_rdata, v_rdata;

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_q_buf (
        .clk(clk),
        .we(q_we_ext), .waddr(q_waddr_ext), .wdata(q_wdata_ext),
        .re(1'b1),     .raddr(q_global_rd_addr),   .rdata(q_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_k_buf (
        .clk(clk),
        .we(kv_we_ext), .waddr(k_waddr_ext), .wdata(k_wdata_ext),
        .re(1'b1),      .raddr(kv_global_rd_addr),  .rdata(k_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_v_buf (
        .clk(clk),
        .we(kv_we_ext), .waddr(v_waddr_ext), .wdata(v_wdata_ext),
        .re(1'b1),      .raddr(kv_global_rd_addr),  .rdata(v_rdata)
    );

    // =========================================================
    // 3. Tile Registers (KV_FLAT elements for HEAD_DIM-wide tiles)
    //    K stored row-major; K^T obtained by swapping indices at read.
    //    _nxt = prefetch double-buffer.
    // =========================================================
    logic signed [7:0] Q_reg     [KV_FLAT-1:0];
    logic signed [7:0] K_reg     [KV_FLAT-1:0];
    logic signed [7:0] V_reg     [KV_FLAT-1:0];
    logic signed [7:0] K_reg_nxt [KV_FLAT-1:0];
    logic signed [7:0] V_reg_nxt [KV_FLAT-1:0];

    // 1-cycle delayed write signals (SRAM registered read latency)
    logic [KV_IDX_W-1:0] reg_wr_addr;   // sram_addr_raw truncated to KV_FLAT index width
    logic                 q_we_d, kv_we_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_wr_addr <= '0;
            q_we_d      <= 1'b0;
            kv_we_d     <= 1'b0;
        end else begin
            reg_wr_addr <= KV_IDX_W'(sram_addr_raw);
            q_we_d      <= q_we;
            kv_we_d     <= kv_we;
        end
    end

    always_ff @(posedge clk) begin
        // Normal SRAM load: writes 0..KV_FLAT-1 sequentially
        if (q_we_d)  Q_reg[reg_wr_addr] <= signed'(q_rdata);
        if (kv_we_d) begin
            K_reg[reg_wr_addr] <= signed'(k_rdata);   // row-major, no nibble-swap
            V_reg[reg_wr_addr] <= signed'(v_rdata);
        end
        // Prefetch write into shadow buffer
        if (pf_we_d) begin
            K_reg_nxt[pf_cnt_d] <= signed'(k_rdata);
            V_reg_nxt[pf_cnt_d] <= signed'(v_rdata);
        end
        // Swap: copy shadow → active
        if (kv_swap_banks && kv_prefetched) begin
            for (int pi = 0; pi < KV_FLAT; pi++) begin
                K_reg[pi] <= K_reg_nxt[pi];
                V_reg[pi] <= V_reg_nxt[pi];
            end
        end
    end

    // ── Prefetch counter ──────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pf_tile_col <= '0;
            pf_running  <= 1'b0;
            pf_cnt      <= '0;
        end else begin
            if (kv_prefetch_en) begin
                pf_tile_col <= kv_prefetch_col;
                pf_running  <= 1'b1;
                pf_cnt      <= '0;
            end else if (pf_running) begin
                if (pf_cnt != PF_CNT_W'(KV_FLAT - 1))
                    pf_cnt <= pf_cnt + 1'b1;
                else
                    pf_running <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk) begin
        pf_we_d  <= pf_running;
        pf_cnt_d <= pf_cnt;
    end

    assign kv_prefetch_done = pf_we_d && (pf_cnt_d == PF_CNT_W'(KV_FLAT - 1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)             kv_prefetched <= 1'b0;
        else if (kv_swap_banks) kv_prefetched <= 1'b0;
        else if (kv_prefetch_done) kv_prefetched <= 1'b1;
    end

    assign kv_prefetch_rdy = kv_prefetched;

    // =========================================================
    // 4. Systolic Array + Data Slicing Mux
    //
    //  QK phase (is_pv_phase=0), chunk k_chunk (= c):
    //    a_in[r*T + k'] = Q_reg[r*HEAD_DIM + c*T + k']
    //    b_in[k'*T + j] = K_reg[j*HEAD_DIM + c*T + k']   ← K^T on read
    //
    //  PV phase (is_pv_phase=1), chunk k_chunk (= c):
    //    a_in[gi]       = p_matrix_int8[gi]               (unchanged)
    //    b_in[k'*T + j] = V_reg[k'*HEAD_DIM + c*T + j]
    //
    //  With gi = row*T + col: row = gi/T (k' for b), col = gi%T (j for b).
    // =========================================================
    logic is_pv_phase;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)             is_pv_phase <= 1'b0;
        else if (exp_out_valid_top) is_pv_phase <= 1'b1;
        else if (pv_done)           is_pv_phase <= 1'b0;
    end

    logic signed [7:0]  p_matrix_int8 [FLAT-1:0];
    logic signed [7:0]  array_a_in    [FLAT-1:0];
    logic signed [7:0]  array_b_in    [FLAT-1:0];
    logic signed [31:0] array_acc     [FLAT-1:0];

    for (genvar gi = 0; gi < FLAT; gi++) begin : gen_mux
        // QK: a_in[r*T+k'] = Q_reg[r*D + c*T + k'],  r=gi/T, k'=gi%T
        // PV: a_in[gi]      = p_matrix_int8[gi]
        assign array_a_in[gi] = is_pv_phase
            ? p_matrix_int8[gi]
            : Q_reg[(gi/SIZE)*HEAD_DIM + int'(k_chunk)*TILE_SIZE + (gi%SIZE)];
        // QK: b_in[k'*T+j]  = K_reg[j*D + c*T + k'],  k'=gi/T, j=gi%T
        // PV: b_in[k'*T+j]  = V_reg[k'*D + c*T + j]
        assign array_b_in[gi] = is_pv_phase
            ? V_reg[(gi/SIZE)*HEAD_DIM + int'(k_chunk)*TILE_SIZE + (gi%SIZE)]
            : K_reg[(gi%SIZE)*HEAD_DIM + int'(k_chunk)*TILE_SIZE + (gi/SIZE)];
    end

    array_controller #(.SIZE(SIZE)) u_array_ctrl (
        .clk(clk), .rst_n(rst_n),
        .a_flat(array_a_in), .b_flat(array_b_in),
        .start(array_start), .no_clear(array_no_clear),
        .busy(array_busy), .done(array_done),
        .acc(array_acc)
    );

    // =========================================================
    // 5. Dequantizers (triggered only on last QK chunk)
    // =========================================================
    logic signed [15:0] dequant_out   [FLAT-1:0];
    /* verilator lint_off UNUSEDSIGNAL */
    logic               dequant_valid [FLAT-1:0];
    /* verilator lint_on UNUSEDSIGNAL */

    logic is_last_qk_chunk;
    assign is_last_qk_chunk = (k_chunk == CHUNK_W'(NUM_CHUNKS - 1));

    for (genvar gi = 0; gi < FLAT; gi++) begin : gen_dequant
        dequantizer #(.OUT_WIDTH(16), .FRAC_BITS(8)) u_deq (
            .clk(clk), .rst_n(rst_n),
            .valid_in(array_done && !is_pv_phase && is_last_qk_chunk),
            .data_in(array_acc[gi]),
            .scale_q(scale_q), .scale_k(scale_k),
            .valid_out(dequant_valid[gi]),
            .data_out(dequant_out[gi])
        );
    end

    // =========================================================
    // 6. Online Softmax (SIZE rows in parallel)
    // =========================================================
    /* verilator lint_off UNUSEDSIGNAL */
    logic [SIZE*16-1:0] softmax_flat_out [SIZE-1:0];
    logic [SIZE-1:0]    rescale_valid_arr;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [SIZE*16-1:0] exp_flat_out     [SIZE-1:0];
    logic [SIZE-1:0]    softmax_valid_arr;
    logic [SIZE-1:0]    exp_out_valid_arr;
    logic [15:0]        rescale_q88_arr  [SIZE-1:0];
    logic [31:0]        running_sum_arr  [SIZE-1:0];

    logic sfx_triggered;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sfx_triggered <= 1'b0;
        else if (exp_out_valid_top)
            sfx_triggered <= 1'b0;
        else if (softmax_tile_valid)
            sfx_triggered <= 1'b1;
    end
    logic tile_valid_pulse;
    assign tile_valid_pulse = softmax_tile_valid && !sfx_triggered;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [2:0] sfx_dbg_state [SIZE-1:0];
    /* verilator lint_on UNUSEDSIGNAL */

    for (genvar r = 0; r < SIZE; r++) begin : gen_softmax
        logic [SIZE*16-1:0] row_scores;
        for (genvar c = 0; c < SIZE; c++) begin : gen_pack
            logic mask_elem;
            assign mask_elem = causal && ((tile_col > tile_row) ||
                                          ((tile_col == tile_row) && (c > r)));
            assign row_scores[c*16 +: 16] =
                mask_elem ? 16'sh8000 : (dequant_out[r*SIZE + c] >>> SCALE_SHIFT);
            assign p_matrix_int8[r*SIZE + c] = signed'(exp_flat_out[r][c*16 +: 8]);
        end

        online_softmax #(.DIM(SIZE)) u_softmax (
            .clk(clk), .rst_n(rst_n),
            .tile_start(softmax_tile_start),
            .tile_valid(tile_valid_pulse),
            .tile_last(softmax_tile_last),
            .scores_flat(row_scores),
            .exp_out_valid(exp_out_valid_arr[r]),
            .exp_flat(exp_flat_out[r]),
            .rescale_valid(rescale_valid_arr[r]),
            .rescale_q88(rescale_q88_arr[r]),
            .running_sum_out(running_sum_arr[r]),
            .out_valid(softmax_valid_arr[r]),
            .softmax_flat(softmax_flat_out[r]),
            .dbg_state(sfx_dbg_state[r])
        );
    end

    assign exp_out_valid_top = exp_out_valid_arr[0];
    assign softmax_out_valid = softmax_valid_arr[0];

    assign dbg_sfx_tile_valid = softmax_tile_valid;
    assign dbg_sfx_out_valid  = softmax_out_valid;
    assign dbg_softmax_state  = sfx_dbg_state[0];
    assign dbg_is_pv_phase    = is_pv_phase;
    assign dbg_acc0           = array_acc[0];
    assign dbg_accum_en       = accum_en;
    assign dbg_sram_addr      = sram_addr;

    // =========================================================
    // 7. Output Buffer
    //
    //  out_global_addr includes k_chunk offset for HEAD_DIM>TILE_SIZE.
    //  Q-row selector: sram_addr[7:4] (valid in short_cnt_mode, T=16).
    // =========================================================
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [47:0] pv_scaled_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [31:0] accum_data_in;
    assign pv_scaled_wide = 48'(signed'(array_acc[sram_addr])) * 48'(signed'(scale_v));
    assign accum_data_in  = $signed(pv_scaled_wide[39:8]);

    logic [3:0]  qrow_sel;
    logic [15:0] rescale_q88_sel;
    logic [31:0] norm_divisor_sel;
    assign qrow_sel         = sram_addr[2*LOG2_T-1:LOG2_T];
    assign rescale_q88_sel  = rescale_q88_arr [qrow_sel];
    assign norm_divisor_sel = running_sum_arr  [qrow_sel];

    output_buffer #(.DATA_WIDTH(32), .DEPTH(SRAM_DEPTH)) u_out_buf (
        .clk(clk), .rst_n(rst_n),
        .accum_en(accum_en),     .addr(out_global_addr),         .data_in(accum_data_in),
        .rescale_en(rescale_en), .rescale_addr(out_global_addr), .rescale_q88(rescale_q88_sel),
        .norm_en(norm_en),       .norm_addr(out_global_addr),    .norm_divisor(norm_divisor_sel),
        .re_ext(o_re_ext), .raddr_ext(o_raddr_ext), .rdata_ext(o_rdata_ext)
    );

    // ── Suppress unused warnings ──────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused;
    assign _unused = &{
        q_we, kv_we, array_busy,
        dummy_v_off,
        softmax_valid_arr[SIZE-1:1],
        exp_out_valid_arr[SIZE-1:1],
        1'b0
    };
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
