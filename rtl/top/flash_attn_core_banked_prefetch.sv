// ============================================================
//  flash_attn_core_banked_prefetch.sv — banked core WITH KV double-buffering.
//
//  New variant of flash_attn_core_banked.sv (leaves it untouched).  Restores
//  the baseline-style KV prefetch / ping-pong double buffer on the banked path:
//    - Active  K_reg/V_reg  : read by the systolic-array slicing mux (compute).
//    - Shadow  K_shadow/V_shadow : filled by a prefetch load of the NEXT KV tile
//      while the CURRENT tile is being computed.
//    - On the inner-tile boundary the FSM (tile_controller_banked_prefetch) pulses
//      kv_swap_banks; the core copies shadow → active in one cycle and the loader
//      does not have to re-load the tile from scratchpad.
//
//  ── Shared loader arbitration ────────────────────────────────────────────────
//  There is a SINGLE banked_tile_loader (as in flash_attn_core_banked).  It is
//  driven by EITHER the FSM foreground load (ld_start: Q load, first KV load,
//  non-prefetched reloads) OR a prefetch load (pf_start).  These are mutually
//  exclusive by construction (different FSM states), so one loader suffices:
//     loader.start = ld_start | pf_start
//     loader.mode  = pf_start ? KV : ld_mode
//     loader.base  = pf_start ? (next KV tile offset) : (Q/K foreground offset)
//  The loader's `done` is de-multiplexed by pf_load_active into the FSM's
//  foreground ld_done vs the prefetch pf_done.  Its stripe writes are routed to
//  the shadow (prefetch) or the active Q/K/V registers (foreground) by the same
//  flag.
//
//  ── Swap style ───────────────────────────────────────────────────────────────
//  Uses a COPY-style shadow→active swap (a parallel register copy on
//  kv_swap_banks), matching the proven baseline flash_attn_core / flash_attn_top.
//  NOTE (PPA): a copy swap is two full KV-register banks plus a wide copy mux;
//  the lower-area choice is a ping-pong SELECT bit that re-points the slicing mux
//  instead of copying.  That would change the mux structure, so this first
//  variant keeps the copy style to stay bit-identical in behaviour to baseline;
//  ping-pong select is left as future PPA work.
//
//  Everything from the data-slice mux onward (systolic array, dequant, online
//  softmax, output_buffer) is copied verbatim from flash_attn_core_banked.sv.
//
//  Port list is IDENTICAL to flash_attn_core_banked.sv (drop-in replacement).
// ============================================================
`timescale 1ns/1ps

module flash_attn_core_banked_prefetch #(
    parameter int TILE_SIZE  = 16,
    parameter int HEAD_DIM   = 16,
    parameter int SEQ_LEN    = 16,
    parameter int SRAM_DEPTH = 4096
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    input  logic        mode,       // 0 = prefill (only mode supported here)
    input  logic [15:0] kv_len,

    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // Scalar scratchpad write ports (preload path; e.g. testbench)
    input  logic        q_we,
    input  logic [11:0] q_waddr,
    input  logic [7:0]  q_wdata,
    input  logic        k_we,
    input  logic [11:0] k_waddr,
    input  logic [7:0]  k_wdata,
    input  logic        v_we,
    input  logic [11:0] v_waddr,
    input  logic [7:0]  v_wdata,

    // Vector (16-byte stripe) scratchpad write port (DMA fill path).
    input  logic                       dma_v_we,
    input  logic [1:0]                 dma_v_dst,
    input  logic [11:0]                dma_v_addr,
    input  logic [16*8-1:0]            dma_v_data,

    input  logic [11:0]          out_raddr,
    output logic signed [31:0]   out_rdata,

    input  logic        causal,
    input  logic [15:0] kv_tiles_ready
);

    localparam int SIZE       = TILE_SIZE;
    localparam int FLAT       = SIZE * SIZE;
    localparam int KV_FLAT    = SIZE * HEAD_DIM;
    localparam int NUM_CHUNKS = HEAD_DIM / TILE_SIZE;
    localparam int SRAM_ADDR_W = $clog2(SRAM_DEPTH);
    localparam int LOG2_T     = $clog2(TILE_SIZE);
    localparam int CHUNK_W    = (NUM_CHUNKS > 1) ? $clog2(NUM_CHUNKS) : 1;
    localparam int SCALE_SHIFT = $clog2(HEAD_DIM) / 2;
    localparam int NUM_BANKS  = 16;
    localparam int VEC_W      = NUM_BANKS * 8;
    localparam int IDX_W      = $clog2(KV_FLAT);
    localparam int TILE_BYTES = TILE_SIZE * HEAD_DIM;  // stride to next KV tile

    // =========================================================
    // 1. Tile Controller (banked prefetch variant) + Addr Gen
    // =========================================================
    logic [15:0] tile_row, tile_col;
    logic        cnt_en, cnt_clr, cnt_done;
    logic        ld_start, ld_mode, fsm_ld_done;
    logic        pf_start, kv_swap_banks;
    logic [15:0] pf_tile_col;
    logic        array_start, array_done, array_busy;
    logic        array_no_clear;
    logic        softmax_tile_start, softmax_tile_valid, softmax_tile_last;
    logic        softmax_out_valid;
    logic        accum_en, rescale_en, norm_en;
    logic        short_cnt_mode;
    logic [CHUNK_W-1:0] k_chunk;
    logic        pv_done;
    logic [11:0] sram_addr_w12;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [SRAM_ADDR_W-1:0] sram_addr_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [7:0]  sram_addr;

    logic        pf_valid;   // shadow buffer holds a complete next KV tile

    // Declared here (ahead of the tile_controller instance that reads bit 0)
    // so DC does not flag a non-standard forward reference. Softmax array
    // below drives bits [SIZE-1:0]; see section 6.
    logic [SIZE-1:0]     sfx_exp_valid;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [3:0] _dbg_state;
    /* verilator lint_on UNUSEDSIGNAL */

    tile_controller_banked_prefetch #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM), .SEQ_LEN(SEQ_LEN)
    ) u_fsm (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .mode(mode), .kv_len(kv_len),
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr), .cnt_done(cnt_done),
        .ld_start(ld_start), .ld_mode(ld_mode), .ld_done(fsm_ld_done),
        .pf_start(pf_start), .pf_tile_col(pf_tile_col),
        .pf_rdy(pf_valid), .kv_swap_banks(kv_swap_banks),
        .array_start(array_start), .array_no_clear(array_no_clear),
        .array_done(array_done),
        .softmax_tile_start(softmax_tile_start),
        .softmax_tile_valid(softmax_tile_valid),
        .softmax_tile_last(softmax_tile_last),
        .exp_out_valid(sfx_exp_valid[0]),
        .softmax_out_valid(softmax_out_valid),
        .accum_en(accum_en), .rescale_en(rescale_en), .norm_en(norm_en),
        .short_cnt_mode(short_cnt_mode),
        .k_chunk(k_chunk),
        .pv_done(pv_done),
        .causal(causal),
        .kv_tiles_ready(kv_tiles_ready),
        .dbg_state(_dbg_state)
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
        .sram_addr(sram_addr_w12), .cnt_done(cnt_done),
        .q_global_offset(q_global_offset),
        .k_global_offset(k_global_offset),
        .v_global_offset(dummy_v_off)
    );
    assign sram_addr_wide = SRAM_ADDR_W'(sram_addr_w12);
    assign sram_addr      = sram_addr_w12[7:0];

    // Output address (output_buffer), identical to flash_attn_core_banked
    logic [SRAM_ADDR_W-1:0] out_global_addr;
    assign out_global_addr  = q_global_offset[SRAM_ADDR_W-1:0]
                            + SRAM_ADDR_W'(sram_addr[2*LOG2_T-1:LOG2_T]) * SRAM_ADDR_W'(HEAD_DIM)
                            + SRAM_ADDR_W'(k_chunk) * SRAM_ADDR_W'(TILE_SIZE)
                            + SRAM_ADDR_W'(sram_addr[LOG2_T-1:0]);

    // =========================================================
    // 2. Banked scratchpads (Q, K, V) + shared tile loader
    // =========================================================
    localparam logic MODE_Q  = 1'b0;
    localparam logic MODE_KV = 1'b1;

    logic                 q_r_en, k_r_en, v_r_en;
    logic [11:0]          q_r_addr, k_r_addr, v_r_addr;
    logic [VEC_W-1:0]     q_r_vdata, k_r_vdata, v_r_vdata;

    // Foreground load base (Q offset for Q load, K offset for KV load).
    logic [11:0] ld_base_addr;
    assign ld_base_addr = (ld_mode == MODE_Q) ? q_global_offset[11:0]
                                              : k_global_offset[11:0];
    // Prefetch base: the NEXT KV tile = current K tile offset + one tile stride.
    logic [11:0] pf_base_addr;
    assign pf_base_addr = k_global_offset[11:0] + 12'(TILE_BYTES);

    // ── Shared loader arbitration ────────────────────────────────────────────
    // pf_start and ld_start never coincide (different FSM states).
    logic             loader_start;
    logic             loader_mode;
    logic [11:0]      loader_base;
    logic             loader_done;
    assign loader_start = ld_start | pf_start;
    assign loader_mode  = pf_start ? MODE_KV : ld_mode;
    assign loader_base  = pf_start ? pf_base_addr : ld_base_addr;

    // pf_load_active: high for the duration of a prefetch load. Set on pf_start,
    // cleared when the loader completes.  Used to (a) de-mux loader_done into the
    // FSM's foreground ld_done vs pf_done, and (b) route stripe writes to shadow.
    logic pf_load_active;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)               pf_load_active <= 1'b0;
        else if (pf_start)        pf_load_active <= 1'b1;
        else if (loader_done)     pf_load_active <= 1'b0;
    end

    // De-mux the single loader's done pulse.
    assign fsm_ld_done = loader_done & ~pf_load_active;   // foreground completion
    logic pf_done;
    assign pf_done     = loader_done &  pf_load_active;   // prefetch completion

    // pf_valid: shadow holds a complete next tile.  Set on prefetch completion,
    // cleared on swap / new prefetch / new run.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)              pf_valid <= 1'b0;
        else if (start)          pf_valid <= 1'b0;
        else if (pf_start)       pf_valid <= 1'b0;   // shadow being refilled
        else if (kv_swap_banks)  pf_valid <= 1'b0;   // shadow consumed
        else if (pf_done)        pf_valid <= 1'b1;   // shadow ready
    end

    logic             ld_wr_en;
    logic [IDX_W-1:0] ld_wr_index;
    logic [VEC_W-1:0] q_stripe, k_stripe, v_stripe;
    /* verilator lint_off UNUSEDSIGNAL */
    logic             ld_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    banked_tile_loader #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM),
        .NUM_BANKS(NUM_BANKS), .SP_ADDR_W(12)
    ) u_loader (
        .clk(clk), .rst_n(rst_n),
        .start(loader_start), .mode(loader_mode), .base_addr(loader_base),
        .q_r_en(q_r_en), .q_r_addr(q_r_addr), .q_r_vdata(q_r_vdata),
        .k_r_en(k_r_en), .k_r_addr(k_r_addr), .k_r_vdata(k_r_vdata),
        .v_r_en(v_r_en), .v_r_addr(v_r_addr), .v_r_vdata(v_r_vdata),
        .wr_en(ld_wr_en), .wr_index(ld_wr_index),
        .q_stripe(q_stripe), .k_stripe(k_stripe), .v_stripe(v_stripe),
        .busy(ld_busy), .done(loader_done)
    );

    // Per-scratchpad vector write enable (DMA fill, dst-routed)
    logic q_v_we, k_v_we, v_v_we;
    assign q_v_we = dma_v_we & (dma_v_dst == 2'd0);
    assign k_v_we = dma_v_we & (dma_v_dst == 2'd1);
    assign v_v_we = dma_v_we & (dma_v_dst == 2'd2);

    /* verilator lint_off PINCONNECTEMPTY */
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_q_sp (
        .clk(clk),
        .w_en(q_we | q_v_we), .w_vec(q_v_we),
        .w_addr(q_v_we ? dma_v_addr : q_waddr), .w_sdata(q_wdata), .w_vdata(dma_v_data),
        .r_en(q_r_en), .r_vec(1'b1), .r_addr(q_r_addr), .r_sdata(), .r_vdata(q_r_vdata)
    );
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_k_sp (
        .clk(clk),
        .w_en(k_we | k_v_we), .w_vec(k_v_we),
        .w_addr(k_v_we ? dma_v_addr : k_waddr), .w_sdata(k_wdata), .w_vdata(dma_v_data),
        .r_en(k_r_en), .r_vec(1'b1), .r_addr(k_r_addr), .r_sdata(), .r_vdata(k_r_vdata)
    );
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_v_sp (
        .clk(clk),
        .w_en(v_we | v_v_we), .w_vec(v_v_we),
        .w_addr(v_v_we ? dma_v_addr : v_waddr), .w_sdata(v_wdata), .w_vdata(dma_v_data),
        .r_en(v_r_en), .r_vec(1'b1), .r_addr(v_r_addr), .r_sdata(), .r_vdata(v_r_vdata)
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // =========================================================
    // 3. Tile Registers — active + shadow (double buffer)
    //    Active  : consumed by the slicing mux (compute).
    //    Shadow  : filled by a prefetch load; copied to active on kv_swap_banks.
    // =========================================================
    logic signed [7:0] Q_reg     [KV_FLAT-1:0];
    logic signed [7:0] K_reg     [KV_FLAT-1:0];
    logic signed [7:0] V_reg     [KV_FLAT-1:0];
    logic signed [7:0] K_shadow  [KV_FLAT-1:0];
    logic signed [7:0] V_shadow  [KV_FLAT-1:0];

    // Loader stripe writes:
    //   pf_load_active=1 → prefetch load → shadow K/V (loader is always KV mode).
    //   pf_load_active=0 → foreground load → Q_reg (ld_mode=Q) or K/V_reg (KV).
    // ld_mode is held stable by the FSM for the whole foreground load, so it is a
    // valid routing signal here.
    always_ff @(posedge clk) begin
        if (ld_wr_en) begin
            if (pf_load_active) begin
                for (int i = 0; i < NUM_BANKS; i++) begin
                    K_shadow[int'(ld_wr_index) + i] <= signed'(k_stripe[i*8 +: 8]);
                    V_shadow[int'(ld_wr_index) + i] <= signed'(v_stripe[i*8 +: 8]);
                end
            end else if (ld_mode == MODE_Q) begin
                for (int i = 0; i < NUM_BANKS; i++)
                    Q_reg[int'(ld_wr_index) + i] <= signed'(q_stripe[i*8 +: 8]);
            end else begin
                for (int i = 0; i < NUM_BANKS; i++) begin
                    K_reg[int'(ld_wr_index) + i] <= signed'(k_stripe[i*8 +: 8]);
                    V_reg[int'(ld_wr_index) + i] <= signed'(v_stripe[i*8 +: 8]);
                end
            end
        end
        // Copy-style swap: shadow → active in one cycle (baseline style).
        if (kv_swap_banks && pf_valid) begin
            for (int pi = 0; pi < KV_FLAT; pi++) begin
                K_reg[pi] <= K_shadow[pi];
                V_reg[pi] <= V_shadow[pi];
            end
        end
    end

    // =========================================================
    // 4. Systolic Array + Data Slicing Mux  (verbatim from flash_attn_core_banked)
    // =========================================================
    logic is_pv_phase;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            is_pv_phase <= 1'b0;
        else if (sfx_exp_valid[0])
            is_pv_phase <= 1'b1;
        else if (pv_done)
            is_pv_phase <= 1'b0;
    end

    logic signed [7:0]  p_matrix_int8 [FLAT-1:0];
    logic signed [7:0]  array_a_in    [FLAT-1:0];
    logic signed [7:0]  array_b_in    [FLAT-1:0];
    logic signed [31:0] array_acc     [FLAT-1:0];

    for (genvar gi = 0; gi < FLAT; gi++) begin : gen_mux
        assign array_a_in[gi] = is_pv_phase
            ? p_matrix_int8[gi]
            : Q_reg[(gi/SIZE)*HEAD_DIM + int'(k_chunk)*TILE_SIZE + (gi%SIZE)];
        assign array_b_in[gi] = is_pv_phase
            ? V_reg[(gi/SIZE)*HEAD_DIM + int'(k_chunk)*TILE_SIZE + (gi%SIZE)]
            : K_reg[(gi%SIZE)*HEAD_DIM + int'(k_chunk)*TILE_SIZE + (gi/SIZE)];
    end

    array_controller #(.SIZE(SIZE)) u_array_ctrl (
        .clk(clk), .rst_n(rst_n),
        .a_flat(array_a_in), .b_flat(array_b_in),
        .start(array_start), .no_clear(array_no_clear),
        .a_unsigned(is_pv_phase),
        .busy(array_busy), .done(array_done),
        .acc(array_acc)
    );

    // =========================================================
    // 5. Dequantizers (last QK chunk only)  (verbatim)
    // =========================================================
    logic signed [15:0] dequant_out   [FLAT-1:0];
    logic [FLAT-1:0]    dequant_valid;

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
    // 6. Online Softmax (SIZE rows parallel)  (verbatim)
    // =========================================================
    logic [15:0]         sfx_rescale_q88  [SIZE-1:0];
    logic [31:0]         sfx_running_sum  [SIZE-1:0];
    logic [SIZE*16-1:0]  sfx_exp_flat     [SIZE-1:0];
    logic [SIZE-1:0]     softmax_valid_arr;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [SIZE*16-1:0]  softmax_flat_out [SIZE-1:0];
    logic [2:0]          sfx_dbg_state    [SIZE-1:0];
    logic [SIZE-1:0]     sfx_rescale_valid;
    /* verilator lint_on UNUSEDSIGNAL */

    logic sfx_triggered;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                    sfx_triggered <= 1'b0;
        else if (sfx_exp_valid[0])     sfx_triggered <= 1'b0;
        else if (softmax_tile_valid)   sfx_triggered <= 1'b1;
    end
    logic tile_valid_pulse;
    assign tile_valid_pulse = softmax_tile_valid && !sfx_triggered;

    for (genvar r = 0; r < SIZE; r++) begin : gen_softmax
        logic [SIZE*16-1:0] row_scores;
        for (genvar c = 0; c < SIZE; c++) begin : gen_pack
            logic mask_elem;
            assign mask_elem = causal && ((tile_col > tile_row) ||
                                          ((tile_col == tile_row) && (c > r)));
            assign row_scores[c*16 +: 16] =
                mask_elem ? 16'sh8000 : (dequant_out[r*SIZE + c] >>> SCALE_SHIFT);
            assign p_matrix_int8[r*SIZE + c] = signed'(sfx_exp_flat[r][c*16 +: 8]);
        end

        online_softmax #(.DIM(SIZE)) u_softmax (
            .clk(clk), .rst_n(rst_n),
            .tile_start(softmax_tile_start),
            .tile_valid(tile_valid_pulse),
            .tile_last(softmax_tile_last),
            .scores_flat(row_scores),
            .exp_out_valid(sfx_exp_valid[r]),
            .exp_flat(sfx_exp_flat[r]),
            .rescale_valid(sfx_rescale_valid[r]),
            .rescale_q88(sfx_rescale_q88[r]),
            .running_sum_out(sfx_running_sum[r]),
            .out_valid(softmax_valid_arr[r]),
            .softmax_flat(softmax_flat_out[r]),
            .dbg_state(sfx_dbg_state[r])
        );
    end

    assign softmax_out_valid = softmax_valid_arr[0];

    // =========================================================
    // 7. Output Buffer  (verbatim)
    // =========================================================
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [47:0] pv_scaled_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [31:0] accum_data_in;
    assign pv_scaled_wide = 48'(signed'(array_acc[sram_addr])) * 48'(signed'(scale_v));
    assign accum_data_in  = $signed(pv_scaled_wide[39:8]);

    logic done_latch;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)     done_latch <= 1'b0;
        else if (done)  done_latch <= 1'b1;
        else if (start) done_latch <= 1'b0;
    end

    logic [3:0]  qrow_sel;
    logic [15:0] rescale_q88_sel;
    logic [31:0] norm_divisor_sel;
    assign qrow_sel         = sram_addr[2*LOG2_T-1:LOG2_T];
    assign rescale_q88_sel  = sfx_rescale_q88[qrow_sel];
    assign norm_divisor_sel = sfx_running_sum [qrow_sel];

    output_buffer #(.DATA_WIDTH(32), .DEPTH(SRAM_DEPTH)) u_out_buf (
        .clk(clk), .rst_n(rst_n),
        .accum_en(accum_en),     .addr(out_global_addr[11:0]),         .data_in(accum_data_in),
        .rescale_en(rescale_en), .rescale_addr(out_global_addr[11:0]), .rescale_q88(rescale_q88_sel),
        .norm_en(norm_en),       .norm_addr(out_global_addr[11:0]),    .norm_divisor(norm_divisor_sel),
        .re_ext(done_latch), .raddr_ext(out_raddr),
        .rdata_ext(out_rdata)
    );

    // ── Suppress unused warnings ──────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused;
    assign _unused = &{
        array_busy, ld_busy, dummy_v_off, pf_tile_col,
        dequant_valid, softmax_valid_arr[SIZE-1:1],
        sfx_exp_valid[SIZE-1:1],
        1'b0
    };
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
