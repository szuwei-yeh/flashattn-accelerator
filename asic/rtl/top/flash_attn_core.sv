// ============================================================
//  flash_attn_core.sv — FlashAttention Single-Head Core (v5)
//
//  v5 changes (HEAD_DIM=64 inner-dimension tiling):
//  Same structural changes as flash_attn_top.sv v5:
//  - KV_FLAT = TILE_SIZE * HEAD_DIM; NUM_CHUNKS inner-dim passes.
//  - K stored row-major; K^T extracted at read time via data-slicing mux.
//  - array_no_clear, short_cnt_mode, k_chunk, pv_done from tile_controller.
//  - is_pv_phase cleared via pv_done (all PV chunks done).
//  - Prefetch counter widened to KV_FLAT elements.
//  - Backward compatible: HEAD_DIM=16 (NUM_CHUNKS=1) identical to v4.
// ============================================================
`timescale 1ns/1ps

module flash_attn_core #(
    parameter int TILE_SIZE  = 16,
    parameter int HEAD_DIM   = 16,
    parameter int SEQ_LEN    = 16,
    parameter int SRAM_DEPTH = 4096   // must be >= SEQ_LEN * HEAD_DIM
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

    input  logic        causal,       // 1 = causal (decoder) masking enabled

    // KV cache interface
    input  logic                    kc_write_en,
    input  logic [7:0]              kc_write_ptr,
    input  logic [HEAD_DIM*8-1:0]   kc_k_flat,
    input  logic [HEAD_DIM*8-1:0]   kc_v_flat,
    input  logic [7:0]              kc_read_addr,
    output logic [HEAD_DIM*8-1:0]   kc_k_out,
    output logic [HEAD_DIM*8-1:0]   kc_v_out,
    output logic [8:0]              kc_cache_len
);

    localparam int SIZE       = TILE_SIZE;
    localparam int FLAT       = SIZE * SIZE;
    localparam int KV_FLAT    = SIZE * HEAD_DIM;
    localparam int NUM_CHUNKS = HEAD_DIM / TILE_SIZE;
    localparam int SRAM_ADDR_W = $clog2(SRAM_DEPTH);
    localparam int LOG2_T     = $clog2(TILE_SIZE);
    localparam int CHUNK_W    = (NUM_CHUNKS > 1) ? $clog2(NUM_CHUNKS) : 1;
    localparam int PF_CNT_W   = $clog2(KV_FLAT);   // exact for power-of-2 KV_FLAT
    localparam int SCALE_SHIFT = $clog2(HEAD_DIM) / 2; // 1/√d as right-shift

    // =========================================================
    // 1. Tile Controller + Addr Gen
    // =========================================================
    logic [15:0] tile_row, tile_col;
    logic        cnt_en, cnt_clr, cnt_done;
    logic        kv_swap_banks, fsm_q_we, fsm_kv_we;
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
    logic [SRAM_ADDR_W-1:0] sram_addr_wide;  // kept for documentation; upper bits unused for small d
    /* verilator lint_on UNUSEDSIGNAL */
    logic [7:0]  sram_addr;

    // KV prefetch interface
    logic        kv_prefetch_en;
    logic [15:0] kv_prefetch_col;
    logic        kv_prefetch_rdy;

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
        .kv_prefetch_en(kv_prefetch_en),
        .kv_prefetch_col(kv_prefetch_col),
        .kv_prefetch_rdy(kv_prefetch_rdy),
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

    // Global SRAM read addresses
    logic [SRAM_ADDR_W-1:0] q_global_rd_addr;
    logic [SRAM_ADDR_W-1:0] kv_global_rd_addr;

    // Output address: includes k_chunk offset for HEAD_DIM>TILE_SIZE
    logic [SRAM_ADDR_W-1:0] out_global_addr;
    assign q_global_rd_addr = q_global_offset[SRAM_ADDR_W-1:0] + SRAM_ADDR_W'(sram_addr_w12);
    assign out_global_addr  = q_global_offset[SRAM_ADDR_W-1:0]
                            + SRAM_ADDR_W'(sram_addr[2*LOG2_T-1:LOG2_T]) * SRAM_ADDR_W'(HEAD_DIM)
                            + SRAM_ADDR_W'(k_chunk) * SRAM_ADDR_W'(TILE_SIZE)
                            + SRAM_ADDR_W'(sram_addr[LOG2_T-1:0]);

    // ── Prefetch address ──────────────────────────────────────────────
    logic [15:0]       pf_tile_col;
    logic [PF_CNT_W-1:0] pf_cnt;
    logic              pf_running;
    logic              pf_we_d;
    logic [PF_CNT_W-1:0] pf_cnt_d;
    logic              kv_prefetch_done;
    logic              kv_prefetched;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] pf_k_global_offset;
    /* verilator lint_on UNUSEDSIGNAL */
    assign pf_k_global_offset = 32'(pf_tile_col) * 32'(HEAD_DIM);

    logic [SRAM_ADDR_W-1:0] pf_kv_global_rd_addr;
    assign pf_kv_global_rd_addr = pf_k_global_offset[SRAM_ADDR_W-1:0]
                                 + SRAM_ADDR_W'(pf_cnt);

    logic [SRAM_ADDR_W-1:0] kv_global_rd_addr_base;
    assign kv_global_rd_addr_base = k_global_offset[SRAM_ADDR_W-1:0] + SRAM_ADDR_W'(sram_addr_w12);
    assign kv_global_rd_addr = pf_running ? pf_kv_global_rd_addr : kv_global_rd_addr_base;

    // =========================================================
    // 2. Byte-addressable flat SRAMs
    // =========================================================
    logic [7:0] q_rdata, k_rdata, v_rdata;

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_q_buf (
        .clk(clk),
        .we(q_we),  .waddr(q_waddr), .wdata(q_wdata),
        .re(1'b1),  .raddr(q_global_rd_addr[11:0]), .rdata(q_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_k_buf (
        .clk(clk),
        .we(k_we),  .waddr(k_waddr), .wdata(k_wdata),
        .re(1'b1),  .raddr(kv_global_rd_addr[11:0]), .rdata(k_rdata)
    );

    sram_1r1w #(.DATA_WIDTH(8), .DEPTH(SRAM_DEPTH)) u_v_buf (
        .clk(clk),
        .we(v_we),  .waddr(v_waddr), .wdata(v_wdata),
        .re(1'b1),  .raddr(kv_global_rd_addr[11:0]), .rdata(v_rdata)
    );

    // =========================================================
    // 3. Tile Registers (KV_FLAT elements; K stored row-major)
    // =========================================================
    logic signed [7:0] Q_reg     [KV_FLAT-1:0];
    logic signed [7:0] K_reg     [KV_FLAT-1:0];
    logic signed [7:0] V_reg     [KV_FLAT-1:0];
    logic signed [7:0] K_reg_nxt [KV_FLAT-1:0];
    logic signed [7:0] V_reg_nxt [KV_FLAT-1:0];

    logic                q_we_d, kv_we_d;
    logic [PF_CNT_W-1:0] reg_wr_addr;   // delayed sram address, KV_FLAT-range

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_wr_addr <= '0;
            q_we_d      <= 1'b0;
            kv_we_d     <= 1'b0;
        end else begin
            reg_wr_addr <= PF_CNT_W'(sram_addr_w12);
            q_we_d      <= fsm_q_we;
            kv_we_d     <= fsm_kv_we;
        end
    end

    always_ff @(posedge clk) begin
        if (q_we_d)  Q_reg[reg_wr_addr] <= $signed(q_rdata);
        if (kv_we_d) begin
            K_reg[reg_wr_addr] <= $signed(k_rdata);
            V_reg[reg_wr_addr] <= $signed(v_rdata);
        end
        if (pf_we_d) begin
            K_reg_nxt[pf_cnt_d] <= $signed(k_rdata);
            V_reg_nxt[pf_cnt_d] <= $signed(v_rdata);
        end
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
                if (pf_cnt < PF_CNT_W'(KV_FLAT - 1))
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
        if (!rst_n)
            kv_prefetched <= 1'b0;
        else begin
            if      (kv_swap_banks)    kv_prefetched <= 1'b0;
            else if (kv_prefetch_done) kv_prefetched <= 1'b1;
        end
    end

    assign kv_prefetch_rdy = kv_prefetched;

    // =========================================================
    // 4. Systolic Array + Data Slicing Mux
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
            : Q_reg[(gi/SIZE)*HEAD_DIM + k_chunk*TILE_SIZE + (gi%SIZE)];
        assign array_b_in[gi] = is_pv_phase
            ? V_reg[(gi/SIZE)*HEAD_DIM + k_chunk*TILE_SIZE + (gi%SIZE)]
            : K_reg[(gi%SIZE)*HEAD_DIM + k_chunk*TILE_SIZE + (gi/SIZE)];
    end

    // Packed adapters for Yosys-compatible port connections
    logic [FLAT*8-1:0]  array_a_in_packed;
    logic [FLAT*8-1:0]  array_b_in_packed;
    logic [FLAT*32-1:0] array_acc_packed;
    for (genvar pk = 0; pk < FLAT; pk++) begin : gen_pack_io
        assign array_a_in_packed[pk*8+:8] = array_a_in[pk];
        assign array_b_in_packed[pk*8+:8] = array_b_in[pk];
        assign array_acc[pk]               = $signed(array_acc_packed[pk*32+:32]);
    end

    array_controller #(.SIZE(SIZE)) u_array_ctrl (
        .clk(clk), .rst_n(rst_n),
        .a_flat(array_a_in_packed), .b_flat(array_b_in_packed),
        .a_unsigned(is_pv_phase),
        .start(array_start), .no_clear(array_no_clear),
        .busy(array_busy), .done(array_done),
        .acc(array_acc_packed)
    );

    // =========================================================
    // 5. Dequantizers — TIME-MULTIPLEXED (16 units x 16 passes)
    //    Folds the former 256 parallel dequant units down to 16 (one
    //    score-matrix row per pass) to remove the 256-way scale/valid
    //    broadcast fanout and ~16x the area. Mirrors the PV-phase streaming
    //    style (index array_acc by a counter). Dequant pipeline latency = 2,
    //    so the sequence is 16 feed cycles + 2 drain.
    //    Functionally identical: dequant_out[gi] = dequant(array_acc[gi]).
    // =========================================================
    (* mem2reg *) logic signed [15:0] dequant_out [FLAT-1:0];
    logic dequant_done;                 // 1-cycle pulse: all 256 dequant_out ready

    logic is_last_qk_chunk;
    assign is_last_qk_chunk = (k_chunk == CHUNK_W'(NUM_CHUNKS - 1));

    // Registered scale broadcast (now drives only the 16 folded units)
    logic signed [15:0] scale_q_bcast;
    logic signed [15:0] scale_k_bcast;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scale_q_bcast <= 16'sh0100;
            scale_k_bcast <= 16'sh0100;
        end else begin
            scale_q_bcast <= scale_q;
            scale_k_bcast <= scale_k;
        end
    end

    // ── Pass sequencer: feed one row (16 cols) per cycle, 16 rows ────────────
    logic              deq_trigger;
    assign deq_trigger = array_done && !is_pv_phase && is_last_qk_chunk;

    logic              deq_feeding;                 // high during the 16 feed cycles
    logic [LOG2_T-1:0] deq_pass;                    // 0..SIZE-1 : row being fed
    logic              deq_fv_d1, deq_fv_d2;        // feeding delayed to match pipe (lat 2)
    logic [LOG2_T-1:0] deq_pass_d1, deq_pass_d2;    // pass    delayed to match pipe (lat 2)
    logic              deq_busy;
    assign deq_busy = deq_feeding | deq_fv_d1 | deq_fv_d2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            deq_feeding  <= 1'b0;
            deq_pass     <= '0;
            deq_fv_d1    <= 1'b0;  deq_fv_d2   <= 1'b0;
            deq_pass_d1  <= '0;    deq_pass_d2 <= '0;
            dequant_done <= 1'b0;
        end else begin
            // align feed/pass with the 2-stage dequant pipeline
            deq_fv_d1    <= deq_feeding;   deq_pass_d1 <= deq_pass;
            deq_fv_d2    <= deq_fv_d1;     deq_pass_d2 <= deq_pass_d1;
            dequant_done <= 1'b0;          // default: pulse low

            if (deq_trigger && !deq_busy) begin
                deq_feeding <= 1'b1;
                deq_pass    <= '0;
            end else if (deq_feeding) begin
                if (deq_pass == LOG2_T'(SIZE-1)) deq_feeding <= 1'b0;
                else                             deq_pass    <= deq_pass + 1'b1;
            end

            // last row's outputs land when deq_fv_d2 & pass_d2==SIZE-1; pulse the
            // cycle after so every dequant_out entry is settled before softmax reads
            if (deq_fv_d2 && (deq_pass_d2 == LOG2_T'(SIZE-1)))
                dequant_done <= 1'b1;
        end
    end

    // ── 16 time-multiplexed dequant units (module body unchanged) ────────────
    logic signed [15:0] deq_unit_out [SIZE-1:0];
    logic [SIZE-1:0]    deq_unit_vld;
    for (genvar j = 0; j < SIZE; j++) begin : gen_dequant
        dequantizer #(.OUT_WIDTH(16), .FRAC_BITS(8)) u_deq (
            .clk(clk), .rst_n(rst_n),
            .valid_in (deq_feeding),
            .data_in  (array_acc[deq_pass*SIZE + j]),  // row=deq_pass, col=j
            .scale_q  (scale_q_bcast), .scale_k(scale_k_bcast),
            .valid_out(deq_unit_vld[j]),
            .data_out (deq_unit_out[j])
        );
    end

    // ── Write the 16 unit outputs back into dequant_out[row*SIZE + col] ──────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < FLAT; i++) dequant_out[i] <= '0;
        end else if (deq_fv_d2) begin
            for (int j = 0; j < SIZE; j++)
                dequant_out[deq_pass_d2*SIZE + j] <= deq_unit_out[j];
        end
    end

    // =========================================================
    // 6. Online Softmax (SIZE rows parallel) — true cross-tile
    // =========================================================
    logic [SIZE-1:0]     sfx_exp_valid;
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
    logic tile_valid_pulse;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                    sfx_triggered <= 1'b0;
        else if (sfx_exp_valid[0])     sfx_triggered <= 1'b0;
        else if (tile_valid_pulse)     sfx_triggered <= 1'b1;
    end
    assign tile_valid_pulse = softmax_tile_valid && dequant_done && !sfx_triggered;

    for (genvar r = 0; r < SIZE; r++) begin : gen_softmax
        logic [SIZE*16-1:0] row_scores;
        for (genvar c = 0; c < SIZE; c++) begin : gen_pack
            logic mask_elem;
            assign mask_elem = causal && ((tile_col > tile_row) ||
                                          ((tile_col == tile_row) && (c > r)));
            assign row_scores[c*16 +: 16] =
                mask_elem ? 16'sh8000 : (dequant_out[r*SIZE + c] >>> SCALE_SHIFT);
            assign p_matrix_int8[r*SIZE + c] = $signed(sfx_exp_flat[r][c*16 +: 8]);
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
    // 7. Output Buffer
    // =========================================================
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [47:0] pv_scaled_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [31:0] accum_data_in;
    assign pv_scaled_wide = $signed({{16{array_acc[sram_addr][31]}}, array_acc[sram_addr]}) * $signed({{32{scale_v[15]}}, scale_v});
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

    // =========================================================
    // 8. KV Cache
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
        fsm_q_we, fsm_kv_we, array_busy,
        dummy_v_off,
        deq_unit_vld, softmax_valid_arr[SIZE-1:1],
        sfx_exp_valid[SIZE-1:1],
        1'b0
    };
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
