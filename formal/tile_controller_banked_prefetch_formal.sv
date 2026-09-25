`timescale 1ns/1ps

// Small formal boundary for scheduler/residency/prefetch safety.  Datapath
// blocks are intentionally replaced by legal nondeterministic completion
// models; no arithmetic, SRAM, or softmax implementation is instantiated.
module tile_controller_banked_prefetch_formal;
    localparam int TILE_SIZE = 16;
    localparam int HEAD_DIM  = 16;
    localparam int SEQ_LEN   = 64;
    localparam int LOG2_T    = $clog2(TILE_SIZE);

    reg clk = 1'b0;
    always @($global_clock)
        clk <= !clk;

    reg rst_n = 1'b0;
    always @(posedge clk)
        rst_n <= 1'b1;

    (* anyseq *) logic        start;
    (* anyseq *) logic        causal;
    (* anyseq *) logic [15:0] kv_tiles_ready;

    // The synthesized DMA-banked-prefetch top uses the controller in prefill
    // mode.  Decode-mode scheduling is intentionally outside this proof scope.
    wire        mode   = 1'b0;
    wire [15:0] kv_len = 16'(SEQ_LEN);

    (* anyseq *) logic f_finish_load;
    (* anyseq *) logic f_finish_prefetch;
    (* anyseq *) logic f_finish_array;
    (* anyseq *) logic f_finish_softmax;
    (* anyseq *) logic f_finish_counter;

    logic done, start_accepted;
    logic [15:0] tile_row, tile_col;
    logic cnt_en, cnt_clr, cnt_done;
    logic ld_start, ld_mode, ld_done;
    logic pf_start, pf_rdy, kv_swap_banks;
    logic [15:0] pf_tile_col;
    logic array_start, array_no_clear, array_done;
    logic softmax_tile_start, softmax_tile_valid, softmax_tile_last;
    logic exp_out_valid;
    logic fused_update_en, norm_en, short_cnt_mode;
    logic pv_done;
    logic [0:0] k_chunk;
    logic [3:0] dbg_state;

    localparam logic [3:0] S_IDLE        = 4'd0;
    localparam logic [3:0] S_LOAD_KV     = 4'd2;

    // Legal request/completion models.  A response can be delayed for any
    // finite number of bounded-proof cycles, but cannot arrive before request.
    logic f_load_pending, f_array_pending;
    logic f_prefetch_pending, f_shadow_valid;
    logic [15:0] f_shadow_tile;

    assign ld_done       = f_load_pending     && f_finish_load;
    assign array_done    = f_array_pending    && f_finish_array;
    assign exp_out_valid = softmax_tile_valid && f_finish_softmax;
    assign cnt_done      = cnt_en             && f_finish_counter;
    assign pf_rdy        = f_shadow_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            f_load_pending     <= 1'b0;
            f_array_pending    <= 1'b0;
            f_prefetch_pending <= 1'b0;
            f_shadow_valid     <= 1'b0;
            f_shadow_tile      <= '0;
        end else begin
            if (ld_start)
                f_load_pending <= 1'b1;
            else if (ld_done)
                f_load_pending <= 1'b0;

            if (array_start)
                f_array_pending <= 1'b1;
            else if (array_done)
                f_array_pending <= 1'b0;

            if (start_accepted) begin
                f_prefetch_pending <= 1'b0;
                f_shadow_valid     <= 1'b0;
            end else if (pf_start) begin
                f_prefetch_pending <= 1'b1;
                f_shadow_valid     <= 1'b0;
                f_shadow_tile      <= pf_tile_col >> LOG2_T;
            end else if (f_prefetch_pending && f_finish_prefetch) begin
                f_prefetch_pending <= 1'b0;
                f_shadow_valid     <= 1'b1;
            end else if (kv_swap_banks) begin
                f_shadow_valid <= 1'b0;
            end
        end
    end

    tile_controller_banked_prefetch #(
        .TILE_SIZE(TILE_SIZE),
        .HEAD_DIM (HEAD_DIM),
        .SEQ_LEN  (SEQ_LEN)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .start_accepted(start_accepted),
        .mode(mode), .kv_len(kv_len),
        .tile_row(tile_row), .tile_col(tile_col),
        .cnt_en(cnt_en), .cnt_clr(cnt_clr), .cnt_done(cnt_done),
        .ld_start(ld_start), .ld_mode(ld_mode), .ld_done(ld_done),
        .pf_start(pf_start), .pf_tile_col(pf_tile_col),
        .pf_rdy(pf_rdy), .kv_swap_banks(kv_swap_banks),
        .array_start(array_start), .array_no_clear(array_no_clear),
        .array_done(array_done),
        .softmax_tile_start(softmax_tile_start),
        .softmax_tile_valid(softmax_tile_valid),
        .softmax_tile_last(softmax_tile_last),
        .exp_out_valid(exp_out_valid), .softmax_out_valid(1'b0),
        .fused_update_en(fused_update_en), .norm_en(norm_en),
        .short_cnt_mode(short_cnt_mode), .k_chunk(k_chunk),
        .pv_done(pv_done), .causal(causal),
        .kv_tiles_ready(kv_tiles_ready), .dbg_state(dbg_state)
    );

    wire [15:0] effective_kv_len = mode ? kv_len : 16'(SEQ_LEN);
    wire [15:0] configured_tiles = effective_kv_len >> LOG2_T;
    wire [15:0] current_tile_idx = tile_col >> LOG2_T;
    wire [15:0] prefetch_tile_idx = pf_tile_col >> LOG2_T;

    // The causal mode may be configured in IDLE and remains stable for an
    // accepted transaction.  The resident count is bounded by four N=64 tiles.
    logic f_causal_latched;
    always @(posedge clk) begin
        if (!rst_n) begin
            f_causal_latched <= 1'b0;
        end else if (start_accepted) begin
            f_causal_latched <= causal;
        end

        if (rst_n) begin
            assume(kv_tiles_ready <= configured_tiles);

            if (dbg_state != S_IDLE)
                assume(causal == f_causal_latched);
        end
    end

    // Residency is a count of completed DMA tile pairs.  It may grow while a
    // transaction runs, but resident data cannot disappear mid-transaction.
    always @(posedge clk) begin
        if (rst_n && $past(rst_n) &&
            dbg_state != S_IDLE && $past(dbg_state) != S_IDLE)
            assume(kv_tiles_ready >= $past(kv_tiles_ready));
    end

    // Verification-only abstract state corresponding to complete active K/V
    // registers and the tile identity they contain.
    logic        f_active_valid;
    logic [15:0] f_active_tile;
    always @(posedge clk) begin
        if (!rst_n || start_accepted) begin
            f_active_valid <= 1'b0;
            f_active_tile  <= '0;
        end else begin
            if (ld_done && dbg_state == S_LOAD_KV) begin
                f_active_valid <= 1'b1;
                f_active_tile  <= current_tile_idx;
            end
            if (kv_swap_banks && f_shadow_valid) begin
                f_active_valid <= 1'b1;
                f_active_tile  <= f_shadow_tile;
            end
        end
    end

    // Property 1: every QK/PV array launch consumes a complete, resident active
    // K/V tile, and that active tile is the controller's current tile.
    always @(posedge clk) begin : p_compute_uses_ready_kv
        if (rst_n) begin
            assert(!(ld_start && pf_start));
            if (ld_start) assert(!f_prefetch_pending && !f_load_pending);
            if (pf_start) assert(!f_prefetch_pending && !f_load_pending);
        end
        if (rst_n && array_start) begin
            assert(f_active_valid);
            assert(f_active_tile == current_tile_idx);
            assert(current_tile_idx < kv_tiles_ready);
        end
    end

    // Property 2: the controller may promote shadow to active only after the
    // modeled shared loader has completed that exact next tile.
    always @(posedge clk) begin : p_shadow_promotion_is_valid
        if (rst_n && kv_swap_banks) begin
            assert(f_shadow_valid);
            assert(f_shadow_tile == current_tile_idx);
        end
    end

    // Property 3: all compute and prefetch activity remains inside the legal,
    // configured tile range.  Idle/done values are intentionally not restricted.
    always @(posedge clk) begin : p_tile_indices_in_bounds
        if (rst_n && array_start) begin
            assert(current_tile_idx < configured_tiles);
            assert(tile_col[LOG2_T-1:0] == '0);
        end
        if (rst_n && pf_start) begin
            assert(prefetch_tile_idx < configured_tiles);
            assert(pf_tile_col[LOG2_T-1:0] == '0);
        end
    end

    // Reachability checks prevent a vacuous proof: exercise foreground compute,
    // a prefetch launch, and a valid shadow promotion.
    always @(posedge clk) begin
        cover(rst_n && array_start);
        cover(rst_n && pf_start);
        cover(rst_n && kv_swap_banks);
    end
endmodule
