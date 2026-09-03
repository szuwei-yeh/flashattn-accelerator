// ============================================================
//  tile_controller_banked_prefetch.sv — banked FSM WITH KV double-buffering.
//
//  New variant of tile_controller_banked.sv that restores the baseline-style
//  KV prefetch / ping-pong double buffer on the banked-scratchpad path, WITHOUT
//  touching the already-green tile_controller_banked.sv.
//
//  What changed vs tile_controller_banked.sv (everything else is identical):
//    - New handshake to the core:
//        pf_start       : 1-cycle pulse — start prefetching the NEXT KV tile
//                         into the shadow registers (via the shared loader).
//        pf_rdy         : shadow holds a complete next tile (from the core).
//        kv_swap_banks  : 1-cycle pulse — copy shadow → active KV registers.
//    - S_UPDATE_SOFTMAX now fires pf_start (and latches pf_pending) for the next
//      non-skipped, RESIDENT KV tile, exactly like baseline tile_controller.sv.
//    - S_CHECK_INNER decides between: causal-skip spin / prefetch-consume /
//      normal reload — and NEVER falls back to S_LOAD_KV while a prefetch is in
//      flight (the banked path has a SINGLE shared loader, so overlapping a
//      foreground load with a prefetch load would corrupt the active tile).
//    - New S_PF_WAIT state: after advancing tile_col once, wait for pf_rdy, then
//      swap and skip S_LOAD_KV.  Stalling here does NOT re-advance tile_col.
//
//  Residency gate (kv_tiles_ready) semantics are unchanged: a prefetch is only
//  launched for a tile that is already resident; if the next tile is not yet
//  resident, no prefetch fires and S_CHECK_INNER takes the normal S_LOAD_KV path
//  (which stalls on the same residency gate) — correctness is latency-independent.
//
//  This variant also fuses output rescale and PV accumulation into one SRAM
//  traversal.  Its compute flow is QK → softmax → PV → fused update; the fused
//  enable tells output_buffer to preserve the original rescale truncation before
//  adding the scaled PV value.
// ============================================================
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
module tile_controller_banked_prefetch #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 64,
    parameter int SEQ_LEN   = 64,
    localparam int NUM_CHUNKS = HEAD_DIM / TILE_SIZE
)(
/* verilator lint_on UNUSEDPARAM */
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    output logic start_accepted,  // high only while S_IDLE accepts start

    input  logic        mode,       // 0 = prefill, 1 = decode
    input  logic [15:0] kv_len,     // runtime KV length (decode mode)

    output logic [15:0] tile_row,
    output logic [15:0] tile_col,
    output logic        cnt_en,
    output logic        cnt_clr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        cnt_done,
    /* verilator lint_on UNUSEDSIGNAL */

    // Foreground stripe-loader handshake (Q load, first KV load, non-prefetched
    // reloads). Identical to tile_controller_banked.sv.
    output logic        ld_start,   // 1-cycle pulse to start a foreground tile load
    output logic        ld_mode,    // 0 = Q load, 1 = KV load
    input  logic        ld_done,    // FOREGROUND loader finished (core de-muxes)

    // KV prefetch double-buffer handshake (new)
    output logic        pf_start,       // 1-cycle pulse: prefetch next KV tile → shadow
    output logic [15:0] pf_tile_col,    // tile_col of the tile being prefetched
    input  logic        pf_rdy,         // shadow holds a complete next tile
    output logic        kv_swap_banks,  // 1-cycle pulse: copy shadow → active

    output logic        array_start,
    output logic        array_no_clear,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        array_done,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        softmax_tile_start,
    output logic        softmax_tile_valid,
    output logic        softmax_tile_last,
    input  logic        exp_out_valid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        softmax_out_valid,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        fused_update_en,
    output logic        norm_en,
    output logic        short_cnt_mode,
    output logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] k_chunk,
    output logic        pv_done,

    input  logic        causal,

    // DMA streaming interlock (same semantics as tile_controller_banked). Tie to
    // 16'hFFFF when K/V is fully preloaded.
    input  logic [15:0] kv_tiles_ready,

    output logic [3:0]  dbg_state
);

    localparam logic MODE_Q  = 1'b0;
    localparam logic MODE_KV = 1'b1;

    typedef enum logic [3:0] {
        S_IDLE             = 4'd0,
        S_LOAD_Q           = 4'd1,
        S_LOAD_KV          = 4'd2,
        S_MATMUL_QK        = 4'd3,
        S_UPDATE_SOFTMAX   = 4'd4,
        S_MATMUL_PV        = 4'd6,
        S_FUSED_UPDATE     = 4'd7,
        S_CHECK_INNER      = 4'd8,
        S_NORMALIZE        = 4'd9,
        S_CHECK_OUTER      = 4'd10,
        S_DONE             = 4'd11,
        S_PF_WAIT          = 4'd12    // NEW: wait for shadow prefetch, then swap
    } state_t;

    state_t state;
    logic   array_started;
    logic   load_started;     // one-shot guard for foreground ld_start
    logic   pf_pending;       // a prefetch was launched, not yet consumed by swap
    logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] chunk_cnt;

    assign dbg_state = 4'(state);
    assign start_accepted = (state == S_IDLE) && start;

    logic is_first_kv;
    logic is_last_kv;
    logic [15:0] effective_kv_len;
    logic [15:0] next_tile_col;
    logic        next_tile_valid;      // next tile exists and is not above-diagonal

    localparam int LOG2_T = $clog2(TILE_SIZE);
    logic        tile_resident;
    logic        next_tile_resident;

    assign is_first_kv        = (tile_col == 16'b0);
    assign effective_kv_len   = mode ? kv_len : 16'(SEQ_LEN);
    assign is_last_kv         = ((tile_col + 16'(TILE_SIZE)) >= effective_kv_len);
    assign next_tile_col      = tile_col + 16'(TILE_SIZE);
    // Valid to prefetch: there is a next tile AND it is not above the diagonal.
    assign next_tile_valid    = !is_last_kv && (!causal || next_tile_col <= tile_row);
    // Residency: a tile is consumable once its index < the resident count.
    assign tile_resident      = ((tile_col      >> LOG2_T) < kv_tiles_ready);
    assign next_tile_resident = ((next_tile_col >> LOG2_T) < kv_tiles_ready);
    // The tile currently being prefetched is always the next tile column.
    assign pf_tile_col        = next_tile_col;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            tile_row           <= '0;
            tile_col           <= '0;
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            ld_start           <= 1'b0;
            ld_mode            <= MODE_Q;
            load_started       <= 1'b0;
            pf_start           <= 1'b0;
            pf_pending         <= 1'b0;
            kv_swap_banks      <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            array_started      <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            fused_update_en    <= 1'b0;
            norm_en            <= 1'b0;
            short_cnt_mode     <= 1'b0;
            k_chunk            <= '0;
            pv_done            <= 1'b0;
            chunk_cnt          <= '0;
        end else begin
            // Default: de-assert all pulses every cycle
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            ld_start           <= 1'b0;
            pf_start           <= 1'b0;
            kv_swap_banks      <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            fused_update_en    <= 1'b0;
            norm_en            <= 1'b0;
            pv_done            <= 1'b0;
            // short_cnt_mode, k_chunk, ld_mode, pf_pending are level signals — hold

            case (state)

                S_IDLE: begin
                    short_cnt_mode <= 1'b0;
                    pf_pending     <= 1'b0;    // no prefetch outstanding at start
                    if (start_accepted) begin
                        tile_row      <= '0;
                        tile_col      <= '0;
                        array_started <= 1'b0;
                        load_started  <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        cnt_clr       <= 1'b1;
                        state         <= S_LOAD_Q;
                    end
                end

                // Stripe-load the Q tile (16 bytes/cycle) — foreground load.
                S_LOAD_Q: begin
                    short_cnt_mode <= 1'b0;
                    if (!load_started) begin
                        ld_start     <= 1'b1;
                        ld_mode      <= MODE_Q;
                        load_started <= 1'b1;
                    end
                    if (ld_done) begin
                        load_started <= 1'b0;
                        cnt_clr      <= 1'b1;
                        state        <= S_LOAD_KV;
                    end
                end

                // Stripe-load the KV tile (K and V concurrently) — foreground load.
                // Residency gate: stall until the tile is resident.  Reached only
                // when NO prefetch supplied this tile (pf_pending was 0).
                S_LOAD_KV: begin
                    short_cnt_mode <= 1'b0;
                    if (tile_resident && !load_started) begin
                        ld_start     <= 1'b1;
                        ld_mode      <= MODE_KV;
                        load_started <= 1'b1;
                    end
                    if (ld_done) begin
                        load_started  <= 1'b0;
                        cnt_clr       <= 1'b1;
                        array_started <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        state         <= S_MATMUL_QK;
                    end
                end

                S_MATMUL_QK: begin
                    if (!array_started) begin
                        array_no_clear <= (chunk_cnt != '0);
                        array_start    <= 1'b1;
                        array_started  <= 1'b1;
                    end
                    if (array_done) begin
                        array_started <= 1'b0;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_MATMUL_QK;
                        end else begin
                            chunk_cnt <= '0;
                            k_chunk   <= '0;
                            state     <= S_UPDATE_SOFTMAX;
                        end
                    end
                end

                // Send scores to softmax; on exp_out_valid launch the prefetch of
                // the next non-skipped, resident KV tile into the shadow buffer.
                S_UPDATE_SOFTMAX: begin
                    softmax_tile_valid <= 1'b1;
                    softmax_tile_start <= is_first_kv;
                    softmax_tile_last  <= is_last_kv;
                    if (exp_out_valid) begin
                        softmax_tile_valid <= 1'b0;
                        array_started      <= 1'b0;
                        cnt_clr            <= 1'b1;
                        // Fire prefetch only if the next tile exists (not last, not
                        // above-diagonal) AND is already resident in the scratchpad.
                        if (next_tile_valid && next_tile_resident) begin
                            pf_start   <= 1'b1;
                            pf_pending <= 1'b1;
                        end
                        state              <= S_MATMUL_PV;
                    end
                end

                S_MATMUL_PV: begin
                    if (!array_started) begin
                        array_start   <= 1'b1;
                        array_started <= 1'b1;
                    end
                    if (array_done) begin
                        array_started  <= 1'b0;
                        short_cnt_mode <= 1'b1;
                        cnt_clr        <= 1'b1;
                        state          <= S_FUSED_UPDATE;
                    end
                end

                S_FUSED_UPDATE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    fused_update_en <= 1'b1;
                    if (cnt_done) begin
                        cnt_en          <= 1'b0;
                        fused_update_en <= 1'b0;
                        cnt_clr         <= 1'b1;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_MATMUL_PV;
                        end else begin
                            pv_done        <= 1'b1;
                            short_cnt_mode <= 1'b0;
                            chunk_cnt      <= '0;
                            k_chunk        <= '0;
                            state          <= S_CHECK_INNER;
                        end
                    end
                end

                // Advance to the next inner tile.  Three mutually-exclusive cases:
                //   (a) causal above-diagonal → spin (skip), no prefetch was fired;
                //   (b) prefetch pending      → wait for shadow, then swap;
                //   (c) no prefetch           → foreground reload via S_LOAD_KV.
                // tile_col is advanced exactly ONCE here; S_PF_WAIT never re-advances.
                S_CHECK_INNER: begin
                    if (!is_last_kv) begin
                        tile_col <= next_tile_col;
                        cnt_clr  <= 1'b1;
                        if (causal && next_tile_col > tile_row) begin
                            state <= S_CHECK_INNER;   // skip above-diagonal tile
                        end else if (pf_pending) begin
                            state <= S_PF_WAIT;       // consume the prefetch
                        end else begin
                            state <= S_LOAD_KV;       // no prefetch: normal reload
                        end
                    end else begin
                        cnt_clr <= 1'b1;
                        state   <= S_NORMALIZE;
                    end
                end

                // Wait for the shadow prefetch to complete, then swap shadow→active
                // and skip S_LOAD_KV.  tile_col was already advanced in S_CHECK_INNER;
                // staying here does not change it (correct stall).
                S_PF_WAIT: begin
                    if (pf_rdy) begin
                        kv_swap_banks <= 1'b1;
                        pf_pending    <= 1'b0;
                        array_started <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        state         <= S_MATMUL_QK;
                    end
                    // else: stay in S_PF_WAIT until the shadow buffer is ready
                end

                S_NORMALIZE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    norm_en        <= 1'b1;
                    if (cnt_done) begin
                        cnt_en  <= 1'b0;
                        norm_en <= 1'b0;
                        cnt_clr <= 1'b1;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_NORMALIZE;
                        end else begin
                            short_cnt_mode <= 1'b0;
                            chunk_cnt      <= '0;
                            k_chunk        <= '0;
                            state          <= S_CHECK_OUTER;
                        end
                    end
                end

                S_CHECK_OUTER: begin
                    if (!mode && ((tile_row + 16'(TILE_SIZE)) < 16'(SEQ_LEN))) begin
                        tile_row <= tile_row + 16'(TILE_SIZE);
                        tile_col <= '0;
                        cnt_clr  <= 1'b1;
                        state    <= S_LOAD_Q;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
