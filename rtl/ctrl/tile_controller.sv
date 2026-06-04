// ============================================================
//  tile_controller.sv — Main FSM for FlashAttention tiling loop
//
//  v2 (true cross-tile online softmax):
//  - softmax_tile_start: asserted only on the FIRST KV tile of each Q-tile.
//  - softmax_tile_last : asserted only on the LAST  KV tile of each Q-tile.
//  - Waits for exp_out_valid (not softmax_out_valid) to advance; this signal
//    fires every tile after unnormalised exp values are ready.
//  - New state S_RESCALE_OUTPUT: walks output-buffer elements for the current
//    Q-tile and applies the per-row rescale factor exp(m_old − m_new).
//  - New output rescale_en: asserted during S_RESCALE_OUTPUT.
//  - S_NORMALIZE is now a multi-cycle state (uses cnt_en/cnt_done) that
//    divides each accumulated element by the global running sum l_global.
// ============================================================
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
module tile_controller #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 64,
    parameter int SEQ_LEN   = 64,
    localparam int NUM_CHUNKS = HEAD_DIM / TILE_SIZE   // inner-dim chunks (1 for d=16, 4 for d=64)
)(
/* verilator lint_on UNUSEDPARAM */
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic        mode,       // 0 = prefill, 1 = decode
    input  logic [15:0] kv_len,     // runtime KV length (decode mode)

    output logic [15:0] tile_row,
    output logic [15:0] tile_col,
    output logic        cnt_en,
    output logic        cnt_clr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        cnt_done,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        kv_swap_banks,
    output logic        q_we,
    output logic        kv_we,

    output logic        array_start,
    output logic        array_no_clear,  // skip CLEAR for chunks 1..NUM_CHUNKS-1
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        array_done,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        softmax_tile_start,
    output logic        softmax_tile_valid,
    output logic        softmax_tile_last,
    // exp_out_valid: fires every tile when unnormalised exp values are ready
    input  logic        exp_out_valid,
    /* verilator lint_off UNUSEDSIGNAL */
    // softmax_out_valid: fires only on last tile (normalised); unused in FSM now
    input  logic        softmax_out_valid,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        accum_en,
    output logic        rescale_en,   // NEW: walk output-buffer with rescale
    output logic        norm_en,      // NEW: walk output-buffer with normalise
    output logic        short_cnt_mode, // count TILE_SIZE² not TILE_SIZE*HEAD_DIM
    output logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] k_chunk, // inner-dim chunk index
    output logic        pv_done,        // all PV chunks complete (clear is_pv_phase)

    input  logic        causal,       // 1 = causal (decoder) masking enabled

    // KV prefetch interface
    output logic        kv_prefetch_en,     // 1-cycle pulse: start prefetch of next KV tile
    output logic [15:0] kv_prefetch_col,    // tile_col of the tile to prefetch
    input  logic        kv_prefetch_rdy,    // high when prefetch data is ready in _nxt regs

    output logic [3:0]  dbg_state
);

    typedef enum logic [3:0] {
        S_IDLE             = 4'd0,
        S_LOAD_Q           = 4'd1,
        S_LOAD_KV          = 4'd2,
        S_MATMUL_QK        = 4'd3,
        S_UPDATE_SOFTMAX   = 4'd4,
        S_RESCALE_OUTPUT   = 4'd5,   // NEW: rescale accumulated O by exp(m_old-m_new)
        S_MATMUL_PV        = 4'd6,
        S_ACCUMULATE       = 4'd7,
        S_CHECK_INNER      = 4'd8,
        S_NORMALIZE        = 4'd9,   // now multi-cycle: divide O by l_global
        S_CHECK_OUTER      = 4'd10,
        S_DONE             = 4'd11
    } state_t;

    state_t state;
    logic   array_started;
    // inner-dim chunk counter (used for both QK and PV multi-pass)
    logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] chunk_cnt;

    assign dbg_state = 4'(state);

    logic is_first_kv;
    logic is_last_kv;
    logic [15:0] effective_kv_len;
    logic [15:0] next_tile_col;        // combinatorial: tile_col + TILE_SIZE
    logic        next_tile_valid;      // next tile exists and is not above-diagonal

    assign is_first_kv      = (tile_col == 16'b0);
    assign effective_kv_len = mode ? kv_len : 16'(SEQ_LEN);
    assign is_last_kv       = ((tile_col + 16'(TILE_SIZE)) >= effective_kv_len);
    assign next_tile_col    = tile_col + 16'(TILE_SIZE);
    // Valid to prefetch: there is a next tile AND it is not above the diagonal
    assign next_tile_valid  = !is_last_kv && (!causal || next_tile_col <= tile_row);
    // Prefetch col is always the next tile column (consumed only when en fires)
    assign kv_prefetch_col  = next_tile_col;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            tile_row           <= '0;
            tile_col           <= '0;
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            kv_swap_banks      <= 1'b0;
            q_we               <= 1'b0;
            kv_we              <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            array_started      <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            accum_en           <= 1'b0;
            rescale_en         <= 1'b0;
            norm_en            <= 1'b0;
            short_cnt_mode     <= 1'b0;
            k_chunk            <= '0;
            pv_done            <= 1'b0;
            chunk_cnt          <= '0;
            kv_prefetch_en     <= 1'b0;
        end else begin
            // Default: de-assert all pulses every cycle
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            kv_swap_banks      <= 1'b0;
            q_we               <= 1'b0;
            kv_we              <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            accum_en           <= 1'b0;
            rescale_en         <= 1'b0;
            norm_en            <= 1'b0;
            pv_done            <= 1'b0;
            kv_prefetch_en     <= 1'b0;
            // short_cnt_mode and k_chunk are level signals — hold their values

            case (state)

                S_IDLE: begin
                    short_cnt_mode <= 1'b0;
                    if (start) begin
                        tile_row      <= '0;
                        tile_col      <= '0;
                        array_started <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        cnt_clr       <= 1'b1;
                        kv_swap_banks <= 1'b1;
                        state         <= S_LOAD_Q;
                    end
                end

                S_LOAD_Q: begin
                    short_cnt_mode <= 1'b0;   // full HEAD_DIM count
                    cnt_en <= 1'b1;
                    q_we   <= 1'b1;
                    if (cnt_done) begin
                        cnt_en  <= 1'b0;
                        cnt_clr <= 1'b1;
                        state   <= S_LOAD_KV;
                    end
                end

                S_LOAD_KV: begin
                    short_cnt_mode <= 1'b0;   // full HEAD_DIM count
                    cnt_en <= 1'b1;
                    kv_we  <= 1'b1;
                    if (cnt_done) begin
                        cnt_en        <= 1'b0;
                        cnt_clr       <= 1'b1;
                        array_started <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        state         <= S_MATMUL_QK;
                    end
                end

                S_MATMUL_QK: begin
                    if (!array_started) begin
                        // chunk 0: normal CLEAR; chunks 1+: no_clear to accumulate
                        array_no_clear <= (chunk_cnt != '0);
                        array_start    <= 1'b1;
                        array_started  <= 1'b1;
                    end
                    if (array_done) begin
                        array_started <= 1'b0;
                        if (chunk_cnt < NUM_CHUNKS - 1) begin
                            // More inner chunks — advance chunk_cnt and restart array
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_MATMUL_QK;
                        end else begin
                            // All chunks done — move to softmax
                            chunk_cnt <= '0;
                            k_chunk   <= '0;
                            state     <= S_UPDATE_SOFTMAX;
                        end
                    end
                end

                // Send scores to online_softmax.
                // tile_start = 1 only for first KV tile → resets running_max/sum.
                // tile_last  = 1 only for last  KV tile → fires normalised output.
                // Wait for exp_out_valid (unnormalised exp values ready) before
                // proceeding to S_RESCALE_OUTPUT / S_MATMUL_PV.
                S_UPDATE_SOFTMAX: begin
                    softmax_tile_valid <= 1'b1;
                    softmax_tile_start <= is_first_kv;   // ← fixed (was always 1)
                    softmax_tile_last  <= is_last_kv;    // ← fixed (was always 1)
                    if (exp_out_valid) begin
                        softmax_tile_valid <= 1'b0;
                        array_started      <= 1'b0;
                        cnt_clr            <= 1'b1;
                        // Fire prefetch for the next non-skipped KV tile (if any)
                        kv_prefetch_en     <= next_tile_valid;
                        state              <= S_RESCALE_OUTPUT;
                    end
                end

                // Rescale every accumulated output element by exp(m_old − m_new).
                // For HEAD_DIM=64 (NUM_CHUNKS=4): 4 × 256 passes; for d=16: 1 pass.
                S_RESCALE_OUTPUT: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    rescale_en     <= 1'b1;
                    if (cnt_done) begin
                        cnt_en     <= 1'b0;
                        rescale_en <= 1'b0;
                        cnt_clr    <= 1'b1;
                        if (chunk_cnt < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_RESCALE_OUTPUT;
                        end else begin
                            short_cnt_mode <= 1'b0;
                            array_started  <= 1'b0;
                            chunk_cnt      <= '0;
                            k_chunk        <= '0;
                            state          <= S_MATMUL_PV;
                        end
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
                        state          <= S_ACCUMULATE;
                    end
                end

                // S_ACCUMULATE: for each PV chunk, walk 256 elements.
                // After last chunk, advance to S_CHECK_INNER.
                S_ACCUMULATE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    accum_en       <= 1'b1;
                    if (cnt_done) begin
                        cnt_en   <= 1'b0;
                        accum_en <= 1'b0;
                        cnt_clr  <= 1'b1;
                        if (chunk_cnt < NUM_CHUNKS - 1) begin
                            // More PV chunks to process
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_MATMUL_PV;
                        end else begin
                            // All PV chunks done
                            pv_done        <= 1'b1;
                            short_cnt_mode <= 1'b0;
                            chunk_cnt      <= '0;
                            k_chunk        <= '0;
                            state          <= S_CHECK_INNER;
                        end
                    end
                end

                S_CHECK_INNER: begin
                    if (!is_last_kv) begin
                        tile_col <= next_tile_col;
                        cnt_clr  <= 1'b1;
                        // Causal: if the next tile is strictly above the
                        // diagonal (tile_col_new > tile_row) skip it without
                        // loading or computing — loop back to S_CHECK_INNER.
                        if (causal && next_tile_col > tile_row) begin
                            state <= S_CHECK_INNER;
                        end else if (kv_prefetch_rdy) begin
                            // Prefetch complete: swap banks and skip S_LOAD_KV
                            kv_swap_banks <= 1'b1;
                            state         <= S_MATMUL_QK;
                        end else begin
                            // No prefetch available: go through LOAD_KV normally
                            kv_swap_banks <= 1'b1;
                            state         <= S_LOAD_KV;
                        end
                    end else begin
                        cnt_clr <= 1'b1;
                        state   <= S_NORMALIZE;
                    end
                end

                // Normalise: divide each output element by l_global.
                // For HEAD_DIM=64: 4 × 256 passes; for d=16: 1 pass.
                S_NORMALIZE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    norm_en        <= 1'b1;
                    if (cnt_done) begin
                        cnt_en  <= 1'b0;
                        norm_en <= 1'b0;
                        cnt_clr <= 1'b1;
                        if (chunk_cnt < NUM_CHUNKS - 1) begin
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
