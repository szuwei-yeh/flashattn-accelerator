// ============================================================
//  tile_controller_banked.sv — FSM variant for the banked-scratchpad core.
//
//  Identical to tile_controller.sv EXCEPT for the tile-load mechanism:
//    - S_LOAD_Q / S_LOAD_KV no longer drive a byte-serial counter (cnt_en/q_we/
//      kv_we). Instead they pulse a stripe loader (ld_start + ld_mode) and wait
//      for ld_done. The loader fills Q_reg / K_reg+V_reg at 16 bytes/cycle.
//    - The byte-serial KV prefetch double-buffer (kv_prefetch_*, kv_swap_banks)
//      is removed; S_CHECK_INNER always reloads the next KV tile via the loader.
//  Everything from S_MATMUL_QK onward (compute, softmax, output walks) is
//  unchanged, including the kv_tiles_ready residency gate on S_LOAD_KV.
//
//  Base address selection (q_global_offset vs k_global_offset) lives in the
//  core; this FSM only emits ld_start + ld_mode.
// ============================================================
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
module tile_controller_banked #(
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

    input  logic        mode,       // 0 = prefill, 1 = decode
    input  logic [15:0] kv_len,     // runtime KV length (decode mode)

    output logic [15:0] tile_row,
    output logic [15:0] tile_col,
    output logic        cnt_en,
    output logic        cnt_clr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        cnt_done,
    /* verilator lint_on UNUSEDSIGNAL */

    // Stripe-loader handshake (replaces q_we/kv_we byte counting)
    output logic        ld_start,   // 1-cycle pulse to start a tile load
    output logic        ld_mode,    // 0 = Q load, 1 = KV load
    input  logic        ld_done,    // loader finished filling the tile registers

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

    output logic        accum_en,
    output logic        rescale_en,
    output logic        norm_en,
    output logic        short_cnt_mode,
    output logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] k_chunk,
    output logic        pv_done,

    input  logic        causal,

    // DMA streaming interlock (same semantics as tile_controller). Tie to
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
        S_RESCALE_OUTPUT   = 4'd5,
        S_MATMUL_PV        = 4'd6,
        S_ACCUMULATE       = 4'd7,
        S_CHECK_INNER      = 4'd8,
        S_NORMALIZE        = 4'd9,
        S_CHECK_OUTER      = 4'd10,
        S_DONE             = 4'd11
    } state_t;

    state_t state;
    logic   array_started;
    logic   load_started;     // one-shot guard for ld_start
    logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] chunk_cnt;

    assign dbg_state = 4'(state);

    logic is_first_kv;
    logic is_last_kv;
    logic [15:0] effective_kv_len;
    logic [15:0] next_tile_col;

    localparam int LOG2_T = $clog2(TILE_SIZE);
    logic        tile_resident;

    assign is_first_kv      = (tile_col == 16'b0);
    assign effective_kv_len = mode ? kv_len : 16'(SEQ_LEN);
    assign is_last_kv       = ((tile_col + 16'(TILE_SIZE)) >= effective_kv_len);
    assign next_tile_col    = tile_col + 16'(TILE_SIZE);
    assign tile_resident    = ((tile_col >> LOG2_T) < kv_tiles_ready);

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
        end else begin
            // Default: de-assert all pulses every cycle
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            ld_start           <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            accum_en           <= 1'b0;
            rescale_en         <= 1'b0;
            norm_en            <= 1'b0;
            pv_done            <= 1'b0;
            // short_cnt_mode, k_chunk, ld_mode are level signals — hold

            case (state)

                S_IDLE: begin
                    short_cnt_mode <= 1'b0;
                    if (start) begin
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

                // Stripe-load the Q tile (16 bytes/cycle)
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

                // Stripe-load the KV tile (K and V concurrently). Residency gate:
                // stall (do not start the loader) until the tile is resident.
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

                S_UPDATE_SOFTMAX: begin
                    softmax_tile_valid <= 1'b1;
                    softmax_tile_start <= is_first_kv;
                    softmax_tile_last  <= is_last_kv;
                    if (exp_out_valid) begin
                        softmax_tile_valid <= 1'b0;
                        array_started      <= 1'b0;
                        cnt_clr            <= 1'b1;
                        state              <= S_RESCALE_OUTPUT;
                    end
                end

                S_RESCALE_OUTPUT: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    rescale_en     <= 1'b1;
                    if (cnt_done) begin
                        cnt_en     <= 1'b0;
                        rescale_en <= 1'b0;
                        cnt_clr    <= 1'b1;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
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

                S_ACCUMULATE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    accum_en       <= 1'b1;
                    if (cnt_done) begin
                        cnt_en   <= 1'b0;
                        accum_en <= 1'b0;
                        cnt_clr  <= 1'b1;
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

                // No prefetch: always reload the next KV tile via the loader.
                S_CHECK_INNER: begin
                    if (!is_last_kv) begin
                        tile_col <= next_tile_col;
                        cnt_clr  <= 1'b1;
                        if (causal && next_tile_col > tile_row) begin
                            state <= S_CHECK_INNER;   // skip above-diagonal tile
                        end else begin
                            state <= S_LOAD_KV;
                        end
                    end else begin
                        cnt_clr <= 1'b1;
                        state   <= S_NORMALIZE;
                    end
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
