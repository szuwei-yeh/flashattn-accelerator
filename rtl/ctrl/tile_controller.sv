// ============================================================
//  tile_controller.sv — Main FSM for FlashAttention tiling loop
// ============================================================
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
module tile_controller #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 64,
    parameter int SEQ_LEN   = 64
)(
/* verilator lint_on UNUSEDPARAM */
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic        mode,       // 0 = prefill (use SEQ_LEN param), 1 = decode (use kv_len)
    input  logic [15:0] kv_len,     // runtime KV sequence length, used when mode=1

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
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        array_done,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        softmax_tile_start,
    output logic        softmax_tile_valid,
    output logic        softmax_tile_last,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        softmax_out_valid,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        accum_en,
    output logic        norm_en,

    output logic [3:0]  dbg_state
);

    typedef enum logic [3:0] {
        S_IDLE            = 4'd0,
        S_LOAD_Q          = 4'd1,
        S_LOAD_KV         = 4'd2,
        S_MATMUL_QK       = 4'd3,
        S_UPDATE_SOFTMAX  = 4'd4,
        S_MATMUL_PV       = 4'd5,
        S_ACCUMULATE      = 4'd6,
        S_CHECK_INNER     = 4'd7,
        S_NORMALIZE       = 4'd8,
        S_CHECK_OUTER     = 4'd9,
        S_DONE            = 4'd10
    } state_t;

    state_t state;
    logic   array_started;  // prevents re-pulsing array_start every cycle

    assign dbg_state = 4'(state);

    /* verilator lint_off UNUSEDSIGNAL */
    logic is_first_kv;
    /* verilator lint_on UNUSEDSIGNAL */
    logic is_last_kv;
    logic [15:0] effective_kv_len;
    assign is_first_kv      = (tile_col == 16'b0);
    assign effective_kv_len = mode ? kv_len : 16'(SEQ_LEN);
    assign is_last_kv       = ((tile_col + 16'(TILE_SIZE)) >= effective_kv_len);

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
            array_started      <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            accum_en           <= 1'b0;
            norm_en            <= 1'b0;
        end else begin
            // Default: de-assert all pulses every cycle
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            kv_swap_banks      <= 1'b0;
            q_we               <= 1'b0;
            kv_we              <= 1'b0;
            array_start        <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            accum_en           <= 1'b0;
            norm_en            <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (start) begin
                        tile_row      <= '0;
                        tile_col      <= '0;
                        array_started <= 1'b0;
                        cnt_clr       <= 1'b1;
                        kv_swap_banks <= 1'b1;   // bank_sel: 0→1; read bank becomes bank 0 where testbench wrote K/V
                        state         <= S_LOAD_Q;
                    end
                end

                S_LOAD_Q: begin
                    cnt_en <= 1'b1;
                    q_we   <= 1'b1;
                    if (cnt_done) begin
                        cnt_en  <= 1'b0;  // override: prevent cnt_done from firing on 1st cycle of LOAD_KV
                        cnt_clr <= 1'b1;
                        state   <= S_LOAD_KV;
                    end
                end

                S_LOAD_KV: begin
                    cnt_en <= 1'b1;
                    kv_we  <= 1'b1;
                    if (cnt_done) begin
                        cnt_en        <= 1'b0;  // override: stop cnt_en when done
                        cnt_clr       <= 1'b1;
                        array_started <= 1'b0;
                        state         <= S_MATMUL_QK;
                    end
                end

                // Pulse array_start exactly once, then wait for array_done
                S_MATMUL_QK: begin
                    if (!array_started) begin
                        array_start   <= 1'b1;
                        array_started <= 1'b1;
                    end
                    if (array_done) begin
                        array_started <= 1'b0;
                        state         <= S_UPDATE_SOFTMAX;
                    end
                end

                S_UPDATE_SOFTMAX: begin
                    softmax_tile_valid <= 1'b1;
                    softmax_tile_start <= 1'b1;  // always reset: per-tile independent softmax
                    softmax_tile_last  <= 1'b1;  // always finalize: fires out_valid every tile
                    if (softmax_out_valid) begin
                        softmax_tile_valid <= 1'b0;
                        array_started      <= 1'b0;
                        state              <= S_MATMUL_PV;
                    end
                end

                // Pulse array_start exactly once, then wait for array_done
                S_MATMUL_PV: begin
                    if (!array_started) begin
                        array_start   <= 1'b1;
                        array_started <= 1'b1;
                    end
                    if (array_done) begin
                        array_started <= 1'b0;
                        cnt_clr       <= 1'b1;   // reset counter before accumulate walk
                        state         <= S_ACCUMULATE;
                    end
                end

                // Walk sram_addr 0..FLAT-1; write one element per cycle
                S_ACCUMULATE: begin
                    cnt_en   <= 1'b1;
                    accum_en <= 1'b1;
                    if (cnt_done) begin
                        cnt_en  <= 1'b0;  // override: stop cnt_en when done
                        cnt_clr <= 1'b1;
                        state   <= S_CHECK_INNER;
                    end
                end

                S_CHECK_INNER: begin
                    if (!is_last_kv) begin
                        tile_col      <= tile_col + 16'(TILE_SIZE);
                        kv_swap_banks <= 1'b1;
                        cnt_clr       <= 1'b1;
                        state         <= S_LOAD_KV;
                    end else begin
                        state <= S_NORMALIZE;
                    end
                end

                S_NORMALIZE: begin
                    norm_en <= 1'b1;
                    state   <= S_CHECK_OUTER;
                end

                S_CHECK_OUTER: begin
                    // In decode mode (mode=1) there is only one Q tile (row 0),
                    // so always exit after the first Q tile.
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
