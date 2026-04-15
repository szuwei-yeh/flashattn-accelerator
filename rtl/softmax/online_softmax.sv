// ============================================================
//  online_softmax.sv — Flash-Attention online softmax
//
//  v2 (true cross-tile online softmax):
//  - running_max / running_sum carry across KV tiles.
//  - tile_start=1 resets state for a new Q-tile.
//  - rescale_q88 / rescale_valid: exp(m_old − m_new) in Q8.8,
//    valid one cycle after S_RESCALE (for output-buffer rescaling).
//  - exp_flat / exp_out_valid: unnormalized exp values (Q8.8),
//    valid every tile (used for P×V on all KV tiles).
//  - softmax_flat / out_valid: normalised output, fires only when
//    tile_last=1 (kept for compatibility, currently unused in top).
//  - running_sum_out: final l_global per Q-row (for normalisation).
// ============================================================
`timescale 1ns/1ps

module online_softmax #(
    parameter int DIM = 16
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               tile_start,   // 1 = first KV tile of this Q-tile → reset state
    input  logic               tile_valid,   // scores are valid on this cycle
    input  logic               tile_last,    // 1 = last KV tile of this Q-tile

    input  logic [DIM*16-1:0]  scores_flat,

    // ── Unnormalised exp output (every tile) ──────────────────
    output logic               exp_out_valid,           // pulses when exp_flat is valid
    output logic [DIM*16-1:0]  exp_flat,                // exp(score − running_max), Q8.8

    // ── Rescale factor for output-buffer (every tile) ─────────
    output logic               rescale_valid,           // pulses when rescale_q88 is valid
    output logic [15:0]        rescale_q88,             // exp(m_old − m_new), Q8.8

    // ── Running sum (l_global) for final normalisation ────────
    output logic [31:0]        running_sum_out,         // valid after last tile

    // ── Normalised output (last tile only) ───────────────────
    output logic               out_valid,
    output logic [DIM*16-1:0]  softmax_flat,

    output logic [2:0]         dbg_state
);

    localparam int IDX_W  = $clog2(DIM + 1);
    localparam int AIDX_W = $clog2(DIM);

    // ── Unpack input scores ───────────────────────────────────
    logic signed [15:0] scores_in [DIM];
    for (genvar g = 0; g < DIM; g++) begin : gen_unpack
        assign scores_in[g] = signed'(scores_flat[g*16 +: 16]);
    end

    // ── exp LUT ───────────────────────────────────────────────
    logic [7:0]  lut_addr;
    logic [15:0] lut_out;

    exp_lut u_lut (
        .clk    (clk),
        .addr   (lut_addr),
        .exp_val(lut_out)
    );

    // ── Normalisation (last tile only) ────────────────────────
    logic [31:0] norm_numer [DIM];
    logic [31:0] norm_denom;
    logic [31:0] norm_result [DIM];

    assign norm_denom = (running_sum != 0) ? running_sum : 32'd1;
    for (genvar gn = 0; gn < DIM; gn++) begin : gen_norm
        assign norm_numer[gn]  = {16'b0, exp_vals[gn]} << 8;
        assign norm_result[gn] = (norm_numer[gn] + (norm_denom >> 1)) / norm_denom;
    end

    // ── State machine ─────────────────────────────────────────
    typedef enum logic [2:0] {
        S_IDLE, S_FIND_MAX, S_RESCALE, S_RESCALE_WAIT, S_ACCUM, S_NORM
    } state_t;
    state_t              state;

    assign dbg_state = 3'(state);
    logic [IDX_W-1:0]    idx;

    logic signed [15:0]  sc [DIM];
    logic                tile_last_r;
    logic                is_first_r;     // 1 = first KV tile of Q-tile (reset running state)

    logic signed [15:0]  running_max;
    logic        [31:0]  running_sum;

    logic signed [15:0]  tile_max;
    logic        [15:0]  exp_vals [DIM];
    logic        [31:0]  accum_sum;

    logic signed [15:0]  next_max;
    assign next_max = (signed'(tile_max) > signed'(running_max)) ? tile_max : running_max;

    // rescale_prod = running_sum × exp(m_old − m_new)
    logic [63:0] rescale_prod;
    assign rescale_prod = {32'h0, running_sum} * {48'h0, lut_out};

    // Continuous outputs
    assign running_sum_out = running_sum;
    // exp_vals is Q8.8: max = 0x0100 (exp=1.0). Cap upper bound to 0x00FF so
    // the lower byte (used as P_int8) is in [0,255] without sign-wrapping at 256.
    for (genvar ge = 0; ge < DIM; ge++) begin : gen_exp_flat
        assign exp_flat[ge*16 +: 16] = (exp_vals[ge] > 16'h00FF) ? 16'h00FF
                                                                   : exp_vals[ge];
    end

    // ── LUT address select ────────────────────────────────────
    function automatic [7:0] to_addr;
        input logic signed [15:0] score;
        input logic signed [15:0] maxv;
        logic signed [16:0]  diff;
        logic signed [31:0]  numer;
        logic signed [31:0]  result32;
        begin
            diff     = {score[15], score} - {maxv[15], maxv};
            numer    = {{15{diff[16]}}, diff} * 32'sd255 + 32'sd523264;
            result32 = numer >>> 11;
            if      (result32 > 32'sd255) to_addr = 8'd255;
            else if (result32 < 32'sd0)   to_addr = 8'd0;
            else                          to_addr = result32[7:0];
        end
    endfunction

    always_comb begin
        lut_addr = 8'd255;
        case (state)
            S_RESCALE:      lut_addr = to_addr(running_max, next_max);
            S_RESCALE_WAIT: lut_addr = to_addr(sc[0], running_max);
            S_ACCUM: begin
                if (idx < IDX_W'(DIM))
                    lut_addr = to_addr(sc[idx[AIDX_W-1:0]], running_max);
            end
            default: lut_addr = 8'd255;
        endcase
    end

    // ── FSM sequential block ──────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            idx            <= '0;
            running_max    <= 16'sh8000;
            running_sum    <= '0;
            is_first_r     <= 1'b1;
            tile_last_r    <= 1'b0;
            tile_max       <= '0;
            accum_sum      <= '0;
            rescale_q88    <= 16'h0100;  // 1.0 in Q8.8
            rescale_valid  <= 1'b0;
            exp_out_valid  <= 1'b0;
            out_valid      <= 1'b0;
            softmax_flat   <= '0;
            for (int i = 0; i < DIM; i++) begin
                sc[i]       <= '0;
                exp_vals[i] <= '0;
            end
        end else begin
            // Default: de-assert all pulses
            rescale_valid <= 1'b0;
            exp_out_valid <= 1'b0;
            out_valid     <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (tile_valid) begin
                        for (int i = 0; i < DIM; i++) sc[i] <= scores_in[i];
                        tile_last_r <= tile_last;
                        is_first_r  <= tile_start;
                        tile_max    <= scores_in[0];
                        accum_sum   <= '0;
                        idx         <= IDX_W'(1);
                        state       <= S_FIND_MAX;
                    end
                end

                S_FIND_MAX: begin
                    if (signed'(sc[idx[AIDX_W-1:0]]) > signed'(tile_max))
                        tile_max <= sc[idx[AIDX_W-1:0]];

                    if (idx == IDX_W'(DIM - 1)) begin
                        idx   <= '0;
                        state <= S_RESCALE;
                    end else begin
                        idx <= idx + 1'b1;
                    end
                end

                S_RESCALE: begin
                    // Update running_max:
                    //   first tile → tile_max (fresh start for this Q-tile)
                    //   later tiles → max(running_max, tile_max)
                    running_max <= is_first_r ? tile_max : next_max;

                    // First tile: reset running_sum (fresh start for this Q-tile)
                    // Later tiles: running_sum will be rescaled in S_RESCALE_WAIT
                    if (is_first_r)
                        running_sum <= '0;

                    // lut_addr = to_addr(running_max, next_max) [combinatorial]
                    // → lut_out = exp(m_old − m_new) will be ready in S_RESCALE_WAIT
                    state <= S_RESCALE_WAIT;
                end

                S_RESCALE_WAIT: begin
                    // lut_out = exp(m_old − m_new) now valid
                    // Capture rescale factor for output-buffer use
                    rescale_q88 <= lut_out;

                    // Rescale running_sum: l_new = exp(m_old − m_new) × l_old
                    // (The new tile's exp values are added in S_ACCUM)
                    // For is_first_r=1, running_sum was reset to 0, so rescale_prod=0 → OK
                    running_sum <= 32'((rescale_prod + 64'd128) >> 8);

                    accum_sum   <= '0;
                    idx         <= IDX_W'(1);
                    state       <= S_ACCUM;
                end

                S_ACCUM: begin
                    // First cycle of ACCUM: rescale_q88 register is now valid
                    if (idx == IDX_W'(1))
                        rescale_valid <= 1'b1;

                    exp_vals[idx - 1] <= lut_out;
                    accum_sum         <= accum_sum + {16'b0, lut_out};

                    if (idx == IDX_W'(DIM)) begin
                        running_sum <= running_sum + accum_sum + {16'b0, lut_out};
                        state       <= S_NORM;
                    end else begin
                        idx <= idx + 1'b1;
                    end
                end

                S_NORM: begin
                    // exp_flat is combinatorially driven from exp_vals — pulse valid now
                    exp_out_valid <= 1'b1;

                    // Normalised output: only on last tile
                    if (tile_last_r) begin
                        for (int i = 0; i < DIM; i++) begin
                            softmax_flat[i*16 +: 16] <=
                                (norm_result[i] > 32'h0000_FFFF) ? 16'hFFFF
                                                                  : norm_result[i][15:0];
                        end
                        out_valid <= 1'b1;
                    end
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
