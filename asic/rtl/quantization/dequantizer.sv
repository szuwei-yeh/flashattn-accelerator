// ============================================================
//  dequantizer.sv — INT32 accumulation → fixed-point rescale
//
//  Rescales the INT32 result from the systolic array back to
//  fixed-point using the two quantisation scales from Q and K:
//
//    data_out = round( (data_in * scale_q * scale_k) >> (2*FRAC_BITS) )
//
//  where scale_q, scale_k are Q8.8 values (Q-format with FRAC_BITS
//  fractional bits), so the two-step right-shift removes both scale
//  factors' fractional parts, leaving a result in OUT_WIDTH bits.
//
//  Latency: 2 clock cycles (mid-product register + output register).
// ============================================================
`timescale 1ns/1ps

// keep_hierarchy: synthesize this module ONCE (not inlined/flattened ×256) so
// ABC optimizes the multiply pipeline per-instance instead of in a 2.5M-cell
// bulk flatten. Pure synthesis directive — simulators ignore it, regression
// is unaffected (functionally identical).
(* keep_hierarchy *)
module dequantizer #(
    parameter OUT_WIDTH = 16,
    parameter FRAC_BITS = 8
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            valid_in,
    input  logic signed [31:0]              data_in,   // INT32 from systolic array
    input  logic signed [15:0]              scale_q,   // Q scale  (Q8.8)
    input  logic signed [15:0]              scale_k,   // K scale  (Q8.8)
    output logic                            valid_out,
    output logic signed [OUT_WIDTH-1:0]     data_out
);

    // ── Two-stage multiply pipeline ────────────────────────────────────────
    // data_in (32b) × scale_q (16b) → 48b
    // × scale_k (16b)               → 64b
    // Round then >> (2*FRAC_BITS)   → 64 - 2*8 = 48b effective
    // Output is Q8.8: shift removes one Q8.8 scale factor.
    // data_in * sq_q88 * sk_q88 >> FRAC_BITS = score_float * 256 (Q8.8)
    localparam int SHIFT = FRAC_BITS;

    logic signed [47:0] mid_q;      // stage 1: data_in * scale_q
    logic signed [15:0] scale_k_q;
    logic               valid_s1;

    logic signed [63:0] wide;       // stage 2: mid_q * scale_k_q
    logic signed [63:0] rounded;
    logic signed [63:0] shifted;

    // Rounding constant: 2^(SHIFT-1) adds 0.5 LSB before truncation
    localparam logic signed [63:0] ROUND_HALF = 64'(1) << (SHIFT - 1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mid_q     <= '0;
            scale_k_q <= 16'sh0100;
            valid_s1  <= 1'b0;
        end else begin
            valid_s1 <= valid_in;
            if (valid_in) begin
                mid_q     <= $signed(data_in) * $signed(scale_q);
                scale_k_q <= scale_k;
            end
        end
    end

    always_comb begin
        wide    = mid_q * $signed(scale_k_q);
        // Round-to-nearest: add 0.5 ULP in the discarded bits
        // (valid for positive products; sign-correct because we truncate)
        rounded = wide + ROUND_HALF;
        shifted = rounded >>> SHIFT;
    end

    // ── Clamp to OUT_WIDTH and register ───────────────────────────────────
    localparam logic signed [63:0] MAX_OUT =  (64'(1) << (OUT_WIDTH - 1)) - 1;
    localparam logic signed [63:0] MIN_OUT = -(64'(1) << (OUT_WIDTH - 1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out  <= '0;
        end else begin
            valid_out <= valid_s1;
            if (valid_s1) begin
                if (shifted > MAX_OUT)
                    data_out <= OUT_WIDTH'(MAX_OUT[OUT_WIDTH-1:0]);
                else if (shifted < MIN_OUT)
                    data_out <= OUT_WIDTH'(MIN_OUT[OUT_WIDTH-1:0]);
                else
                    data_out <= shifted[OUT_WIDTH-1:0];
            end
        end
    end

endmodule
