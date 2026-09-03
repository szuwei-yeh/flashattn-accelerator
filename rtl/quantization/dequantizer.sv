// ============================================================
//  dequantizer.sv — INT32 accumulation → fixed-point rescale
//
//  Rescales the INT32 result from the systolic array back to
//  fixed-point using the precombined Q/K quantisation scale:
//
//    data_out = round( (data_in * combined_scale) >> FRAC_BITS )
//
//  where combined_scale is the exact signed 32-bit Q16.16 product of
//  the shared Q8.8 scale_q and scale_k values.  Shifting by FRAC_BITS
//  converts the full-precision product to a Q8.8 output.
//
//  Latency: 1 clock cycle (registered output).
// ============================================================
`timescale 1ns/1ps

module dequantizer #(
    parameter int OUT_WIDTH = 16,
    parameter int FRAC_BITS = 8
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            valid_in,
    input  logic signed [31:0]              data_in,   // INT32 from systolic array
    input  logic signed [31:0]              combined_scale, // scale_q * scale_k (Q16.16)
    output logic                            valid_out,
    output logic signed [OUT_WIDTH-1:0]     data_out
);

    // ── Combinational multiply-shift ───────────────────────────────────────
    // data_in (32b) × combined_scale (32b) → full-precision 64b.
    // Output is Q8.8: combined_scale is Q16.16, so shifting by
    // FRAC_BITS retains eight fractional bits in the output.
    localparam int SHIFT = FRAC_BITS;

    logic signed [63:0] product;
    logic signed [63:0] rounded;
    logic signed [63:0] shifted;

    // Rounding constant: 2^(SHIFT-1) adds 0.5 LSB before truncation
    localparam logic signed [63:0] ROUND_HALF = 64'(1) << (SHIFT - 1);

    always_comb begin
        product = signed'(data_in) * signed'(combined_scale);
        // Round-to-nearest: add 0.5 ULP in the discarded bits
        // (valid for positive products; sign-correct because we truncate)
        rounded = product + ROUND_HALF;
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
            valid_out <= valid_in;
            if (valid_in) begin
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
