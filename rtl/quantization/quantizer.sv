// ============================================================
//  quantizer.sv — Fixed-point → INT8 quantization
//
//  data_out = clamp( round(data_in / scale), -128, 127 )
//
//  Both data_in and scale are signed Q(IN_WIDTH-FRAC_BITS).FRAC_BITS.
//  Division is implemented as an integer divide of the raw Q8.8 integers
//  (real_data / real_scale = data_int / scale_int), which is exact for
//  power-of-two scales and correct in general via hardware integer divide.
//
//  Latency: 1 clock cycle (registered output).
// ============================================================
`timescale 1ns/1ps

module quantizer #(
    parameter int IN_WIDTH  = 16,   // input  bit width  (signed)
    /* verilator lint_off UNUSEDPARAM */
    parameter int FRAC_BITS = 8     // documents Q-format; not needed in computation
    /* verilator lint_on UNUSEDPARAM */
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic signed [IN_WIDTH-1:0]    data_in,
    input  logic signed [IN_WIDTH-1:0]    scale,    // Q(IN_WIDTH-FRAC_BITS).FRAC_BITS
    output logic                          valid_out,
    output logic signed [7:0]             data_out  // INT8 output
);

    // ── Combinational quotient (truncate toward zero) ──────────────────────
    // real(data_in) / real(scale)
    //   = (data_in / 2^FRAC_BITS) / (scale / 2^FRAC_BITS)
    //   = data_in / scale          (integer divide of raw fixed-point codes)
    // Round to nearest: add scale/2 with sign-matching before dividing.
    localparam int WIDE = IN_WIDTH + 1;

    logic signed [WIDE-1:0] data_ext;
    logic signed [WIDE-1:0] scale_ext;
    logic signed [WIDE-1:0] half_scale;    // |scale| / 2, with sign of data_in
    logic signed [WIDE-1:0] rounded;
    logic signed [WIDE-1:0] quotient;

    always_comb begin
        data_ext   = signed'({{1{data_in[IN_WIDTH-1]}},  data_in});
        scale_ext  = signed'({{1{scale[IN_WIDTH-1]}},    scale});

        // half_scale = floor(|scale| / 2), same sign as data_in
        // Adds rounding bias so that integer divide truncates to nearest.
        half_scale = (data_ext[WIDE-1] == scale_ext[WIDE-1])
                     ? (scale_ext >>> 1)          // same sign → positive quotient
                     : -(scale_ext >>> 1);        // different sign → negative quotient

        rounded  = data_ext + half_scale;
        quotient = rounded / scale_ext;
    end

    // ── Clamp and register ─────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out  <= 8'sd0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                if (quotient > WIDE'(signed'(8'sd127)))
                    data_out <= 8'sd127;
                else if (quotient < WIDE'(signed'(-8'sd128)))
                    data_out <= -8'sd128;
                else
                    data_out <= quotient[7:0];
            end
        end
    end

endmodule
