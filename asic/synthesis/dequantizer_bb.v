/// sta-blackbox
// Blackbox stub for dequantizer (Verilog-2005 compatible)
module dequantizer #(
    parameter OUT_WIDTH = 16,
    parameter FRAC_BITS = 8
)(
    input  clk,
    input  rst_n,
    input  valid_in,
    input  signed [31:0] data_in,
    input  signed [15:0] scale_q,
    input  signed [15:0] scale_k,
    output valid_out,
    output signed [OUT_WIDTH-1:0] data_out
);
endmodule
