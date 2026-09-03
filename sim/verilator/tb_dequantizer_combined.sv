`timescale 1ns/1ps

module tb_dequantizer_combined (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               valid_in,
    input  logic signed [31:0] data_in,
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    output logic               valid_out,
    output logic signed [15:0] data_out,
    output logic signed [31:0] dbg_combined_scale
);

    // Match the core structure: one exact shared scale product feeds the lane.
    logic signed [31:0] combined_scale;
    assign combined_scale     = signed'(scale_q) * signed'(scale_k);
    assign dbg_combined_scale = combined_scale;

    dequantizer #(
        .OUT_WIDTH(16),
        .FRAC_BITS(8)
    ) u_dequantizer (
        .clk            (clk),
        .rst_n          (rst_n),
        .valid_in       (valid_in),
        .data_in        (data_in),
        .combined_scale (combined_scale),
        .valid_out      (valid_out),
        .data_out       (data_out)
    );

endmodule
