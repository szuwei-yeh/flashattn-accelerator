/// sta-blackbox
// Blackbox stub for output_buffer (Verilog-2005 compatible)
// ADDR_WIDTH = $clog2(4096) = 12
module output_buffer #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 4096,
    parameter ADDR_WIDTH = 12
)(
    input  clk,
    input  rst_n,
    input  accum_en,
    input  [ADDR_WIDTH-1:0] addr,
    input  signed [31:0] data_in,
    input  rescale_en,
    input  [ADDR_WIDTH-1:0] rescale_addr,
    input  [15:0] rescale_q88,
    input  norm_en,
    input  [ADDR_WIDTH-1:0] norm_addr,
    input  [31:0] norm_divisor,
    input  re_ext,
    input  [ADDR_WIDTH-1:0] raddr_ext,
    output signed [31:0] rdata_ext
);
endmodule
