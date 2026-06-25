/// sta-blackbox
(* blackbox *)
module sram_1r1w #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 256,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,
    input  logic                  we,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [DATA_WIDTH-1:0] wdata,
    input  logic                  re,
    input  logic [ADDR_WIDTH-1:0] raddr,
    output logic [DATA_WIDTH-1:0] rdata
);
endmodule
