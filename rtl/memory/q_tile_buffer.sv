// ============================================================
//  q_tile_buffer.sv — Q Tile SRAM Wrapper
//  - Capacity: 64x64 bytes = 4 KB [cite: 358]
// ============================================================
`timescale 1ns/1ps

module q_tile_buffer #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 4096,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,

    input  logic                  we_ext,
    input  logic [ADDR_WIDTH-1:0] waddr_ext,
    input  logic [DATA_WIDTH-1:0] wdata_ext,
    
    input  logic                  re_int,
    input  logic [ADDR_WIDTH-1:0] raddr_int,
    output logic [DATA_WIDTH-1:0] rdata_int
);

    sram_1r1w #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) u_sram_q (
        .clk(clk),
        .we(we_ext),
        .waddr(waddr_ext),
        .wdata(wdata_ext),
        .re(re_int),
        .raddr(raddr_int),
        .rdata(rdata_int)
    );

endmodule
