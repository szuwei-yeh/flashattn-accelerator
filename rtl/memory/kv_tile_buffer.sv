// ============================================================
//  kv_tile_buffer.sv — K and V Tile SRAM Wrapper (Double-Buffered)
//
//  - Uses two instances of sram_1r1w to implement ping-pong buffering.
//  - When bank_sel==0: Write to Bank 0, Read from Bank 1.
//  - When bank_sel==1: Write to Bank 1, Read from Bank 0.
//  - swap_banks pulse flips the active banks.
// ============================================================
`timescale 1ns/1ps

module kv_tile_buffer #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 4096,  
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  swap_banks, 

    input  logic                  we_ext,
    input  logic [ADDR_WIDTH-1:0] waddr_ext,
    input  logic [DATA_WIDTH-1:0] wdata_ext,

    input  logic                  re_int,
    input  logic [ADDR_WIDTH-1:0] raddr_int,
    output logic [DATA_WIDTH-1:0] rdata_int
);

    logic bank_sel;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_sel <= 1'b0;
        end else if (swap_banks) begin
            bank_sel <= ~bank_sel; 
        end
    end

    logic we_bank0, we_bank1;
    logic re_bank0, re_bank1;
    logic [DATA_WIDTH-1:0] rdata_bank0, rdata_bank1;

    assign we_bank0 = (bank_sel == 1'b0) ? we_ext : 1'b0;
    assign we_bank1 = (bank_sel == 1'b1) ? we_ext : 1'b0;

    assign re_bank0 = (bank_sel == 1'b1) ? re_int : 1'b0;
    assign re_bank1 = (bank_sel == 1'b0) ? re_int : 1'b0;

    assign rdata_int = (bank_sel == 1'b1) ? rdata_bank0 : rdata_bank1;

    sram_1r1w #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) bank0 (
        .clk(clk),
        .we(we_bank0),
        .waddr(waddr_ext),
        .wdata(wdata_ext),
        .re(re_bank0),
        .raddr(raddr_int),
        .rdata(rdata_bank0)
    );

    sram_1r1w #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) bank1 (
        .clk(clk),
        .we(we_bank1),
        .waddr(waddr_ext),
        .wdata(wdata_ext),
        .re(re_bank1),
        .raddr(raddr_int),
        .rdata(rdata_bank1)
    );

endmodule
