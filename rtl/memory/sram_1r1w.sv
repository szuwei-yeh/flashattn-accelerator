// ============================================================
//  sram_1r1w.sv — 1-Read 1-Write SRAM Behavioral Model
//
//  - Synchronous read and write
//  - Fully parameterized for width and depth
//  - Infers BRAM (Block RAM) in most synthesis tools
// ============================================================
`timescale 1ns/1ps

module sram_1r1w #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 256,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,
    
    // Write Port
    input  logic                  we,     // Write Enable
    input  logic [ADDR_WIDTH-1:0] waddr,  // Write Address
    input  logic [DATA_WIDTH-1:0] wdata,  // Write Data
    
    // Read Port
    input  logic                  re,     // Read Enable
    input  logic [ADDR_WIDTH-1:0] raddr,  // Read Address
    output logic [DATA_WIDTH-1:0] rdata   // Read Data (1 cycle latency)
);

    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) begin
            mem[waddr] <= wdata;
        end
        
        if (re) begin
            rdata <= mem[raddr];
        end
    end

endmodule
