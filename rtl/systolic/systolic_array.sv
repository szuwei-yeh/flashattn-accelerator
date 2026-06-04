`timescale 1ns/1ps

// Internal wires are 2D for readability.
// a_wire[r][c]  = horizontal wire entering PE[r][c] from the left
// b_wire[r][c]  = vertical   wire entering PE[r][c] from above
// acc[r*SIZE+c] = PE[r][c] accumulator output (flat port for C++ TB)

// Ports use packed vectors instead of unpacked arrays for Yosys compatibility.
module systolic_array #(
    parameter int SIZE = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        en,
    input  logic        clear,
    input  logic [SIZE*8-1:0]        a_in,  // packed: SIZE elements × 8b
    input  logic [SIZE*8-1:0]        b_in,  // packed: SIZE elements × 8b
    output logic [SIZE*SIZE*32-1:0]  acc    // packed: SIZE×SIZE elements × 32b
);
    // a_wire[r][0..SIZE]: left-edge (col 0) fed from a_in[r],
    //                     col c+1 driven by PE[r][c].a_out
    logic signed [7:0] a_wire [SIZE-1:0][SIZE:0];
    // b_wire[0..SIZE][c]: top-edge (row 0) fed from b_in[c],
    //                     row r+1 driven by PE[r][c].b_out
    logic signed [7:0] b_wire [SIZE:0][SIZE-1:0];

    genvar r, c;
    generate
        // Unpack packed inputs to left/top edge
        for (r = 0; r < SIZE; r++) begin : ai
            assign a_wire[r][0] = a_in[r*8 +: 8];
        end
        for (c = 0; c < SIZE; c++) begin : bi
            assign b_wire[0][c] = b_in[c*8 +: 8];
        end

        // Instantiate SIZE×SIZE PEs
        for (r = 0; r < SIZE; r++) begin : row
            for (c = 0; c < SIZE; c++) begin : col
                pe u_pe (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .en    (en),
                    .clear (clear),
                    .a_in  (a_wire[r][c]),
                    .b_in  (b_wire[r][c]),
                    .a_out (a_wire[r][c+1]),
                    .b_out (b_wire[r+1][c]),
                    .acc   (acc[(r*SIZE+c)*32 +: 32])
                );
            end
        end
    endgenerate

endmodule
