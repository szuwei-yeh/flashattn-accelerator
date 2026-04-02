`timescale 1ns/1ps

// Icarus-safe: ALL internal arrays are 1D flat.
// a_wire[r * (SIZE+1) + c]  = horizontal wire at row r, col-slot c
// b_wire[r * SIZE + c]      = vertical wire at row-slot r, col c
// acc[r * SIZE + c]         = PE[r][c] accumulator output

module systolic_array #(
    parameter int SIZE = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        en,
    input  logic        clear,
    input  logic signed [7:0] a_in [SIZE-1:0],
    input  logic signed [7:0] b_in [SIZE-1:0],
    output logic signed [31:0] acc [SIZE*SIZE-1:0]
);
    // Horizontal wires: SIZE rows × (SIZE+1) positions
    logic signed [7:0] a_wire [(SIZE)*(SIZE+1)-1:0];
    // Vertical wires: (SIZE+1) rows × SIZE cols
    logic signed [7:0] b_wire [(SIZE+1)*(SIZE)-1:0];

    genvar r, c;
    generate
        // Connect external inputs to left/top edge
        for (r = 0; r < SIZE; r++) begin : ai
            assign a_wire[r*(SIZE+1) + 0] = a_in[r];
        end
        for (c = 0; c < SIZE; c++) begin : bi
            assign b_wire[0*SIZE + c] = b_in[c];
        end

        // Instantiate SIZE×SIZE PEs
        for (r = 0; r < SIZE; r++) begin : row
            for (c = 0; c < SIZE; c++) begin : col
                pe u_pe (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .en    (en),
                    .clear (clear),
                    .a_in  (a_wire[r*(SIZE+1) + c]),
                    .b_in  (b_wire[r*SIZE     + c]),
                    .a_out (a_wire[r*(SIZE+1) + c+1]),
                    .b_out (b_wire[(r+1)*SIZE + c]),
                    .acc   (acc[r*SIZE + c])
                );
            end
        end
    endgenerate

endmodule
