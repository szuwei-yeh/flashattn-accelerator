`timescale 1ns/1ps

// Synthesis-only blackbox for exp LUT ROM macro modeling.
// Replaces rtl/softmax/exp_lut.sv in macro-based DC runs.
module exp_lut (
    input  logic        clk,
    input  logic [7:0]  addr,
    output logic [15:0] exp_val
);
endmodule
