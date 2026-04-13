// ============================================================
//  exp_lut.sv — 256-entry exp look-up table (ROM)
//
//  Loaded from  ../../data/exp_lut.hex  (relative to simulation
//  working directory, typically  sim/verilator/).
//
//  Mapping:
//    addr = 0   → x = -8.0    exp_val ≈ 0x0000
//    addr = 255 → x =  0.0    exp_val = 0x0100  (1.0 in Q8.8)
//
//  Output: Q8.8 unsigned — round(exp(x) * 256)
//  Latency: 1 clock cycle (registered output)
// ============================================================
`timescale 1ns/1ps

module exp_lut (
    input  logic        clk,
    input  logic [7:0]  addr,      // 0..255 → maps [-8, 0]
    output logic [15:0] exp_val    // Q8.8 unsigned
);

    logic [15:0] lut_mem [0:255];

    initial begin
        $readmemh("../../data/exp_lut.hex", lut_mem);
    end

    always_ff @(posedge clk) begin
        exp_val <= lut_mem[addr];
    end

endmodule
