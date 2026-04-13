// ============================================================
//  pe.sv  —  Single Processing Element
//  INT8 × INT8  →  INT32 accumulation
//
//  Data flow:
//    a_in  comes from the LEFT  (Q row element)
//    b_in  comes from the TOP   (K col element)
//    a_out passes RIGHT         (pipelining Q across columns)
//    b_out passes DOWN          (pipelining K across rows)
//    acc   accumulates Q·K dot-product terms
//
//  Reset:   synchronous active-low  (negedge rst_n)
//  Enable:  en must be high to load / accumulate
//  Clear:   assert clear to zero acc without reset
// ============================================================
`timescale 1ns/1ps

module pe (
    input  logic        clk,
    input  logic        rst_n,   // active-low async reset
    input  logic        en,      // clock enable
    input  logic        clear,   // zero accumulator (sync)

    input  logic signed [7:0]  a_in,   // from left  (Q element)
    input  logic signed [7:0]  b_in,   // from top   (K element)

    output logic signed [7:0]  a_out,  // pass right
    output logic signed [7:0]  b_out,  // pass down
    output logic signed [31:0] acc     // accumulated dot product
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= '0;
            a_out <= '0;
            b_out <= '0;
        end else if (en) begin
            if (clear) begin
                acc   <= '0;
                a_out <= a_in;
                b_out <= b_in;
            end else begin
                // INT8 × INT8  →  sign-extended to INT32 before adding
                acc   <= acc + 32'(signed'(a_in) * signed'(b_in));
                a_out <= a_in;
                b_out <= b_in;
            end
        end
    end

endmodule
