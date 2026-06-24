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
    input  logic        a_unsigned, // 1 = treat a_in as UNSIGNED [0,255] (PV phase: a = softmax P)

    input  logic signed [7:0]  a_in,   // from left  (Q element, or P in PV phase)
    input  logic signed [7:0]  b_in,   // from top   (K element)

    output logic signed [7:0]  a_out,  // pass right
    output logic signed [7:0]  b_out,  // pass down
    output logic signed [31:0] acc     // accumulated dot product
);

    // 9-bit signed view of a_in: zero-extended (always positive) when a_unsigned,
    // else sign-extended. Keeps P (softmax weight, [0,255]) positive in PV phase.
    logic signed [8:0] a_ext;
    assign a_ext = a_unsigned ? signed'({1'b0, a_in}) : signed'({a_in[7], a_in});

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
                // a_ext × INT8 → INT32. In PV phase a_ext is the unsigned softmax
                // weight P (MSB not misread as sign); QK phase keeps a sign-extended.
                acc   <= acc + 32'(a_ext * signed'(b_in));
                a_out <= a_in;
                b_out <= b_in;
            end
        end
    end

endmodule
