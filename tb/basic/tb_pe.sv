// ============================================================
//  tb_pe.sv  —  Unit test for single Processing Element
//
//  Run:  iverilog -g2012 -o tb_pe.out tb/basic/tb_pe.sv rtl/systolic/pe.sv
//        ./tb_pe.out
// ============================================================
`timescale 1ns/1ps

module tb_pe;

    logic        clk   = 0;
    logic        rst_n = 0;
    logic        en    = 0;
    logic        clear = 0;
    logic signed [7:0]  a_in  = 0;
    logic signed [7:0]  b_in  = 0;
    logic signed [7:0]  a_out;
    logic signed [7:0]  b_out;
    logic signed [31:0] acc;

    pe dut (.*);

    always #5 clk = ~clk;

    int pass_cnt = 0;
    int fail_cnt = 0;

    task automatic check(input string name,
                         input logic signed [31:0] got, expected);
        if (got === expected) begin
            $display("  PASS  %s  got=%0d", name, got);
            pass_cnt++;
        end else begin
            $display("  FAIL  %s  got=%0d  expected=%0d", name, got, expected);
            fail_cnt++;
        end
    endtask

    task tick(input int n = 1);
        repeat (n) @(posedge clk);
        #1;
    endtask

    // KEY FIX: always pull en=0 before reset so no stale data
    // accumulates on the first cycle coming out of reset.
    task reset_dut;
        en = 0; a_in = 0; b_in = 0; clear = 0;
        rst_n = 0; tick(2);
        rst_n = 1; tick(1);
    endtask

    initial begin
        $dumpfile("dump_pe.vcd");
        $dumpvars(0, tb_pe);
    end

    initial begin
        $display("========================================");
        $display("  PE Unit Test");
        $display("========================================");

        // [0] Reset
        reset_dut();
        check("reset: acc=0", acc, 0);

        // [1] Single MAC: 3x5=15
        en = 1; a_in = 8'sd3; b_in = 8'sd5;
        tick(1);
        check("single MAC 3x5=15", acc, 15);
        check("a_out passthrough=3", a_out, 8'sd3);
        check("b_out passthrough=5", b_out, 8'sd5);

        // [2] Dot product [1,2,3,4].[5,6,7,8] = 5+12+21+32 = 70
        reset_dut();
        check("reset before dot-product: acc=0", acc, 0);
        en = 1;
        a_in = 8'sd1; b_in = 8'sd5; tick(1);
        a_in = 8'sd2; b_in = 8'sd6; tick(1);
        a_in = 8'sd3; b_in = 8'sd7; tick(1);
        a_in = 8'sd4; b_in = 8'sd8; tick(1);
        check("dot product [1..4].[5..8]=70", acc, 70);

        // [3] Negative: -10 x 3 = -30
        reset_dut();
        en = 1;
        a_in = -8'sd10; b_in = 8'sd3;
        tick(1);
        check("neg x pos = -30", acc, -30);

        // [4] Continue: -30 + (-5 x -4=20) = -10
        a_in = -8'sd5; b_in = -8'sd4;
        tick(1);
        check("accumulated neg x neg = -10", acc, -10);

        // [5] Clear zeroes accumulator
        clear = 1; a_in = 8'sd7; b_in = 8'sd7;
        tick(1);
        clear = 0;
        check("after clear: acc=0", acc, 0);

        // [6] First MAC after clear
        a_in = 8'sd7; b_in = 8'sd7;
        tick(1);
        check("after clear + one MAC 7x7=49", acc, 49);

        // [7] en=0 holds state
        en = 0; a_in = 8'sd100; b_in = 8'sd100;
        tick(3);
        check("en=0: acc held at 49", acc, 49);

        // [8] Max values: 16 x 127x127 = 257984 (fits in INT32)
        reset_dut();
        en = 1;
        a_in = 8'sd127; b_in = 8'sd127;
        repeat (16) tick(1);
        check("16 x 127x127=258064 no overflow", acc, 258064);

        $display("========================================");
        $display("  Results: %0d/%0d passed", pass_cnt, pass_cnt + fail_cnt);
        if (fail_cnt == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  %0d TESTS FAILED", fail_cnt);
        $display("========================================");
        $finish;
    end

endmodule