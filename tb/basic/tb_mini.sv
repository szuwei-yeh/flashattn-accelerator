`timescale 1ns/1ps

module tb_mini;
    localparam int SIZE = 4;

    logic        clk   = 0;
    logic        rst_n = 0;
    logic        en    = 0;
    logic        clear = 0;

    logic signed [7:0]  a_in [SIZE-1:0];
    logic signed [7:0]  b_in [SIZE-1:0];
    logic signed [31:0] acc  [SIZE*SIZE-1:0];  // flat: acc[r*SIZE+c]

    systolic_array #(.SIZE(SIZE)) dut (.*);

    always #5 clk = ~clk;

    int pass_cnt = 0;
    int fail_cnt = 0;

    task tick(input int n=1); repeat(n) @(posedge clk); #1; endtask

    task check32(input string name,
                 input logic signed [31:0] got, exp);
        if (got === exp) begin
            $display("  PASS  %s  got=%0d", name, got);
            pass_cnt++;
        end else begin
            $display("  FAIL  %s  got=%0d  exp=%0d", name, got, exp);
            fail_cnt++;
        end
    endtask

    initial begin
        $dumpfile("dump_mini.vcd");
        $dumpvars(0, tb_mini);
    end

    initial begin
        $display("====================================");
        $display("  Mini Systolic Array Debug Test");
        $display("====================================");

        for (int i = 0; i < SIZE; i++) begin
            a_in[i] = '0; b_in[i] = '0;
        end

        // ── Reset ────────────────────────────────────────────
        rst_n = 0; tick(4); rst_n = 1; tick(1);
        $display("[1] After reset:");
        check32("acc[0][0]=0", acc[0*SIZE+0], 0);
        check32("acc[1][1]=0", acc[1*SIZE+1], 0);

        // ── Single MAC: a[0]=3, b[0]=5 → PE[0][0]=15 ────────
        en = 1;
        a_in[0] = 8'sd3; b_in[0] = 8'sd5;
        tick(1);
        $display("[2] After 1 MAC (3x5):");
        check32("acc[0][0]=15", acc[0*SIZE+0], 15);
        check32("acc[0][1]=0",  acc[0*SIZE+1], 0);

        // ── Clear ────────────────────────────────────────────
        clear = 1; tick(1); clear = 0;
        $display("[3] After clear:");
        check32("acc[0][0]=0", acc[0*SIZE+0], 0);

        // ── 4×4 Identity × all-2 → all-2 ─────────────────────
        // Manual skewed feeding over 2*SIZE-1 = 7 cycles
        en = 0; rst_n = 0; tick(2); rst_n = 1; tick(1);
        en = 1; clear = 1; tick(1); clear = 0;

        $display("[4] 4x4 I x 2 -> all-2 (manual skewing):");
        for (int cycle = 0; cycle < 2*SIZE-1; cycle++) begin
            for (int i = 0; i < SIZE; i++) begin
                // a_in[i] = A[i][cycle-i], A=identity → 1 only if i==cycle-i
                if (cycle >= i && (cycle-i) < SIZE)
                    a_in[i] = ((i) == (cycle-i)) ? 8'sd1 : 8'sd0;
                else
                    a_in[i] = 8'sd0;

                // b_in[i] = B[cycle-i][i], B=all-2
                if (cycle >= i && (cycle-i) < SIZE)
                    b_in[i] = 8'sd2;
                else
                    b_in[i] = 8'sd0;
            end
            tick(1);
        end

        begin
            int wrong = 0;
            for (int r = 0; r < SIZE; r++) begin
                for (int c = 0; c < SIZE; c++) begin
                    if (acc[r*SIZE+c] !== 32'sd2) begin
                        $display("  FAIL C[%0d][%0d]=%0d (exp 2)", r, c, acc[r*SIZE+c]);
                        wrong++; fail_cnt++;
                    end else pass_cnt++;
                end
            end
            if (wrong == 0) $display("  PASS all 16 elements = 2");
        end

        $display("====================================");
        $display("  %0d/%0d passed", pass_cnt, pass_cnt+fail_cnt);
        if (fail_cnt == 0) $display("  ALL PASSED");
        else               $display("  %0d FAILED", fail_cnt);
        $display("====================================");
        $finish;
    end
endmodule