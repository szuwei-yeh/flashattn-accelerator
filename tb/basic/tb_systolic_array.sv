// ============================================================
//  tb_systolic_array.sv  —  16×16 matrix multiply verification
//
//  Strategy:
//    1. Use a known 16×16 A and B where C = A×B can be checked
//       easily (identity, all-ones, diagonal patterns).
//    2. Compare against precomputed expected values.
//    3. Dump VCD for waveform inspection.
//
//  Run:
//    iverilog -g2012 -o tb_array.out \
//        tb/basic/tb_systolic_array.sv \
//        rtl/systolic/pe.sv \
//        rtl/systolic/systolic_array.sv \
//        rtl/systolic/array_controller.sv
//    ./tb_array.out
// ============================================================
`timescale 1ns/1ps

module tb_systolic_array;

    localparam int SIZE = 16;

    // ── DUT signals ──────────────────────────────────────────
    logic        clk   = 0;
    logic        rst_n = 0;
    logic        start = 0;
    logic        busy;
    logic        done;

    logic signed [7:0]  a_flat [SIZE*SIZE-1:0];
    logic signed [7:0]  b_flat [SIZE*SIZE-1:0];
    logic signed [31:0] acc    [SIZE-1:0][SIZE-1:0];

    // ── DUT ──────────────────────────────────────────────────
    array_controller #(.SIZE(SIZE)) dut (
        .clk    (clk),
        .rst_n  (rst_n),
        .a_flat (a_flat),
        .b_flat (b_flat),
        .start  (start),
        .busy   (busy),
        .done   (done),
        .acc    (acc)
    );

    // ── Clock ────────────────────────────────────────────────
    always #5 clk = ~clk;   // 100 MHz

    // ── Helpers ──────────────────────────────────────────────
    int pass_cnt = 0;
    int fail_cnt = 0;

    task tick(input int n = 1);
        repeat (n) @(posedge clk);
        #1;
    endtask

    task run_matmul;
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;
        // Wait for done
        @(posedge done);
        #1;
    endtask

    // Check one cell of the output matrix
    task check_cell(
        input string  test_name,
        input int     row, col,
        input logic signed [31:0] expected
    );
        if (acc[row][col] === expected) begin
            // Only print first few passes to avoid spam
            if (row < 2 && col < 2)
                $display("  PASS  %s [%0d][%0d]  got=%0d", test_name, row, col, acc[row][col]);
            pass_cnt++;
        end else begin
            $display("  FAIL  %s [%0d][%0d]  got=%0d  expected=%0d",
                     test_name, row, col, acc[row][col], expected);
            fail_cnt++;
        end
    endtask

    // ── VCD dump ─────────────────────────────────────────────
    initial begin
        $dumpfile("dump_systolic.vcd");
        $dumpvars(0, tb_systolic_array);
    end

    // module-level temp (avoids 'automatic' inside loops for Icarus)
    int exp_val;

    // ── Test sequence ─────────────────────────────────────────
    initial begin
        $display("========================================");
        $display("  Systolic Array 16×16 Test");
        $display("========================================");

        // Reset
        rst_n = 0; tick(4); rst_n = 1; tick(2);

        // ─────────────────────────────────────────────────────
        // TEST 1: A = Identity, B = all-2s
        //   C = I × B = B → every element = 2
        // ─────────────────────────────────────────────────────
        $display("\n[Test 1] A=Identity, B=all-2s → C=all-2s");
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                a_flat[r*SIZE+c] = (r == c) ? 8'sd1 : 8'sd0;   // identity
                b_flat[r*SIZE+c] = 8'sd2;                        // all 2s
            end

        run_matmul();

        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                check_cell("I×2", r, c, 32'sd2);

        // ─────────────────────────────────────────────────────
        // TEST 2: A = B = Identity
        //   C = I × I = I
        // ─────────────────────────────────────────────────────
        $display("\n[Test 2] A=B=Identity → C=Identity");
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                a_flat[r*SIZE+c] = (r == c) ? 8'sd1 : 8'sd0;
                b_flat[r*SIZE+c] = (r == c) ? 8'sd1 : 8'sd0;
            end

        run_matmul();

        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                exp_val = (r == c) ? 1 : 0;
                check_cell("IxI", r, c, exp_val);
            end

        // ─────────────────────────────────────────────────────
        // TEST 3: A = row-filled (A[r][c] = r+1), B = column-filled (B[r][c] = c+1)
        //   C[r][c] = sum_{k=0}^{15} A[r][k] * B[k][c]
        //           = (r+1) * (c+1) * SIZE
        //   e.g. C[0][0] = 1 * 1 * 16 = 16
        //        C[1][2] = 2 * 3 * 16 = 96
        // ─────────────────────────────────────────────────────
        $display("\n[Test 3] A[r][c]=r+1, B[r][c]=c+1 → C[r][c]=(r+1)*(c+1)*16");
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                a_flat[r*SIZE+c] = 8'(r + 1);
                b_flat[r*SIZE+c] = 8'(c + 1);
            end

        run_matmul();

        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                exp_val = (r+1) * (c+1) * SIZE;
                check_cell("row×col", r, c, exp_val);
            end

        // ─────────────────────────────────────────────────────
        // TEST 4: Negative values  A[r][c] = -(r+1), B = all-1s
        //   C[r][c] = sum_{k=0}^{15} -(r+1)*1 = -(r+1)*SIZE
        // ─────────────────────────────────────────────────────
        $display("\n[Test 4] Negative A, all-1s B");
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                a_flat[r*SIZE+c] = -8'(r + 1);
                b_flat[r*SIZE+c] = 8'sd1;
            end

        run_matmul();

        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                exp_val = -(r+1) * SIZE;
                check_cell("neg×pos", r, c, exp_val);
            end

        // ─────────────────────────────────────────────────────
        // TEST 5: Max values — 127×127 accumulated 16 times
        //   Each PE[r][c]: all A entries = 127, all B entries = 127
        //   C[r][c] = 16 × 127 × 127 = 257984 (fits in INT32)
        // ─────────────────────────────────────────────────────
        $display("\n[Test 5] All-127 matrices → check no INT32 overflow");
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++) begin
                a_flat[r*SIZE+c] = 8'sd127;
                b_flat[r*SIZE+c] = 8'sd127;
            end

        run_matmul();

        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                check_cell("max_val", r, c, 32'sd257984);

        // ─────────────────────────────────────────────────────
        // Summary
        // ─────────────────────────────────────────────────────
        $display("\n========================================");
        $display("  Results: %0d/%0d passed",
                 pass_cnt, pass_cnt + fail_cnt);
        if (fail_cnt == 0)
            $display("  ✅  ALL TESTS PASSED");
        else
            $display("  ❌  %0d TESTS FAILED", fail_cnt);
        $display("========================================");
        $finish;
    end

    // ── Watchdog timer ───────────────────────────────────────
    initial begin
        #50000;
        $display("ERROR: Watchdog timeout");
        $finish;
    end

endmodule