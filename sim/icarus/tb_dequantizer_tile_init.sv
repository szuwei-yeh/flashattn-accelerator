`timescale 1ns/1ps
module tb_dequantizer_tile_init;
    parameter int LANES = 32;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0, valid_in = 0;
    reg signed [31:0] data_in [255:0];
    reg signed [31:0] combined_scale = 256;
    wire signed [15:0] data_out [255:0];
    wire busy, done, tile_ready;
    dequantizer_tile #(.LANES(LANES)) dut (.*);

    task automatic prepare(input integer sign);
        @(negedge clk);
        combined_scale = sign * 256;
        for (integer i = 0; i < 256; i++) data_in[i] = i*13-1664;
        valid_in = 1;
        @(negedge clk); valid_in = 0;
        if (!busy || tile_ready || done) $fatal(1,"request did not invalidate tile");
    endtask

    task automatic complete(input integer sign);
        integer cycles;
        cycles=0;
        while (!done && cycles < 40) begin
            if (tile_ready !== 1'b0) $fatal(1,"partial tile published");
            @(negedge clk); cycles++;
        end
        if (done !== 1'b1 || busy !== 1'b0 || tile_ready !== 1'b1)
            $fatal(1,"missing complete tile");
        for (integer i = 0; i < 256; i++)
            if (data_out[i] !== 16'(sign*(i*13-1664)))
                $fatal(1,"index=%0d got=%h expected=%h",i,data_out[i],16'(sign*(i*13-1664)));
        @(negedge clk);
        if (done || !tile_ready) $fatal(1,"completion must pulse; ready must hold");
    endtask

    initial begin
        repeat(2) @(negedge clk);
        if (!$isunknown(dut.data_out[255])) $fatal(1,"requires unknown initial score storage");
        rst_n=1;
        prepare(1); complete(1);
        prepare(-1);
        repeat(3) @(negedge clk);
        rst_n=0;
        repeat(2) @(negedge clk);
        if (busy || done || tile_ready) $fatal(1,"reset did not abort partial tile");
        rst_n=1;
        prepare(-1); complete(-1);
        prepare(1); complete(1);
        $display("RESULT: PASS four-state score initialization, complete-tile publication and partial-reset recovery; lanes=%0d",LANES);
        $finish;
    end
    initial begin
        #10000;
        $fatal(1,"timeout");
    end
endmodule
