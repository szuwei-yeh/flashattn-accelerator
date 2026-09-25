`timescale 1ns/1ps
module tb_output_buffer_init;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0, first_tile = 0, accum_en = 0, rescale_en = 0, norm_en = 0, re_ext = 0;
    reg [3:0] addr = 0, rescale_addr = 0, norm_addr = 0, raddr_ext = 0;
    reg signed [31:0] data_in = 0;
    reg [15:0] rescale_q88 = 256;
    reg [31:0] norm_divisor = 256;
    wire signed [31:0] rdata_ext;
    output_buffer #(.DEPTH(16)) dut (.*);
    task automatic update(input bit first, input integer a, input integer value,
                          input bit accum, input bit rescale, input bit norm,
                          input integer factor, input integer divisor);
        @(negedge clk);
        first_tile=first; addr=a; rescale_addr=a; norm_addr=a; data_in=value;
        accum_en=accum; rescale_en=rescale; norm_en=norm;
        rescale_q88=factor; norm_divisor=divisor; re_ext=0;
        @(negedge clk); accum_en=0; rescale_en=0; norm_en=0;
        @(negedge clk);
    endtask
    task automatic expect_word(input integer a, input integer value);
        @(negedge clk); re_ext=1; raddr_ext=a;
        @(negedge clk);
        if (rdata_ext !== value) $fatal(1,"addr=%0d got=%h expected=%h",a,rdata_ext,value);
        re_ext=0;
    endtask
    initial begin
        repeat(2) @(negedge clk); rst_n=1;
        // In a four-state simulator the memory has never been initialized.
        if (!$isunknown(dut.u_sram_o.mem[3])) $fatal(1,"test requires unknown SRAM initial state");
        update(1,3,123,1,1,0,256,256); expect_word(3,123);
        update(0,3,-10,1,1,0,128,256); expect_word(3,51); // truncate 61.5 then add
        update(0,3,0,0,0,1,256,512); expect_word(3,25); // signed divide toward zero
        update(1,3,-123,1,1,0,256,256); expect_word(3,-123); // stale data overwritten
        update(0,3,0,0,0,1,256,512); expect_word(3,-61);
        update(0,3,0,0,0,1,256,0); expect_word(3,-61); // zero divisor preserves value
        update(1,4,0,0,1,0,256,256); expect_word(4,0); // legacy two-pass initialization
        update(1,4,7,1,0,0,256,256); expect_word(4,7);
        update(1,5,32'h7fffffff,1,1,0,256,256);
        update(0,5,1,1,1,0,256,256); expect_word(5,32'h80000000); // specified wrap
        @(negedge clk); rst_n=0;
        repeat(2) @(negedge clk); rst_n=1;
        expect_word(3,-61); // reset does not erase memory
        update(1,3,42,1,1,0,256,256); expect_word(3,42);
        $display("RESULT: PASS four-state initialization, stale overwrite, reset, wrap, normalization");
        $finish;
    end
endmodule
