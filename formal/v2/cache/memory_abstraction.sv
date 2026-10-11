// Formal-only abstraction: SRAM payload arbitrary; no numerical/content proof.
module banked_scratchpad(input clk,w_en,w_vec,input [11:0] w_addr,
 input [7:0] w_sdata,input [127:0] w_vdata,input r_en,r_vec,
 input [11:0] r_addr,output [7:0] r_sdata,output [127:0] r_vdata);
 (* anyseq *) reg [127:0] payload;
 assign r_sdata=payload[7:0];assign r_vdata=payload;
endmodule
