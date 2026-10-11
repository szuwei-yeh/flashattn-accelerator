// Verification-only overlay in the actual cache module. SRAM contents abstracted.
// Harness assumptions reproduce the top's accepted-command contract, not AXI.
reg f_past_valid=0;
always @(posedge clk) begin
 f_past_valid<=1;
 if(!f_past_valid) assume(!rst_n); else assume(rst_n);
 if(rst_n) begin
  assume(!cmd_begin || (!owner && !fault_seen));
  assume(!cmd_commit || owner);
  assume(!(cmd_begin && cmd_commit));
  assume(!cmd_begin || ((cmd_op==PREFILL && cmd_length==16) ||
   (cmd_op==APPEND && cmd_length==cache_length && cmd_length<64) ||
   (cmd_op==DECODE && cache_valid && cmd_length==cache_length) || cmd_op==CLEAR));
  // The actual top commits a K/V pair only after ordered stores complete.
  assume(!cmd_commit || (op!=APPEND && op!=PREFILL) || pair_complete);
  assert(cache_length<=64);
  assert(!write_accept || owner);
  assert(!write_accept || (w_addr[3:0]==0 && {1'b0,w_addr}+13'd16<=13'(64*HEAD_DIM)));
  assert(!read_accept || (owner && !cmd_fault && core_r_addr[3:0]==0 &&
   read_end<=(op==PREFILL ? 13'(pending_length)*13'(HEAD_DIM) : committed_end)));
  assert(!fault_seen || !cache_valid);
  assert(!pair_complete || (k_stored==stripes_required && v_stored==stripes_required));
  if(f_past_valid && $past(rst_n))begin
   if($past(owner && op==APPEND && !cmd_fault && !cmd_commit))begin
    assert(cache_length_reg==$past(cache_length_reg));
    assert(cache_epoch==$past(cache_epoch));
   end
   if($past(cmd_commit && owner && op==APPEND && pair_complete && !cmd_fault && !cmd_begin))begin
    assert(cache_valid_reg);
    assert(cache_length_reg==$past(pending_length)+1);
   end
   if($past(cmd_commit && owner && op==DECODE && !cmd_fault && !cmd_begin))begin
    assert(cache_length_reg==$past(cache_length_reg));assert(cache_epoch==$past(cache_epoch));
   end
   if($past(cmd_fault || (cmd_commit && op==CLEAR && !cmd_begin)))begin
    assert(!cache_valid_reg);assert(cache_length_reg==0);
   end
   if($past(cmd_fault))assert(fault_seen);
  end
  cover(cmd_commit && op==APPEND && pair_complete);
  cover(cmd_commit && op==PREFILL && pair_complete);
  cover(cmd_commit && op==DECODE && cache_valid);
  cover(cmd_commit && op==CLEAR);
  cover(cmd_fault && owner);
  cover(read_accept && op==DECODE);
 end
end
