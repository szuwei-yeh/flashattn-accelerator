// Bounded MODULE-boundary control properties. All child/DMA ports arbitrary.
// No memory/payload/real-AXI or integrated-compute theorem is asserted.
reg f_past_valid=0;
always @(posedge clk) begin
 f_past_valid<=1;
 if(!f_past_valid) assume(!rst_n);
 if(rst_n) begin
  assert(!success_event || !fatal_now);
  if(success_event && output_job)begin
   assert(phase==WRITEBACK && source_seen && response_seen && !wb_outstanding);
   assert(child_done_status && child_output_valid);
   assert(child_words_per_head==(active.op==0 ? 16'd256 : 16'd16));
   assert(source_words==expected_words && wb_perf[0]==expected_bytes);
   assert(wb_perf[1]!=0 && wb_perf[1]==wb_perf[2] && wb_perf[2]==wb_perf[3]);
  end
  if(f_past_valid && $past(rst_n))begin
   if($past(owned || phase==DONE || phase==ERROR))assert(active==$past(active));
   if($past(fatal_now))begin
    assert(phase==ERROR && !output_committed && output_bytes==0);
    assert(!child_ack);
   end
   if($past(phase==ERROR))assert(phase==ERROR);
   if($past(finish_now))assert(phase==DONE);
   if(child_ack)assert($past(release_requested));
  end
 end
end
(* keep *) wire f_cover_success=rst_n && success_event && output_job;
(* keep *) wire f_cover_owner_fault=rst_n && phase==WRITEBACK && fatal_kind==3 && !child_output_valid;
(* keep *) wire f_cover_conservation_fault=rst_n && phase==WRITEBACK && source_seen && response_seen && !wb_outstanding && child_output_valid && child_done_status && source_words!=expected_words && fatal_kind==3;
