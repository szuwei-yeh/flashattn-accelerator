`timescale 1ns/1ps
module soc_job_adapter_m5 (
    input logic clk,rst_n,
    input logic command_attempt,encoding_bad,
    input logic [2:0] action,
    input m5_types_pkg::config_t candidate,
    output logic command_ok,
    output logic [31:0] reject_reason,
    output logic config_writable,
    output m5_types_pkg::config_t child_config,
    output logic child_start,child_ack,pack_start,
    input logic child_ready,child_cfg_error,child_done_status,child_output_valid,
    input logic child_error,child_rejected,
    input logic [15:0] child_words_per_head,
    input logic [6:0][31:0] child_perf,
    // M5C boundary: distinct source-retirement, outstanding, final-OKAY and fault.
    input logic source_retired,wb_final_okay,wb_outstanding,wb_error,wb_descriptor_error,
    input logic [31:0] source_words,wb_error_addr,
    input logic [3:0][31:0] wb_perf, // W bytes, AW, B, OKAY B
    output logic [2:0] phase,
    output logic compute_done,output_committed,
    output logic [31:0] output_bytes,job_id,error_code,write_error_addr,
    output logic success_event,fatal_event,reject_event,
    output logic [12:0][31:0] snapshots
);
    import m5_types_pkg::*;
    config_t active;
    logic output_job,source_seen,response_seen;
    logic [31:0] expected_bytes,expected_words,job_cycles,wb_cycles;
    logic [32:0] destination_end;
    logic destination_bad,fatal_now,finish_now,owned,release_requested;
    logic [31:0] fatal_kind;
    assign child_config=phase==IDLE ? candidate : active;
    assign output_job=active.op==0 || active.op==2;
    assign expected_bytes=active.op==0 ? 32'd4096 : 32'd256;
    assign expected_words=active.op==0 ? 32'd1024 : 32'd64;
    assign destination_end={1'b0,candidate.output_base}+(candidate.op==0 ? 33'd4096 : 33'd256);
    assign destination_bad=(candidate.op==0 || candidate.op==2) &&
        (candidate.output_base[2:0]!=0 || destination_end>33'h100000000);
    assign config_writable=phase==IDLE && child_ready && !child_error && !wb_error && !wb_descriptor_error;
    assign owned=phase==DISPATCH || phase==RUN || phase==WRITEBACK;
    always_comb begin
        reject_reason=0;
        if(command_attempt)begin
            if(encoding_bad || !(action==1 || action==2 || action==4))reject_reason=1;
            else if(action==2)begin
                if(phase!=DONE || child_error || wb_error || wb_descriptor_error)reject_reason=2;
            end else if(!config_writable)reject_reason=2;
            else if(child_cfg_error)reject_reason=3;
            else if(destination_bad)reject_reason=4;
        end
        command_ok=command_attempt && reject_reason==0;
        reject_event=command_attempt && !command_ok;
        release_requested=command_ok && action==2;
        fatal_kind=0;
        if(phase!=ERROR)begin
            if(child_error)fatal_kind=1;
            else if(wb_error)fatal_kind=2;
            else if(wb_descriptor_error || (owned && child_rejected) ||
                ((phase==WRITEBACK || (phase==DONE && output_job)) &&
                 (!child_done_status || !child_output_valid ||
                  child_words_per_head!=(active.op==0 ? 16'd256 : 16'd16))))fatal_kind=3;
            else if(phase==RUN && child_done_status && output_job &&
                (!child_output_valid || child_words_per_head!=(active.op==0 ? 16'd256 : 16'd16)))fatal_kind=3;
            else if(phase==WRITEBACK && source_seen && response_seen && !wb_outstanding &&
                (source_words!=expected_words || wb_perf[0]!=expected_bytes ||
                 wb_perf[1]==0 || wb_perf[1]!=wb_perf[2] || wb_perf[2]!=wb_perf[3]))fatal_kind=3;
        end
        fatal_now=fatal_kind!=0;
        finish_now=!fatal_now && ((phase==RUN && child_done_status && !output_job) ||
            (phase==WRITEBACK && source_seen && response_seen && !wb_outstanding));
        success_event=finish_now;
        fatal_event=fatal_now;
    end
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            phase<=IDLE;active<='0;child_start<=0;child_ack<=0;pack_start<=0;
            compute_done<=0;output_committed<=0;output_bytes<=0;job_id<=0;
            error_code<=0;write_error_addr<=0;snapshots<='0;
            source_seen<=0;response_seen<=0;job_cycles<=0;wb_cycles<=0;
        end else begin
            child_start<=0;child_ack<=0;pack_start<=0;
            if(owned)job_cycles<=job_cycles+1'b1;
            if(phase==WRITEBACK)wb_cycles<=wb_cycles+1'b1;
            if(phase==WRITEBACK)begin
                if(source_retired)source_seen<=1;
                if(wb_final_okay)response_seen<=1;
            end
            if(fatal_now || finish_now)begin
                for(int i=0;i<7;i++)snapshots[i]<=child_perf[i];
                snapshots[7]<=job_cycles+(owned ? 32'd1 : 32'd0);
                for(int i=0;i<4;i++)snapshots[i+8]<=output_job &&
                    (phase==WRITEBACK || phase==DONE) ? wb_perf[i] : 32'd0;
                snapshots[12]<=wb_cycles+(phase==WRITEBACK ? 32'd1 : 32'd0);
            end
            if(fatal_now)begin
                phase<=ERROR;error_code<=fatal_kind;
                if(fatal_kind==2)write_error_addr<=wb_error_addr;
                compute_done<=0;output_committed<=0;output_bytes<=0;
                // Never release child ownership or pretend to roll back cache/DRAM.
            end else if(finish_now)begin
                phase<=DONE;output_committed<=output_job;
                output_bytes<=output_job ? expected_bytes : 32'd0;
            end else if(release_requested)begin
                phase<=RELEASE;child_ack<=1;
                compute_done<=0;output_committed<=0;output_bytes<=0;
            end else if(command_ok && action!=2)begin
                active<=candidate;phase<=DISPATCH;job_id<=job_id+1'b1;
                snapshots<='0;compute_done<=0;output_committed<=0;output_bytes<=0;
                source_seen<=0;response_seen<=0;job_cycles<=0;wb_cycles<=0;
            end else case(phase)
                DISPATCH:begin child_start<=1;phase<=RUN;end
                RUN:if(child_done_status && output_job)begin
                    compute_done<=1;pack_start<=1;phase<=WRITEBACK;
                end
                RELEASE:if(child_ready)phase<=IDLE;
                default:begin end
            endcase
        end
    end
endmodule
