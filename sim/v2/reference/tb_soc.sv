`timescale 1ns/1ps
// Isolated M5D wiring. All arithmetic/cache/CSR/transport children unchanged.
module tb_soc_m6b3 #(
    parameter bit COMPACT_DECODE_OUTPUT=1'b1,
    parameter int DEQUANT_LANES=256,
    parameter int HEAD_DIM=16,
    parameter int EXPECTED_LANES=DEQUANT_LANES
)(
    input logic clk,rst_n,
    input logic [11:0] s_axil_awaddr,s_axil_araddr,
    input logic [2:0] s_axil_awprot,s_axil_arprot,
    input logic s_axil_awvalid,s_axil_wvalid,s_axil_bready,s_axil_arvalid,s_axil_rready,
    output logic s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid,
    input logic [31:0] s_axil_wdata,
    input logic [3:0] s_axil_wstrb,
    output logic [1:0] s_axil_bresp,s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic irq,
    output logic [31:0] m_araddr,m_awaddr,
    output logic [7:0] m_arlen,m_awlen,
    output logic [2:0] m_arsize,m_awsize,
    output logic [1:0] m_arburst,m_awburst,
    output logic m_arvalid,m_awvalid,
    input logic m_arready,m_awready,
    input logic [63:0] m_rdata,
    input logic [1:0] m_rresp,m_bresp,
    input logic m_rlast,m_rvalid,m_wready,m_bvalid,
    output logic m_rready,m_bready,
    output logic [63:0] m_wdata,
    output logic [7:0] m_wstrb,
    output logic m_wlast,m_wvalid
,
output logic [2:0] obs_phase,obs_action,
output logic obs_command,obs_ack,obs_success,obs_fatal,obs_child_ack,obs_pack_start,obs_wb_accept,
output logic [1:0] obs_op,
output logic [6:0] obs_length,
output logic [31:0] obs_job_id,obs_error,obs_output_bytes,
output logic [440:0] obs_config,
output logic child_done_status,child_output_valid,
output logic obs_wb_done,obs_wb_busy,obs_src_valid,obs_pack_done,
output logic [31:0] obs_source_words,wb_cycles,wb_aw,wb_w,wb_bytes,wb_b,wb_okay,wb_aw_stall,wb_w_stall,wb_b_wait,wb_src_wait,
    output logic [2-1:0] cache_valid,
    output logic [6:0] cache_length[2],
    output logic [31:0] cache_epoch[2],
    output logic [31:0] perf_cycles,perf_dma_bytes,perf_ar,perf_rbeats,perf_cache_reads,perf_cache_writes,perf_copy_cycles,
    output logic obs_accept,obs_commit,obs_ar,obs_r,obs_store,obs_core_start,obs_core_done,obs_core_active,
    output logic [1:0] obs_qhead,obs_kvhead,obs_loadhead,obs_dst,
    output logic [3:0] obs_state,
    output logic [11:0] obs_cache_addr,obs_core_store_addr,
    output logic [$clog2(16*HEAD_DIM):0] obs_copy_word,
    output logic obs_copy_store,obs_cache_write,obs_cache_read,obs_array_start,obs_array_compute,
    output logic [31:0] obs_araddr,
    output logic [7:0] obs_arlen,
    output logic [31:0] mon_core_stores,mon_result_stores,mon_cache_checks,
    output logic [31:0] reads_by_kv[2],writes_by_kv[2],reads_by_q[4],
    output logic obs_dq_issue,obs_dq_retire,obs_dq_publish,obs_dq_consume,
    output logic [3:0] obs_dq_group
);
    localparam int Q_HEADS=4,KV_HEADS=2,COUNT_W=$clog2(16*HEAD_DIM)+1;
flash_attn_soc_m6b3 #(.COMPACT_DECODE_OUTPUT(COMPACT_DECODE_OUTPUT),.DEQUANT_LANES(DEQUANT_LANES),.HEAD_DIM(HEAD_DIM)) u_dut(.clk(clk),.rst_n(rst_n),.s_axil_awaddr(s_axil_awaddr),.s_axil_araddr(s_axil_araddr),.s_axil_awprot(s_axil_awprot),.s_axil_arprot(s_axil_arprot),.s_axil_awvalid(s_axil_awvalid),.s_axil_wvalid(s_axil_wvalid),.s_axil_bready(s_axil_bready),.s_axil_arvalid(s_axil_arvalid),.s_axil_rready(s_axil_rready),.s_axil_awready(s_axil_awready),.s_axil_wready(s_axil_wready),.s_axil_bvalid(s_axil_bvalid),.s_axil_arready(s_axil_arready),.s_axil_rvalid(s_axil_rvalid),.s_axil_wdata(s_axil_wdata),.s_axil_wstrb(s_axil_wstrb),.s_axil_bresp(s_axil_bresp),.s_axil_rresp(s_axil_rresp),.s_axil_rdata(s_axil_rdata),.irq(irq),.m_araddr(m_araddr),.m_awaddr(m_awaddr),.m_arlen(m_arlen),.m_awlen(m_awlen),.m_arsize(m_arsize),.m_awsize(m_awsize),.m_arburst(m_arburst),.m_awburst(m_awburst),.m_arvalid(m_arvalid),.m_awvalid(m_awvalid),.m_arready(m_arready),.m_awready(m_awready),.m_rdata(m_rdata),.m_rresp(m_rresp),.m_bresp(m_bresp),.m_rlast(m_rlast),.m_rvalid(m_rvalid),.m_wready(m_wready),.m_bvalid(m_bvalid),.m_rready(m_rready),.m_bready(m_bready),.m_wdata(m_wdata),.m_wstrb(m_wstrb),.m_wlast(m_wlast),.m_wvalid(m_wvalid));
assign obs_phase=u_dut.phase;
assign obs_action=u_dut.action;
assign obs_command=u_dut.command_ok && u_dut.action!=2;
assign obs_ack=u_dut.command_ok && u_dut.action==2;
assign obs_success=u_dut.success_event;
assign obs_fatal=u_dut.fatal_event;
assign obs_child_ack=u_dut.child_ack;
assign obs_pack_start=u_dut.pack_start;
assign obs_wb_accept=u_dut.wb_desc_valid && u_dut.wb_desc_ready;
assign obs_op=u_dut.child_config.op;
assign obs_length=u_dut.child_config.kv_length;
assign obs_config=u_dut.child_config;
assign obs_job_id=u_dut.job_id;
assign obs_error=u_dut.error_code;
assign obs_output_bytes=u_dut.output_bytes;
assign child_done_status=u_dut.child_done_status;
assign child_output_valid=u_dut.child_output_valid;
assign cache_valid=u_dut.cache_valid;
assign obs_wb_done=u_dut.wb_done;
assign obs_wb_busy=u_dut.wb_outstanding;
assign obs_src_valid=u_dut.src_valid;
assign obs_pack_done=u_dut.pack_done;
assign obs_source_words=u_dut.source_words;
assign perf_cycles=u_dut.child_perf[0];
assign perf_dma_bytes=u_dut.child_perf[1];
assign perf_ar=u_dut.child_perf[2];
assign perf_rbeats=u_dut.child_perf[3];
assign perf_cache_reads=u_dut.child_perf[4];
assign perf_cache_writes=u_dut.child_perf[5];
assign perf_copy_cycles=u_dut.child_perf[6];
assign wb_cycles=u_dut.u_write.perf_cycles;
assign wb_aw=u_dut.u_write.perf_aw;
assign wb_w=u_dut.u_write.perf_wbeats;
assign wb_bytes=u_dut.u_write.perf_wbytes;
assign wb_b=u_dut.u_write.perf_b;
assign wb_okay=u_dut.u_write.perf_okay_b;
assign wb_aw_stall=u_dut.u_write.perf_aw_stall;
assign wb_w_stall=u_dut.u_write.perf_w_stall;
assign wb_b_wait=u_dut.u_write.perf_b_wait;
assign wb_src_wait=u_dut.u_write.perf_src_wait;
for(genvar h=0;h<2;h++)begin : g_metadata
 assign cache_length[h]=u_dut.cache_length[h];assign cache_epoch[h]=u_dut.cache_epoch[h];
end
    assign obs_accept=u_dut.u_engine.accepted;assign obs_commit=u_dut.u_engine.completion;
    assign obs_ar=m_arvalid && m_arready;assign obs_r=m_rvalid && m_rready;
    assign obs_araddr=m_araddr;assign obs_arlen=m_arlen;
    assign obs_store=u_dut.u_engine.u_core.u_out_buf.we;assign obs_core_store_addr=u_dut.u_engine.u_core.u_out_buf.waddr;
    assign obs_core_start=u_dut.u_engine.core_start;assign obs_core_done=u_dut.u_engine.core_done;assign obs_core_active=u_dut.u_engine.core_active;
    assign obs_qhead=u_dut.u_engine.q_head;assign obs_kvhead=u_dut.u_engine.selected_kv;assign obs_loadhead=u_dut.u_engine.load_head;
    assign obs_state=u_dut.u_engine.state;assign obs_dst=u_dut.u_engine.w_dst;assign obs_cache_addr=u_dut.u_engine.w_addr;
    assign obs_copy_word=u_dut.u_engine.copy_word;assign obs_copy_store=u_dut.u_engine.busy && u_dut.u_engine.state==4'd11;
    assign obs_cache_write=|u_dut.u_engine.write_accept;assign obs_cache_read=|u_dut.u_engine.read_accept;
    assign obs_array_start=u_dut.u_engine.u_core.array_start;
    assign obs_array_compute=u_dut.u_engine.u_core.u_array_ctrl.state==2'd2;
    logic [127:0] expected_k[KV_HEADS][256],expected_v[KV_HEADS][256];
    logic pending_read;logic [1:0] pending_head;logic [7:0] pending_addr;
    logic [12:0] per_head_stores; logic [$clog2(16*HEAD_DIM):0] per_head_result;
    logic [Q_HEADS-1:0] completed_heads;
    logic [6:0] old_length;
    logic [31:0] k_seen[KV_HEADS],v_seen[KV_HEADS];
    logic [31:0] expected_stripes;
    assign expected_stripes=u_dut.u_engine.r_op==0 ? 32'(HEAD_DIM) : 32'(HEAD_DIM/16);
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            pending_read<=0;pending_head<=0;pending_addr<=0;mon_cache_checks<=0;mon_core_stores<=0;mon_result_stores<=0;
            per_head_stores<=0;per_head_result<=0;completed_heads<=0;old_length<=0;
            for(int h=0;h<KV_HEADS;h++)begin k_seen[h]<=0;v_seen[h]<=0;reads_by_kv[h]<=0;writes_by_kv[h]<=0;end
            for(int h=0;h<Q_HEADS;h++)reads_by_q[h]<=0;
        end else begin
            if(obs_accept)begin
                mon_cache_checks<=0;mon_core_stores<=0;mon_result_stores<=0;completed_heads<=0;old_length<=u_dut.child_config.kv_length;
                for(int h=0;h<KV_HEADS;h++)begin k_seen[h]<=0;v_seen[h]<=0;reads_by_kv[h]<=0;writes_by_kv[h]<=0;end
            for(int h=0;h<Q_HEADS;h++)reads_by_q[h]<=0;
            end
            if(pending_read)begin
                if(u_dut.u_engine.k_data[int'(pending_head)]!==expected_k[int'(pending_head)][pending_addr] ||
                   u_dut.u_engine.v_data[int'(pending_head)]!==expected_v[int'(pending_head)][pending_addr])$fatal(1,"M4_SAFETY stale cache data head=%0d addr=%0d",pending_head,pending_addr);
                mon_cache_checks<=mon_cache_checks+1;
            end
            pending_read<=obs_cache_read;pending_head<=obs_kvhead;pending_addr<=u_dut.u_engine.kv_r_addr[11:4];
            if(obs_cache_read)begin
                if($countones(u_dut.u_engine.read_accept)!=1)$fatal(1,"M4_SAFETY multiple physical read owners");
                reads_by_kv[int'(obs_kvhead)]<=reads_by_kv[int'(obs_kvhead)]+1;reads_by_q[int'(obs_qhead)]<=reads_by_q[int'(obs_qhead)]+1;
                if(int'(obs_kvhead)!=int'(obs_qhead)/(Q_HEADS/KV_HEADS))$fatal(1,"M4_SAFETY wrong KV mapping");
                if({1'b0,u_dut.u_engine.kv_r_addr}+13'd16>13'(int'(u_dut.u_engine.r_length)*HEAD_DIM))$fatal(1,"M4_SAFETY invalid tail read");
            end
            if(u_dut.u_engine.w_en && u_dut.u_engine.w_dst!=0 && $countones(u_dut.u_engine.write_accept)!=1)$fatal(1,"M4_SAFETY missing accepted DMA cache write");
            if(obs_cache_write)begin
                if($countones(u_dut.u_engine.write_accept)!=1)$fatal(1,"M4_SAFETY multiple physical write owners");
                writes_by_kv[int'(obs_loadhead)]<=writes_by_kv[int'(obs_loadhead)]+1;
                if(obs_dst==1)begin
                    if(v_seen[int'(obs_loadhead)]!=0 || k_seen[int'(obs_loadhead)]>=expected_stripes)$fatal(1,"M4_SAFETY K write order");
                    expected_k[int'(obs_loadhead)][obs_cache_addr[11:4]]<=u_dut.u_engine.w_data;k_seen[int'(obs_loadhead)]<=k_seen[int'(obs_loadhead)]+1;
                end else begin
                    if(k_seen[int'(obs_loadhead)]!=expected_stripes || v_seen[int'(obs_loadhead)]>=expected_stripes)$fatal(1,"M4_SAFETY V write order");
                    expected_v[int'(obs_loadhead)][obs_cache_addr[11:4]]<=u_dut.u_engine.w_data;v_seen[int'(obs_loadhead)]<=v_seen[int'(obs_loadhead)]+1;
                end
            end
            if(u_dut.u_engine.busy && (u_dut.u_engine.r_op==0 || u_dut.u_engine.r_op==1) && !obs_commit)
                for(int h=0;h<KV_HEADS;h++)if(cache_length[h]!=(u_dut.u_engine.r_op==0 ? 7'd0 : old_length))$fatal(1,"M4_SAFETY premature KV publication");
            if(u_dut.u_engine.core_active && u_dut.u_engine.u_core.query_position_reg!=u_dut.u_engine.r_position)$fatal(1,"M4_SAFETY absolute query position");
            if(obs_core_start)begin per_head_stores<=0;per_head_result<=0;end
            if(obs_store)begin mon_core_stores<=mon_core_stores+1;per_head_stores<=per_head_stores+1'b1;end
            if(obs_core_done && int'(per_head_stores)!=((int'(u_dut.u_engine.r_length)+15)/16+1)*int'(u_dut.u_engine.words))$fatal(1,"M4_SAFETY core DONE before stores");
            if(obs_copy_store)begin
                if(obs_copy_word!=per_head_result)$fatal(1,"M4_SAFETY wrong output address");
                if(!u_dut.u_engine.pipe_empty || u_dut.u_engine.core_active || int'(per_head_stores)!=((int'(u_dut.u_engine.r_length)+15)/16+1)*int'(u_dut.u_engine.words))$fatal(1,"M4_SAFETY result before core retirement");
                per_head_result<=per_head_result+1'b1;mon_result_stores<=mon_result_stores+1;
                if(obs_copy_word+COUNT_W'(1)==u_dut.u_engine.words)completed_heads[int'(obs_qhead)]<=1;
            end
            if(obs_commit)begin
                if((u_dut.u_engine.r_op==0 || u_dut.u_engine.r_op==2) && (completed_heads!={Q_HEADS{1'b1}} || mon_result_stores!=Q_HEADS*int'(u_dut.u_engine.words)))$fatal(1,"M4_SAFETY premature completion");
                if(u_dut.u_engine.r_op==0 || u_dut.u_engine.r_op==1)
                    for(int h=0;h<KV_HEADS;h++)if(k_seen[h]!=expected_stripes || v_seen[h]!=expected_stripes)$fatal(1,"M4_SAFETY missing paired cache writes");
                $display("STORE_JOB id=%0d op=%0d length=%0d core_stores=%0d result_stores=%0d cache_checks=%0d",obs_job_id,obs_op,obs_length,mon_core_stores,mon_result_stores,mon_cache_checks);
            end
        end
    end


    // Read-only score assembly observers; never drive DUT or cache storage.
    initial begin
        if(u_dut.u_engine.u_core.DEQUANT_LANES!=EXPECTED_LANES)
            $fatal(1,"M6B1_SCORE incorrect physical resource selection");
        $display("RESOURCE lanes=%0d expected=%0d core_instances=1 physical_KV_heads=2",DEQUANT_LANES,EXPECTED_LANES);
    end
    logic dq_request,dq_publish,dq_issue,dq_retire;
    logic [3:0] dq_issue_group,dq_retire_group;
    assign dq_request=u_dut.u_engine.u_core.array_done &&
        !u_dut.u_engine.u_core.is_pv_phase && u_dut.u_engine.u_core.is_last_qk_chunk;
    assign dq_publish=&u_dut.u_engine.u_core.dequant_valid;
    assign obs_dq_issue=dq_issue;assign obs_dq_retire=dq_retire;
    assign obs_dq_publish=dq_publish;assign obs_dq_group=dq_retire_group;
    assign obs_dq_consume=u_dut.u_engine.u_core.tile_valid_pulse;
    if(DEQUANT_LANES==16)begin : g_dq_observer
        assign dq_issue=u_dut.u_engine.u_core.gen_shared_dequant.u_dequant_tile.issue_active;
        assign dq_retire=u_dut.u_engine.u_core.gen_shared_dequant.u_dequant_tile.retire_valid;
        assign dq_issue_group=u_dut.u_engine.u_core.gen_shared_dequant.u_dequant_tile.issue_group;
        assign dq_retire_group=u_dut.u_engine.u_core.gen_shared_dequant.u_dequant_tile.retire_group;
    end else begin : g_dq_observer
        assign dq_issue=dq_request;assign dq_retire=dq_publish;
        assign dq_issue_group=0;assign dq_retire_group=0;
    end
    function automatic logic signed [15:0] independent_score(
        input logic signed [31:0] dot,scale);
        logic signed [63:0] value;
        value=($signed(dot)*$signed(scale)+64'sd128)>>>8;
        if(value>64'sd32767)return 16'sh7fff;
        if(value< -64'sd32768)return 16'sh8000;
        return value[15:0];
    endfunction
    logic score_pending,score_complete;
    logic signed [31:0] held_dot[256],held_scale;
    int issue_seen,retire_seen;
    int dq_requests,dq_issues,dq_retires,dq_consumes,dq_words,dq_wait;
    int softmax_wait,update_cycles,norm_cycles;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            score_pending<=0;score_complete<=0;held_scale<=0;
            issue_seen<=0;retire_seen<=0;
            dq_requests<=0;dq_issues<=0;dq_retires<=0;dq_consumes<=0;dq_words<=0;dq_wait<=0;
            softmax_wait<=0;update_cycles<=0;norm_cycles<=0;
        end else if(obs_accept)begin
            if(score_pending)$fatal(1,"M6B1_SCORE stale pending score at new job");
            score_complete<=0;issue_seen<=0;retire_seen<=0;
            dq_requests<=0;dq_issues<=0;dq_retires<=0;dq_consumes<=0;dq_words<=0;dq_wait<=0;
            softmax_wait<=0;update_cycles<=0;norm_cycles<=0;
        end else begin
            if(dq_request)begin
                if(score_pending)$fatal(1,"M6B1_SCORE overlapping score request");
                score_pending<=1;score_complete<=0;issue_seen<=0;retire_seen<=0;
                held_scale<=u_dut.u_engine.u_core.combined_scale_reg;
                for(int i=0;i<256;i++)held_dot[i]<=u_dut.u_engine.u_core.array_acc[i];
                dq_requests<=dq_requests+1;
            end
            if(score_pending && !dq_publish && !score_complete)begin
                dq_wait<=dq_wait+1;
                if(u_dut.u_engine.u_core.array_start)$fatal(1,"M6B1_SCORE array reused before scores assembled");
                if(held_scale!==u_dut.u_engine.u_core.combined_scale_reg)$fatal(1,"M6B1_SCORE changed captured score scale");
                for(int i=0;i<256;i++)
                    if(held_dot[i]!==u_dut.u_engine.u_core.array_acc[i])$fatal(1,"M6B1_SCORE changed QK accumulator");
            end
            if(dq_issue)begin
                dq_issues<=dq_issues+1;
                if(DEQUANT_LANES==16)begin
                    if(!score_pending || int'(dq_issue_group)!=issue_seen)$fatal(1,"M6B1_SCORE issue order");
                    issue_seen<=issue_seen+1;
                end
            end
            if(dq_retire)begin
                dq_retires<=dq_retires+1;dq_words<=dq_words+DEQUANT_LANES;
                if(DEQUANT_LANES==16)begin
                    if(!score_pending || int'(dq_retire_group)!=retire_seen || retire_seen>=issue_seen)$fatal(1,"M6B1_SCORE retire order");
                    retire_seen<=retire_seen+1;
                end
            end
            if(dq_publish)begin
                if(!score_pending || (DEQUANT_LANES==16 && (issue_seen!=16 || retire_seen!=16)))
                    $fatal(1,"M6B1_SCORE incomplete score publication");
                for(int i=0;i<256;i++)
                    if(u_dut.u_engine.u_core.dequant_out[i]!==independent_score(held_dot[i],held_scale))
                        $fatal(1,"M6B1_SCORE exact published score mismatch lane=%0d",i);
                score_complete<=1;
            end
            if(u_dut.u_engine.u_core.tile_valid_pulse)begin
                if(!score_pending || !(score_complete || dq_publish))
                    $fatal(1,"M6B1_SCORE incomplete score tile consumed");
                score_pending<=0;score_complete<=0;dq_consumes<=dq_consumes+1;
            end
            if(obs_core_active && u_dut.u_engine.u_core._dbg_state==4)softmax_wait<=softmax_wait+1;
            if(obs_core_active && u_dut.u_engine.u_core._dbg_state==7)update_cycles<=update_cycles+1;
            if(obs_core_active && u_dut.u_engine.u_core._dbg_state==9)norm_cycles<=norm_cycles+1;
            if(obs_commit)begin
                if(score_pending || dq_requests!=dq_consumes ||
                    dq_words!=dq_requests*256 ||
                    dq_requests!=((obs_op==0 || obs_op==2) ? 4*((int'(obs_length)+15)/16) : 0))
                    $fatal(1,"M6B1_SCORE completion before full score retirement");
                $display("DQ_JOB id=%0d op=%0d length=%0d lanes=%0d compact=%0d requests=%0d issues=%0d retires=%0d words=%0d consumes=%0d dq_wait=%0d softmax_wait=%0d update_cycles=%0d norm_cycles=%0d",
                    obs_job_id,obs_op,obs_length,DEQUANT_LANES,COMPACT_DECODE_OUTPUT,
                    dq_requests,dq_issues,dq_retires,dq_words,dq_consumes,dq_wait,softmax_wait,update_cycles,norm_cycles);
            end
        end
    end
endmodule
