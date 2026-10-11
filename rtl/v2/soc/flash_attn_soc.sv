`timescale 1ns/1ps
// Isolated M5D wiring. All arithmetic/cache/CSR/transport children unchanged.
module flash_attn_soc_m6b3 #(
    parameter bit COMPACT_DECODE_OUTPUT=1'b1,
    parameter int DEQUANT_LANES=256,
    parameter int HEAD_DIM=16
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
);
    import m5_types_pkg::*;
    config_t candidate,child_config;
    logic command_attempt,encoding_bad,command_ok,config_writable;
    logic [2:0] action,phase;
    logic [31:0] reject_reason,output_bytes,job_id,error_code,write_error_addr;
    logic compute_done,output_committed,success_event,fatal_event,reject_event;
    logic [12:0][31:0] snapshots;
    logic child_start,child_ack,pack_start;
    logic child_ready,child_cfg_error,child_done_status,child_output_valid,child_error,child_rejected;
    logic [15:0] child_words_per_head,pack_words;
    logic [6:0][31:0] child_perf;
    logic [31:0] q_base[4],k_base[2],v_base[2];
    logic signed [15:0] scale_q[4],scale_k[2],scale_v[2];
    logic [1:0] cache_valid;
    logic [6:0] cache_length[2];
    logic [31:0] cache_epoch[2];
    logic [1:0][6:0] csr_cache_length;
    logic [1:0][31:0] csr_cache_epoch;
    logic [1:0] out_head;
    logic [$clog2(16*HEAD_DIM)-1:0] out_word;
    logic signed [31:0] out_data;
    logic pack_done,pack_error,pack_rejected,src_valid,src_ready;
    logic [63:0] src_data;
    logic [31:0] source_words;
    logic descriptor_pending,wb_desc_valid,wb_desc_ready,wb_done,wb_outstanding,wb_fatal;
    logic [1:0] wb_error_kind;
    logic [31:0] wb_error_addr,wb_length;
    logic [3:0][31:0] wb_perf;

    for(genvar h=0;h<4;h++)begin : g_q_config
        assign q_base[h]=child_config.q_base[h];
        assign scale_q[h]=$signed(child_config.scale_q[h]);
    end
    for(genvar h=0;h<2;h++)begin : g_kv_config
        assign k_base[h]=child_config.k_base[h];assign v_base[h]=child_config.v_base[h];
        assign scale_k[h]=$signed(child_config.scale_k[h]);assign scale_v[h]=$signed(child_config.scale_v[h]);
        assign csr_cache_length[h]=cache_length[h];assign csr_cache_epoch[h]=cache_epoch[h];
    end
    assign pack_words=child_config.op==0 ? 16'(16*HEAD_DIM) : 16'(HEAD_DIM);
    assign wb_length=child_config.op==0 ? 32'(4*16*HEAD_DIM*4) : 32'(4*HEAD_DIM*4);
    // Active config remains immutable throughout WRITEBACK. Pulse -> held VALID.
    assign wb_desc_valid=phase==WRITEBACK && (pack_start || descriptor_pending);
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)descriptor_pending<=0;
        else if(phase==ERROR || (wb_desc_valid && wb_desc_ready))descriptor_pending<=0;
        else if(pack_start)descriptor_pending<=1;
    end
    axi_lite_csr_m6b3 #(.COMPACT_DECODE_OUTPUT(COMPACT_DECODE_OUTPUT),.DEQUANT_LANES(DEQUANT_LANES),.HEAD_DIM(HEAD_DIM)) u_csr(
        .clk(clk),.rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr),.s_axil_araddr(s_axil_araddr),
        .s_axil_awprot(s_axil_awprot),.s_axil_arprot(s_axil_arprot),
        .s_axil_awvalid(s_axil_awvalid),.s_axil_wvalid(s_axil_wvalid),.s_axil_bready(s_axil_bready),
        .s_axil_arvalid(s_axil_arvalid),.s_axil_rready(s_axil_rready),
        .s_axil_awready(s_axil_awready),.s_axil_wready(s_axil_wready),.s_axil_bvalid(s_axil_bvalid),
        .s_axil_arready(s_axil_arready),.s_axil_rvalid(s_axil_rvalid),.s_axil_wdata(s_axil_wdata),
        .s_axil_wstrb(s_axil_wstrb),.s_axil_bresp(s_axil_bresp),.s_axil_rresp(s_axil_rresp),.s_axil_rdata(s_axil_rdata),
        .irq(irq),.candidate(candidate),.command_attempt(command_attempt),.encoding_bad(encoding_bad),.action(action),
        .command_ok(command_ok),.config_writable(config_writable),.reject_reason(reject_reason),.phase(phase),
        .compute_done(compute_done),.output_committed(output_committed),.output_bytes(output_bytes),
        .job_id(job_id),.error_code(error_code),.write_error_addr(write_error_addr),
        .success_event(success_event),.fatal_event(fatal_event),.reject_event(reject_event),.snapshots(snapshots),
        .cache_valid(cache_valid),.cache_length(csr_cache_length),.cache_epoch(csr_cache_epoch));
    soc_job_adapter_m6b3 #(.HEAD_DIM(HEAD_DIM)) u_job(
        .clk(clk),.rst_n(rst_n),.command_attempt(command_attempt),.encoding_bad(encoding_bad),.action(action),
        .candidate(candidate),.command_ok(command_ok),.reject_reason(reject_reason),.config_writable(config_writable),
        .child_config(child_config),.child_start(child_start),.child_ack(child_ack),.pack_start(pack_start),
        .child_ready(child_ready),.child_cfg_error(child_cfg_error),.child_done_status(child_done_status),
        .child_output_valid(child_output_valid),.child_error(child_error),.child_rejected(child_rejected || pack_rejected),
        .child_words_per_head(child_words_per_head),.child_perf(child_perf),.source_retired(pack_done),
        .wb_final_okay(wb_done),.wb_outstanding(wb_outstanding),.wb_error(wb_fatal && wb_error_kind==2),
        .wb_descriptor_error((wb_fatal && wb_error_kind==1) || pack_error),.source_words(source_words),
        .wb_error_addr(wb_error_addr),.wb_perf(wb_perf),.phase(phase),.compute_done(compute_done),
        .output_committed(output_committed),.output_bytes(output_bytes),.job_id(job_id),.error_code(error_code),
        .write_error_addr(write_error_addr),.success_event(success_event),.fatal_event(fatal_event),
        .reject_event(reject_event),.snapshots(snapshots));
    /* verilator lint_off PINCONNECTEMPTY */
    flash_attn_serial_m6b3 #(.Q_HEADS(4),.KV_HEADS(2),.COMPACT_DECODE_OUTPUT(COMPACT_DECODE_OUTPUT),.DEQUANT_LANES(DEQUANT_LANES),.HEAD_DIM(HEAD_DIM)) u_engine(
        .clk(clk),.rst_n(rst_n),.start(child_start),.ack(child_ack),.cfg_op(child_config.op),
        .cfg_query_count(child_config.query_count),.cfg_query_position(child_config.query_position),
        .cfg_kv_length(child_config.kv_length),.cfg_replace_cache(child_config.replace_cache),.causal(child_config.causal),
        .cfg_q_base(q_base),.cfg_k_base(k_base),.cfg_v_base(v_base),.scale_q(scale_q),.scale_k(scale_k),.scale_v(scale_v),
        .ready(child_ready),.busy(),.done(),.done_status(child_done_status),.output_valid(child_output_valid),
        .error_status(child_error),.cfg_error(child_cfg_error),.start_rejected(child_rejected),
        .output_words_per_head(child_words_per_head),.out_head(out_head),.out_word(out_word),.out_data(out_data),
        .cache_valid(cache_valid),.cache_length(cache_length),.cache_epoch(cache_epoch),
        .m_araddr(m_araddr),.m_arlen(m_arlen),.m_arsize(m_arsize),.m_arburst(m_arburst),.m_arvalid(m_arvalid),
        .m_arready(m_arready),.m_rdata(m_rdata),.m_rresp(m_rresp),.m_rlast(m_rlast),.m_rvalid(m_rvalid),.m_rready(m_rready),
        .perf_cycles(child_perf[0]),.perf_dma_bytes(child_perf[1]),.perf_ar(child_perf[2]),.perf_rbeats(child_perf[3]),
        .perf_cache_reads(child_perf[4]),.perf_cache_writes(child_perf[5]),.perf_copy_cycles(child_perf[6]));
    gqa_output_packer_m6b3 #(.HEAD_DIM(HEAD_DIM)) u_pack(
        .clk(clk),.rst_n(rst_n),.start(pack_start),.ready(),.busy(),.done(pack_done),.start_rejected(pack_rejected),
        .error(pack_error),.words_per_head(pack_words),.result_words_per_head(child_words_per_head),
        .result_valid(child_output_valid && child_done_status),.out_head(out_head),.out_word(out_word),.out_data(out_data),
        .src_valid(src_valid),.src_ready(src_ready),.src_data(src_data),.retired_words(source_words));
    dma_write_engine_m5 u_write(
        .clk(clk),.rst_n(rst_n),.desc_valid(wb_desc_valid),.desc_ready(wb_desc_ready),
        .desc_addr(child_config.output_base),.desc_len_bytes(wb_length),
        .src_valid(src_valid),.src_ready(src_ready),.src_data(src_data),.done(wb_done),.busy(),
        .outstanding(wb_outstanding),.error(wb_fatal),.error_kind(wb_error_kind),.failing_burst_addr(wb_error_addr),
        .m_awaddr(m_awaddr),.m_awlen(m_awlen),.m_awsize(m_awsize),.m_awburst(m_awburst),.m_awvalid(m_awvalid),
        .m_awready(m_awready),.m_wdata(m_wdata),.m_wstrb(m_wstrb),.m_wlast(m_wlast),.m_wvalid(m_wvalid),
        .m_wready(m_wready),.m_bresp(m_bresp),.m_bvalid(m_bvalid),.m_bready(m_bready),
        .perf_aw(wb_perf[1]),.perf_wbeats(),.perf_wbytes(wb_perf[0]),.perf_b(wb_perf[2]),.perf_okay_b(wb_perf[3]),
        .perf_cycles(),.perf_aw_stall(),.perf_w_stall(),.perf_b_wait(),.perf_src_wait());
    /* verilator lint_on PINCONNECTEMPTY */
endmodule
