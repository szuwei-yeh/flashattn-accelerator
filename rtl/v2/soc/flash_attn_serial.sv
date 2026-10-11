// Isolated M4 serial heads: ONE unchanged M3B compute engine; physical KV heads.
`timescale 1ns/1ps
module flash_attn_serial_m6b3 #(
    parameter int Q_HEADS=2, KV_HEADS=2, HEAD_DIM=16,
    parameter bit COMPACT_DECODE_OUTPUT=1'b1,
    parameter int DEQUANT_LANES=256
)(
    input logic clk,rst_n,start,ack,
    input logic [1:0] cfg_op,
    input logic [6:0] cfg_query_count,cfg_query_position,cfg_kv_length,
    input logic cfg_replace_cache,causal,
    input logic [31:0] cfg_q_base[Q_HEADS],cfg_k_base[KV_HEADS],cfg_v_base[KV_HEADS],
    input logic signed [15:0] scale_q[Q_HEADS],scale_k[KV_HEADS],scale_v[KV_HEADS],
    output logic ready,busy,done,done_status,output_valid,error_status,cfg_error,start_rejected,
    output logic [15:0] output_words_per_head,
    input logic [1:0] out_head,
    input logic [$clog2(16*HEAD_DIM)-1:0] out_word,
    output logic signed [31:0] out_data,
    output logic [KV_HEADS-1:0] cache_valid,
    output logic [6:0] cache_length[KV_HEADS],
    output logic [31:0] cache_epoch[KV_HEADS],
    output logic [31:0] m_araddr,
    output logic [7:0] m_arlen,
    output logic [2:0] m_arsize,
    output logic [1:0] m_arburst,
    output logic m_arvalid,
    input logic m_arready,
    input logic [63:0] m_rdata,
    input logic [1:0] m_rresp,
    input logic m_rlast,m_rvalid,
    output logic m_rready,
    output logic [31:0] perf_cycles,perf_dma_bytes,perf_ar,perf_rbeats,
    output logic [31:0] perf_cache_reads,perf_cache_writes,perf_copy_cycles
);
    localparam logic [1:0] PREFILL=0,APPEND=1,DECODE=2,CLEAR=3;
    localparam int GROUP=Q_HEADS/KV_HEADS;
    localparam int HEAD_WORDS=16*HEAD_DIM,COUNT_W=$clog2(HEAD_WORDS)+1;
    typedef enum logic [1:0] {IDLE,BUSY,DONE,ERROR} job_t;
    typedef enum logic [3:0] {S_IDLE,K_ISSUE,K_WAIT,V_ISSUE,V_WAIT,
        Q_ISSUE,Q_WAIT,CORE_START,CORE_WAIT,COPY_ADDR,COPY_WAIT,COPY_STORE,
        REARM,ADVANCE,PUBLISH} sched_t;
    job_t job;
    sched_t state;
    logic previous_start,request,accepted,completion,release_job;
    logic [1:0] r_op;
    logic [6:0] r_length,r_position;
    logic r_causal;
    logic [31:0] r_qbase[Q_HEADS],r_kbase[KV_HEADS],r_vbase[KV_HEADS];
    logic signed [15:0] r_sq[Q_HEADS],r_sk[KV_HEADS],r_sv[KV_HEADS];
    logic [1:0] q_head,load_head,selected_kv;
    logic [COUNT_W-1:0] copy_word,words;
    logic signed [31:0] results[Q_HEADS*HEAD_WORDS];
    logic [KV_HEADS-1:0] pair_complete,write_accept,read_accept,read_valid;
    logic signed [15:0] stored_sk[KV_HEADS],stored_sv[KV_HEADS];
    logic [127:0] k_data[KV_HEADS],v_data[KV_HEADS];
    logic desc_valid,desc_ready,dma_done,dma_error,w_en,w_vec;
    logic [31:0] desc_addr,desc_bytes;
    logic [1:0] desc_dst,w_dst;
    logic [11:0] desc_waddr,w_addr,kv_r_addr;
    logic [127:0] w_data;
    logic kv_r_en,core_start,core_done,core_rearm;
    logic loader_idle,array_idle,pipe_empty,core_active;
    logic signed [31:0] core_out;
    logic [32:0] span_end;
    initial if(!(((Q_HEADS==2 && KV_HEADS==2) || (Q_HEADS==4 && (KV_HEADS==2 || KV_HEADS==4))) && (HEAD_DIM==16 || HEAD_DIM==64) && (DEQUANT_LANES==256 || DEQUANT_LANES==16)))
        $fatal(1,"M6B3 supports 2:2,4:4,4:2 d16/d64 and 256/16 lanes only");
    assign request=start && !previous_start;
    assign ready=job==IDLE && desc_ready && !dma_error;
    assign accepted=ready && request && !cfg_error;
    assign busy=job==BUSY && !dma_error;
    assign done_status=job==DONE;
    assign error_status=job==ERROR || dma_error;
    assign release_job=done_status && ack && !dma_error;
    assign output_valid=done_status && (r_op==PREFILL || r_op==DECODE) && !error_status;
    assign words=r_op==PREFILL ? COUNT_W'(HEAD_WORDS) : COUNT_W'(HEAD_DIM);
    assign output_words_per_head=output_valid ? 16'(words) : 16'd0;
    assign out_data=output_valid && int'(out_head)<Q_HEADS && {1'b0,out_word}<words ?
        results[int'(out_head)*HEAD_WORDS+int'(out_word)] : 32'sd0;
    assign selected_kv=2'(int'(q_head)/GROUP);
    assign core_start=busy && state==CORE_START;
    assign core_rearm=release_job || (busy && state==REARM);
    assign completion=busy && state==PUBLISH && desc_ready && loader_idle && array_idle && pipe_empty &&
        ((r_op==PREFILL || r_op==APPEND) ? (&pair_complete) : 1'b1);
    always_comb begin
        cfg_error=cfg_replace_cache && cfg_op!=PREFILL;
        span_end=0;
        case(cfg_op)
            PREFILL:cfg_error=!cfg_replace_cache || cfg_query_count!=16 || cfg_query_position!=0 || cfg_kv_length!=16;
            APPEND:cfg_error=cfg_error || cfg_query_count!=0 || cfg_query_position!=0 || cfg_kv_length>=64;
            DECODE:cfg_error=cfg_error || cfg_query_count!=1 || cfg_query_position!=cfg_kv_length-7'd1 ||
                !(cfg_kv_length==16 || cfg_kv_length==17 || cfg_kv_length==18);
            CLEAR:cfg_error=cfg_error || cfg_query_count!=0 || cfg_query_position!=0 || cfg_kv_length!=0;
            default:cfg_error=1;
        endcase
        for(int h=0;h<Q_HEADS;h++)begin
            span_end={1'b0,cfg_q_base[h]}+(cfg_op==PREFILL ? 33'(HEAD_WORDS) : 33'(HEAD_DIM));
            if((cfg_op==DECODE || cfg_op==PREFILL) && (cfg_q_base[h][3:0]!=0 || span_end>33'h100000000))cfg_error=1;
        end
        for(int h=0;h<KV_HEADS;h++)begin
            if(cfg_op==APPEND || cfg_op==DECODE)begin
                if(cache_length[h]!=cfg_kv_length || (cfg_kv_length!=0 && (!cache_valid[h] || scale_k[h]!=stored_sk[h] || scale_v[h]!=stored_sv[h])))cfg_error=1;
            end
            if(cfg_op==APPEND || cfg_op==PREFILL)begin
                span_end={1'b0,cfg_k_base[h]}+(cfg_op==PREFILL ? 33'(HEAD_WORDS) : 33'(HEAD_DIM));
                if(cfg_k_base[h][3:0]!=0 || span_end>33'h100000000)cfg_error=1;
                span_end={1'b0,cfg_v_base[h]}+(cfg_op==PREFILL ? 33'(HEAD_WORDS) : 33'(HEAD_DIM));
                if(cfg_v_base[h][3:0]!=0 || span_end>33'h100000000)cfg_error=1;
            end
        end
    end
    always_comb begin
        desc_valid=busy && (state==K_ISSUE || state==V_ISSUE || state==Q_ISSUE);
        desc_addr=r_qbase[int'(q_head)];desc_dst=0;desc_waddr=0;
        desc_bytes=r_op==PREFILL ? 32'(HEAD_WORDS) : 32'(HEAD_DIM);
        if(state==K_ISSUE || state==V_ISSUE)begin
            desc_addr=state==K_ISSUE ? r_kbase[int'(load_head)] : r_vbase[int'(load_head)];
            desc_dst=state==K_ISSUE ? 2'd1 : 2'd2;
            desc_waddr=r_op==PREFILL ? 12'd0 : 12'(int'(r_length)*HEAD_DIM);
        end
    end
    dma_engine_vec u_dma(
        .clk(clk),.rst_n(rst_n),.desc_valid(desc_valid),.desc_ready(desc_ready),
        .desc_addr(desc_addr),.desc_dst_addr(desc_waddr),.desc_len_bytes(desc_bytes),.desc_dst(desc_dst),
        .done(dma_done),.error(dma_error),.m_araddr(m_araddr),.m_arlen(m_arlen),
        .m_arsize(m_arsize),.m_arburst(m_arburst),.m_arvalid(m_arvalid),.m_arready(m_arready),
        .m_rdata(m_rdata),.m_rresp(m_rresp),.m_rlast(m_rlast),.m_rvalid(m_rvalid),.m_rready(m_rready),
        .w_en(w_en),.w_vec(w_vec),.w_dst(w_dst),.w_addr(w_addr),.w_vdata(w_data));
    /* verilator lint_off PINCONNECTEMPTY */
    for(genvar h=0;h<KV_HEADS;h++)begin : g_cache
        kv_cache_banked_decode_m2c3 #(.HEAD_DIM(HEAD_DIM)) u_cache(
            .clk(clk),.rst_n(rst_n),.cmd_begin(accepted),.cmd_commit(completion),.cmd_fault(dma_error),
            .cmd_op(cfg_op),.cmd_length(cfg_kv_length),.cmd_scale_k(scale_k[h]),.cmd_scale_v(scale_v[h]),
            .w_en(w_en && w_dst!=0 && load_head==2'(h)),.w_sel(w_dst==2),.w_addr(w_addr),.w_data(w_data),
            .core_r_en(kv_r_en && selected_kv==2'(h)),.core_r_addr(kv_r_addr),
            .core_k_data(k_data[h]),.core_v_data(v_data[h]),.core_r_valid(read_valid[h]),
            .dbg_enable(1'b0),.dbg_req_valid(1'b0),.dbg_req_ready(),.dbg_resp_valid(),.dbg_error(),
            .dbg_addr(12'd0),.dbg_k_data(),.dbg_v_data(),.cache_valid(cache_valid[h]),.cache_length(cache_length[h]),
            .cache_scale_k(stored_sk[h]),.cache_scale_v(stored_sv[h]),.cache_epoch(cache_epoch[h]),
            .pair_complete(pair_complete[h]),.write_accept(write_accept[h]),.read_accept(read_accept[h]));
    end
    /* verilator lint_on PINCONNECTEMPTY */
    flash_attn_core_decode_m3b #(.HEAD_DIM(HEAD_DIM),.DEQUANT_LANES(DEQUANT_LANES),.COMPACT_DECODE_OUTPUT(COMPACT_DECODE_OUTPUT)) u_core(
        .clk(clk),.rst_n(rst_n),.start(core_start),.rearm(core_rearm),.done(core_done),
        .scale_q(r_sq[int'(q_head)]),.scale_k(r_sk[int'(selected_kv)]),.scale_v(r_sv[int'(selected_kv)]),
        .causal(r_causal),.prefill(r_op==PREFILL),.query_position(r_position),.valid_kv_length(r_length),
        .dma_q_we(w_en && w_dst==0),.dma_q_addr(w_addr),.dma_q_data(w_data),
        .kv_r_en(kv_r_en),.kv_r_addr(kv_r_addr),.kv_k_rdata(k_data[int'(selected_kv)]),.kv_v_rdata(v_data[int'(selected_kv)]),
        .kv_r_valid(read_valid[int'(selected_kv)]),.out_raddr(12'(copy_word)),.out_rdata(core_out),
        .loader_idle(loader_idle),.array_idle(array_idle),.output_pipe_empty(pipe_empty),.core_active(core_active));
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            job<=IDLE;state<=S_IDLE;previous_start<=0;done<=0;start_rejected<=0;
            r_op<=0;r_length<=0;r_position<=0;r_causal<=0;q_head<=0;load_head<=0;copy_word<=0;
            for(int h=0;h<Q_HEADS;h++)begin r_qbase[h]<=0;r_sq[h]<=0;end
            for(int h=0;h<KV_HEADS;h++)begin r_kbase[h]<=0;r_vbase[h]<=0;r_sk[h]<=0;r_sv[h]<=0;end
        end else begin
            previous_start<=start;done<=0;start_rejected<=request && !accepted;
            if(dma_error)begin job<=ERROR;state<=S_IDLE;end
            else if(accepted)begin
                job<=BUSY;r_op<=cfg_op;r_length<=cfg_kv_length;r_position<=cfg_query_position;r_causal<=causal;
                q_head<=0;load_head<=0;copy_word<=0;
                for(int h=0;h<Q_HEADS;h++)begin r_qbase[h]<=cfg_q_base[h];r_sq[h]<=scale_q[h];end
                for(int h=0;h<KV_HEADS;h++)begin r_kbase[h]<=cfg_k_base[h];r_vbase[h]<=cfg_v_base[h];r_sk[h]<=scale_k[h];r_sv[h]<=scale_v[h];end
                case(cfg_op)PREFILL,APPEND:state<=K_ISSUE;DECODE:state<=Q_ISSUE;default:state<=PUBLISH;endcase
            end else if(completion)begin job<=DONE;state<=S_IDLE;done<=1;end
            else if(release_job)begin job<=IDLE;state<=S_IDLE;end
            else if(busy)case(state)
                K_ISSUE:if(desc_ready)state<=K_WAIT;
                K_WAIT:if(dma_done)state<=V_ISSUE;
                V_ISSUE:if(desc_ready)state<=V_WAIT;
                V_WAIT:if(dma_done)begin
                    if(int'(load_head)+1<KV_HEADS)begin load_head<=load_head+1'b1;state<=K_ISSUE;end
                    else state<=r_op==PREFILL ? Q_ISSUE : PUBLISH;
                end
                Q_ISSUE:if(desc_ready)state<=Q_WAIT;
                Q_WAIT:if(dma_done)state<=CORE_START;
                CORE_START:state<=CORE_WAIT;
                CORE_WAIT:if(core_done && loader_idle && array_idle && pipe_empty)begin copy_word<=0;state<=COPY_ADDR;end
                COPY_ADDR:state<=COPY_WAIT;
                COPY_WAIT:state<=COPY_STORE;
                COPY_STORE:begin
                    results[int'(q_head)*HEAD_WORDS+int'(copy_word)]<=core_out;
                    if(copy_word+COUNT_W'(1)==words)state<=REARM;
                    else begin copy_word<=copy_word+1'b1;state<=COPY_ADDR;end
                end
                REARM:state<=ADVANCE;
                ADVANCE:begin
                    if(int'(q_head)+1<Q_HEADS)begin q_head<=q_head+1'b1;state<=Q_ISSUE;end
                    else state<=PUBLISH;
                end
                default:;
            endcase
        end
    end
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            perf_cycles<=0;perf_dma_bytes<=0;perf_ar<=0;perf_rbeats<=0;
            perf_cache_reads<=0;perf_cache_writes<=0;perf_copy_cycles<=0;
        end else if(accepted)begin
            perf_cycles<=0;perf_dma_bytes<=0;perf_ar<=0;perf_rbeats<=0;
            perf_cache_reads<=0;perf_cache_writes<=0;perf_copy_cycles<=0;
        end else if(busy)begin
            perf_cycles<=perf_cycles+1;
            if(w_en)perf_dma_bytes<=perf_dma_bytes+16;
            if(m_arvalid && m_arready)perf_ar<=perf_ar+1;
            if(m_rvalid && m_rready)perf_rbeats<=perf_rbeats+1;
            if(|read_accept)perf_cache_reads<=perf_cache_reads+1;
            if(|write_accept)perf_cache_writes<=perf_cache_writes+1;
            if(state==COPY_ADDR || state==COPY_WAIT || state==COPY_STORE)perf_copy_cycles<=perf_copy_cycles+1;
        end
    end
    // synthesis translate_off
    always @(posedge clk) if(rst_n && w_en && !w_vec) $fatal(1,"M4 non-vector DMA write");
    // synthesis translate_on
endmodule
