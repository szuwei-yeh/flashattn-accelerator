`timescale 1ns/1ps
module dma_write_engine_m5 (
    input logic clk,rst_n,
    input logic desc_valid,
    output logic desc_ready,
    input logic [31:0] desc_addr,desc_len_bytes,
    input logic src_valid,
    output logic src_ready,
    input logic [63:0] src_data,
    output logic done,busy,outstanding,error,
    output logic [1:0] error_kind,
    output logic [31:0] failing_burst_addr,
    output logic [31:0] m_awaddr,
    output logic [7:0] m_awlen,
    output logic [2:0] m_awsize,
    output logic [1:0] m_awburst,
    output logic m_awvalid,
    input logic m_awready,
    output logic [63:0] m_wdata,
    output logic [7:0] m_wstrb,
    output logic m_wlast,m_wvalid,
    input logic m_wready,
    input logic [1:0] m_bresp,
    input logic m_bvalid,
    output logic m_bready,
    output logic [31:0] perf_aw,perf_wbeats,perf_wbytes,perf_b,perf_okay_b,perf_cycles,
    output logic [31:0] perf_aw_stall,perf_w_stall,perf_b_wait,perf_src_wait
);
    typedef enum logic [1:0] {IDLE,TRANSFER,FATAL} state_t;
    state_t state;
    logic aw_accepted;
    logic [32:0] address,descriptor_end,next_address;
    logic [31:0] remaining;
    logic [4:0] beats,sent;
    logic descriptor_bad,accept_descriptor,accept_aw,accept_w,accept_b;
    function automatic logic [4:0] burst_count(input logic [11:0] page_offset,input logic [31:0] left);
        logic [31:0] page_beats,result;
        page_beats=(32'd4096-{20'd0,page_offset})>>3;
        result=left<32'd16 ? left : 32'd16;
        if(page_beats<result)result=page_beats;
        return result[4:0];
    endfunction
    assign descriptor_end={1'b0,desc_addr}+{1'b0,desc_len_bytes};
    assign next_address=address+({28'd0,beats}<<3);
    assign descriptor_bad=desc_len_bytes==0 || desc_len_bytes[2:0]!=0 || desc_addr[2:0]!=0 || descriptor_end>33'h100000000;
    assign desc_ready=state==IDLE;
    assign accept_descriptor=desc_valid && desc_ready;
    assign busy=state==TRANSFER;
    assign outstanding=state==TRANSFER; // descriptor owner incl. source/response waits
    assign error=state==FATAL;
    assign m_awaddr=address[31:0];
    assign m_awlen={3'd0,beats}-8'd1;
    assign m_awsize=3;assign m_awburst=1;
    assign m_awvalid=busy && !aw_accepted;
    assign m_wvalid=busy && sent<beats && src_valid;
    assign m_wdata=src_data;
    assign m_wstrb=8'hff;
    assign m_wlast=sent==beats-5'd1;
    assign src_ready=busy && sent<beats && m_wready;
    assign m_bready=busy && aw_accepted && sent==beats;
    assign accept_aw=m_awvalid && m_awready;
    assign accept_w=m_wvalid && m_wready;
    assign accept_b=m_bvalid && m_bready;
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;remaining<=0;beats<=0;sent<=0;aw_accepted<=0;
            done<=0;error_kind<=0;failing_burst_addr<=0;
            perf_aw<=0;perf_wbeats<=0;perf_wbytes<=0;perf_b<=0;perf_okay_b<=0;perf_cycles<=0;
            perf_aw_stall<=0;perf_w_stall<=0;perf_b_wait<=0;perf_src_wait<=0;
        end else begin
            done<=0;
            if(accept_descriptor)begin
                perf_aw<=0;perf_wbeats<=0;perf_wbytes<=0;perf_b<=0;perf_okay_b<=0;perf_cycles<=0;
                perf_aw_stall<=0;perf_w_stall<=0;perf_b_wait<=0;perf_src_wait<=0;
                error_kind<=0;failing_burst_addr<=0;
                if(descriptor_bad)begin state<=FATAL;error_kind<=1;end
                else begin
                    state<=TRANSFER;address<={1'b0,desc_addr};remaining<=desc_len_bytes>>3;
                    beats<=burst_count(desc_addr[11:0],desc_len_bytes>>3);sent<=0;aw_accepted<=0;
                end
            end else if(busy)begin
                perf_cycles<=perf_cycles+1'b1;
                if(m_awvalid && !m_awready)perf_aw_stall<=perf_aw_stall+1'b1;
                if(m_wvalid && !m_wready)perf_w_stall<=perf_w_stall+1'b1;
                if(m_bready && !m_bvalid)perf_b_wait<=perf_b_wait+1'b1;
                if(sent<beats && !src_valid)perf_src_wait<=perf_src_wait+1'b1;
                if(accept_aw)begin aw_accepted<=1;perf_aw<=perf_aw+1'b1;end
                if(accept_w)begin sent<=sent+1'b1;perf_wbeats<=perf_wbeats+1'b1;perf_wbytes<=perf_wbytes+32'd8;end
                if(accept_b)begin
                    perf_b<=perf_b+1'b1;
                    if(m_bresp!=0)begin state<=FATAL;error_kind<=2;failing_burst_addr<=address[31:0];end
                    else begin
                        perf_okay_b<=perf_okay_b+1'b1;
                        if(remaining=={27'd0,beats})begin state<=IDLE;done<=1;end
                        else begin
                            address<=next_address;
                            remaining<=remaining-{27'd0,beats};
                            beats<=burst_count(next_address[11:0],remaining-{27'd0,beats});
                            sent<=0;aw_accepted<=0;
                        end
                    end
                end
            end
        end
    end
endmodule
