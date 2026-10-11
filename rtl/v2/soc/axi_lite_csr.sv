`timescale 1ns/1ps
module axi_lite_csr_m6b3 #(
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
    output m5_types_pkg::config_t candidate,
    output logic command_attempt,encoding_bad,
    output logic [2:0] action,
    input logic command_ok,config_writable,
    input logic [31:0] reject_reason,
    input logic [2:0] phase,
    input logic compute_done,output_committed,
    input logic [31:0] output_bytes,job_id,error_code,write_error_addr,
    input logic success_event,fatal_event,reject_event,
    input logic [12:0][31:0] snapshots,
    input logic [1:0] cache_valid,
    input logic [1:0][6:0] cache_length,
    input logic [1:0][31:0] cache_epoch
);
    import m5_types_pkg::*;
    logic aw_full,w_full,execute;
    logic [11:0] aw_addr;
    logic [2:0] aw_prot;
    logic [31:0] w_data,byte_mask,effective,defined_mask,merged,read_value;
    logic [3:0] w_strb;
    logic [31:0] shadow[32];
    logic [2:0] irq_enable,irq_pending,irq_clear,new_events;
    logic [31:0] last_reject,command_rejects,status;
    logic valid_address,writable,idle_only,write_bad;
    logic [4:0] word_index;
    // PROT is deliberately unused: this block is not a security boundary.
    logic unused_prot;
    assign unused_prot=^{aw_prot,s_axil_arprot};
    assign s_axil_awready=!aw_full && !s_axil_bvalid;
    assign s_axil_wready=!w_full && !s_axil_bvalid;
    assign s_axil_arready=!s_axil_rvalid;
    assign execute=aw_full && w_full && !s_axil_bvalid;
    assign byte_mask={{8{w_strb[3]}},{8{w_strb[2]}},{8{w_strb[1]}},{8{w_strb[0]}}};
    assign effective=w_data & byte_mask;
    assign word_index=aw_addr[6:2];
    assign action=effective[2:0];
    assign command_attempt=execute && aw_addr==12'h008 && action!=0;
    assign encoding_bad=(effective & 32'hfffffff8)!=0 || !(action==1 || action==2 || action==4);
    assign new_events={reject_event,fatal_event,success_event};
    assign irq=|(irq_pending & irq_enable);
    always_comb begin
        candidate='0;
        candidate.op=shadow[9][1:0];candidate.causal=shadow[9][2];candidate.replace_cache=shadow[9][3];
        if(command_attempt && action==4)candidate.op=3;
        candidate.query_count=shadow[10][6:0];candidate.query_position=shadow[10][14:8];candidate.kv_length=shadow[10][22:16];
        candidate.output_base=shadow[11];
        for(int i=0;i<4;i++)begin candidate.q_base[i]=shadow[16+i];candidate.scale_q[i]=shadow[20+i][15:0];end
        for(int i=0;i<2;i++)begin
            candidate.k_base[i]=shadow[24+i];candidate.v_base[i]=shadow[26+i];
            candidate.scale_k[i]=shadow[28+i][15:0];candidate.scale_v[i]=shadow[30+i][15:0];
        end
        status=0;status[0]=phase==IDLE;status[1]=phase==DISPATCH || phase==RUN || phase==WRITEBACK || phase==RELEASE;
        status[2]=phase==DONE;status[3]=phase==ERROR;status[4]=compute_done;status[5]=output_committed;
        status[6]=irq_pending[2];status[7]=(&cache_valid) && phase!=ERROR;status[10:8]=phase;
        valid_address=aw_addr[1:0]==0 && aw_addr<=12'h0cc;
        writable=0;idle_only=0;defined_mask=32'hffffffff;
        case(aw_addr)
            12'h008,12'h010,12'h018:begin writable=1;defined_mask=7;end
            12'h024:begin writable=1;idle_only=1;defined_mask=15;end
            12'h028:begin writable=1;idle_only=1;defined_mask=32'h007f7f7f;end
            12'h02c:begin writable=1;idle_only=1;end
            default:if(aw_addr>=12'h040 && aw_addr<=12'h07c)begin
                writable=1;idle_only=1;
                if((aw_addr>=12'h050 && aw_addr<=12'h05c) || aw_addr>=12'h070)defined_mask=32'h0000ffff;
            end
        endcase
        merged=(shadow[word_index] & ~byte_mask) | effective;
        if(aw_addr==12'h010)merged=({29'd0,irq_enable} & ~byte_mask) | effective;
        write_bad=!valid_address || !writable || ((effective & ~defined_mask)!=0);
        if(w_strb!=0 && idle_only && !config_writable)write_bad=1;
        if(command_attempt && !command_ok)write_bad=1;
        irq_clear=0;
        if(execute && !write_bad && aw_addr==12'h018)irq_clear=effective[2:0];
        // Full byte offset decode. No masking to create undocumented aliases.
        read_value=0;
        case(s_axil_araddr)
            12'h000:read_value=32'h46415432;
            12'h004:read_value=32'h00010000;
            12'h00c:read_value=status;
            12'h010:read_value={29'd0,irq_enable};
            12'h014:read_value={29'd0,irq_pending};
            12'h01c:read_value=error_code;
            12'h020:read_value=last_reject;
            12'h024,12'h028,12'h02c:read_value=shadow[s_axil_araddr[6:2]];
            12'h030:read_value=32'h10020400 | 32'(HEAD_DIM);
            12'h034:read_value=(COMPACT_DECODE_OUTPUT ? 32'h07000040 : 32'h06000040) | (32'(DEQUANT_LANES) << 8);
            12'h038:read_value=32'h50;
            12'h03c:read_value=job_id;
            12'h0b4:read_value=command_rejects;
            12'h0b8,12'h0bc:if(phase!=ERROR)read_value={23'd0,cache_valid[s_axil_araddr[2]],1'b0,cache_length[s_axil_araddr[2]]};
            12'h0c0,12'h0c4:read_value=cache_epoch[s_axil_araddr[2]];
            12'h0c8:read_value=output_bytes;
            12'h0cc:read_value=write_error_addr;
            default:begin
                if(s_axil_araddr>=12'h040 && s_axil_araddr<=12'h07c && s_axil_araddr[1:0]==0)read_value=shadow[s_axil_araddr[6:2]];
                if(s_axil_araddr>=12'h080 && s_axil_araddr<=12'h0b0 && s_axil_araddr[1:0]==0)read_value=snapshots[(int'(s_axil_araddr)-128)/4];
            end
        endcase
    end
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            aw_full<=0;w_full<=0;aw_addr<=0;aw_prot<=0;w_data<=0;w_strb<=0;
            s_axil_bvalid<=0;s_axil_bresp<=0;s_axil_rvalid<=0;s_axil_rdata<=0;s_axil_rresp<=0;
            irq_enable<=0;irq_pending<=0;last_reject<=0;command_rejects<=0;
            for(int i=0;i<32;i++)shadow[i]<=0;
        end else begin
            irq_pending<=(irq_pending & ~irq_clear) | new_events;
            if(reject_event)begin last_reject<=reject_reason;command_rejects<=command_rejects+1'b1;end
            if(s_axil_awvalid && s_axil_awready)begin aw_full<=1;aw_addr<=s_axil_awaddr;aw_prot<=s_axil_awprot;end
            if(s_axil_wvalid && s_axil_wready)begin w_full<=1;w_data<=s_axil_wdata;w_strb<=s_axil_wstrb;end
            if(s_axil_bvalid && s_axil_bready)s_axil_bvalid<=0;
            if(execute)begin
                aw_full<=0;w_full<=0;s_axil_bvalid<=1;s_axil_bresp<=write_bad ? 2'b10 : 2'b00;
                if(!write_bad && w_strb!=0)begin
                    if(idle_only)shadow[word_index]<=merged;
                    if(aw_addr==12'h010)irq_enable<=merged[2:0];
                end
            end
            if(s_axil_rvalid && s_axil_rready)s_axil_rvalid<=0;
            if(s_axil_arvalid && s_axil_arready)begin
                s_axil_rvalid<=1;s_axil_rresp<=(s_axil_araddr[1:0]!=0 || s_axil_araddr>12'h0cc) ? 2'b10 : 2'b00;
                s_axil_rdata<=(s_axil_araddr[1:0]!=0 || s_axil_araddr>12'h0cc) ? 32'd0 : read_value;
            end
        end
    end
endmodule
