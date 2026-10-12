// Two banked K/V stores for one persistent d16/d64 KV group; capacity 64 tokens.
`timescale 1ns/1ps
module kv_cache_banked_decode_m2c3 #(parameter int HEAD_DIM=16) (
    input logic clk, rst_n,
    input logic cmd_begin, cmd_commit, cmd_fault,
    input logic [1:0] cmd_op,
    input logic [6:0] cmd_length,
    input logic signed [15:0] cmd_scale_k, cmd_scale_v,
    input logic w_en, w_sel,
    input logic [11:0] w_addr,
    input logic [127:0] w_data,
    input logic core_r_en,
    input logic [11:0] core_r_addr,
    output logic [127:0] core_k_data, core_v_data,
    output logic core_r_valid,
    input logic dbg_enable, dbg_req_valid,
    output logic dbg_req_ready, dbg_resp_valid, dbg_error,
    input logic [11:0] dbg_addr,
    output logic [127:0] dbg_k_data, dbg_v_data,
    output logic cache_valid,
    output logic [6:0] cache_length,
    output logic signed [15:0] cache_scale_k, cache_scale_v,
    output logic [31:0] cache_epoch,
    output logic pair_complete, write_accept, read_accept
);
    localparam logic [1:0] PREFILL=2'd0, APPEND=2'd1, DECODE=2'd2, CLEAR=2'd3;
    logic owner, fault_seen;
    logic [6:0] k_stored, v_stored;
    logic [6:0] stripes_required;
    logic [1:0] op;
    assign stripes_required = op==PREFILL ? 7'(16*HEAD_DIM/16) : 7'(HEAD_DIM/16);
    logic cache_valid_reg;
    logic [6:0] cache_length_reg;
    assign cache_valid=cache_valid_reg && !cmd_fault;
    assign cache_length=cmd_fault ? 7'd0 : cache_length_reg;
    logic [6:0] pending_length;
    logic signed [15:0] pending_sk, pending_sv;
    logic [12:0] write_base, committed_end, read_end, dbg_end;
    logic write_ok, core_read_ok, dbg_accept, dbg_good;
    logic mem_re;
    logic [11:0] mem_raddr;
    logic [127:0] k_data, v_data;
    assign write_base = op==PREFILL ? 13'd0 : 13'(pending_length)*13'(HEAD_DIM);
    assign committed_end = 13'(cache_length)*13'(HEAD_DIM);
    assign read_end = {1'b0,core_r_addr}+13'd16;
    assign dbg_end = {1'b0,dbg_addr}+13'd16;
    // Ordered K then V: N16 prefill or one append token; publish complete pairs.
    assign write_ok = owner && (op==APPEND || op==PREFILL) && !cmd_fault && (op==PREFILL || pending_length<64) &&
                      w_addr[3:0]==0 &&
                      ((!w_sel && k_stored<stripes_required && v_stored==0 &&
                        {1'b0,w_addr}==write_base+13'(k_stored)*13'd16) ||
                       (w_sel && k_stored==stripes_required && v_stored<stripes_required &&
                        {1'b0,w_addr}==write_base+13'(v_stored)*13'd16));
    assign write_accept = w_en && write_ok;
    assign pair_complete = owner && (op==APPEND || op==PREFILL) && k_stored==stripes_required && v_stored==stripes_required;
    assign core_read_ok = owner && !cmd_fault && ((op==DECODE && cache_valid) || (op==PREFILL && pair_complete)) &&
                          core_r_addr[3:0]==0 && read_end<=(op==PREFILL ? 13'(pending_length)*13'(HEAD_DIM) : committed_end);
    assign read_accept = core_r_en && core_read_ok;
    assign dbg_req_ready = dbg_enable && !owner && !dbg_resp_valid && !cmd_fault;
    assign dbg_accept = dbg_req_valid && dbg_req_ready;
    assign dbg_good = cache_valid && dbg_addr[3:0]==0 && dbg_end<=committed_end;
    assign mem_re = read_accept || (dbg_accept && dbg_good);
    assign mem_raddr = read_accept ? core_r_addr : dbg_addr;
    assign core_k_data = core_r_valid ? k_data : '0;
    assign core_v_data = core_r_valid ? v_data : '0;
    assign dbg_k_data = dbg_resp_valid ? k_data : '0;
    assign dbg_v_data = dbg_resp_valid ? v_data : '0;
    /* verilator lint_off PINCONNECTEMPTY */
    banked_scratchpad u_k_sp (
        .clk(clk), .w_en(write_accept && !w_sel), .w_vec(1'b1), .w_addr(w_addr),
        .w_sdata(8'd0), .w_vdata(w_data), .r_en(mem_re), .r_vec(1'b1),
        .r_addr(mem_raddr), .r_sdata(), .r_vdata(k_data)
    );
    banked_scratchpad u_v_sp (
        .clk(clk), .w_en(write_accept && w_sel), .w_vec(1'b1), .w_addr(w_addr),
        .w_sdata(8'd0), .w_vdata(w_data), .r_en(mem_re), .r_vec(1'b1),
        .r_addr(mem_raddr), .r_sdata(), .r_vdata(v_data)
    );
    /* verilator lint_on PINCONNECTEMPTY */
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            owner<=0; op<=0; fault_seen<=0; k_stored<=0; v_stored<=0;
            pending_length<=0; pending_sk<=0; pending_sv<=0;
            cache_valid_reg<=0; cache_length_reg<=0; cache_scale_k<=0; cache_scale_v<=0;
            cache_epoch<=0; core_r_valid<=0; dbg_resp_valid<=0; dbg_error<=0;
        end else begin
            core_r_valid<=read_accept;
            dbg_resp_valid<=dbg_accept && dbg_good;
            dbg_error<=dbg_accept && !dbg_good;
            if (cmd_fault) begin
                owner<=0; cache_valid_reg<=0; cache_length_reg<=0;
                core_r_valid<=0; dbg_resp_valid<=0;
                if (!fault_seen) cache_epoch<=cache_epoch+1;
                fault_seen<=1;
            end else if (cmd_begin) begin
                owner<=1; op<=cmd_op; pending_length<=cmd_length;
                pending_sk<=cmd_scale_k; pending_sv<=cmd_scale_v;
                k_stored<=0; v_stored<=0;
                if(cmd_op==PREFILL) begin cache_valid_reg<=0;cache_length_reg<=0;end
            end else begin
                if (write_accept && !w_sel) k_stored<=k_stored+1'b1;
                if (write_accept && w_sel) v_stored<=v_stored+1'b1;
                if (cmd_commit) begin
                    owner<=0;
                    if ((op==APPEND || op==PREFILL) && pair_complete) begin
                        cache_valid_reg<=1; cache_length_reg<=op==PREFILL ? pending_length : pending_length+7'd1;
                        if (op==PREFILL || pending_length==0) begin
                            cache_scale_k<=pending_sk; cache_scale_v<=pending_sv;
                        end
                        cache_epoch<=cache_epoch+1;
                    end else if (op==CLEAR) begin
                        cache_valid_reg<=0; cache_length_reg<=0; cache_epoch<=cache_epoch+1;
                    end
                end
            end
        end
    end
    // synthesis translate_off
    always @(posedge clk or negedge rst_n) if (rst_n && !cmd_fault) begin
        if (w_en && !write_ok) $fatal(1,"cache write outside owned atomic append window");
        if (core_r_en && !core_read_ok) $fatal(1,"cache read outside committed decode ownership");
        if (cmd_begin && owner) $fatal(1,"cache command ownership collision");
        if (cmd_commit && (!owner || ((op==APPEND || op==PREFILL) && !pair_complete)))
            $fatal(1,"cache commit before paired stores");
        if (mem_re && write_accept) $fatal(1,"M2B cache read/write collision");
    end
    // synthesis translate_on
endmodule
