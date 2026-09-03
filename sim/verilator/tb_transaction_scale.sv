`timescale 1ns/1ps

module tb_transaction_scale (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               done,

    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,

    input  logic               q_we,
    input  logic [11:0]        q_waddr,
    input  logic [7:0]         q_wdata,
    input  logic               k_we,
    input  logic [11:0]        k_waddr,
    input  logic [7:0]         k_wdata,
    input  logic               v_we,
    input  logic [11:0]        v_waddr,
    input  logic [7:0]         v_wdata,

    output logic               dbg_start_accepted,
    output logic signed [31:0] dbg_combined_scale,
    output logic               dbg_dequant_valid,
    output logic signed [15:0] dbg_dequant_out,
    output logic signed [31:0] dbg_array_acc,
    output logic [3:0]         dbg_state
);

    logic signed [31:0] unused_out_rdata;

    flash_attn_core_banked_prefetch #(
        .TILE_SIZE(16),
        .HEAD_DIM(16),
        .SEQ_LEN(16),
        .SRAM_DEPTH(4096)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .mode(1'b0),
        .kv_len(16'b0),
        .scale_q(scale_q),
        .scale_k(scale_k),
        .scale_v(16'sh0100),
        .q_we(q_we),
        .q_waddr(q_waddr),
        .q_wdata(q_wdata),
        .k_we(k_we),
        .k_waddr(k_waddr),
        .k_wdata(k_wdata),
        .v_we(v_we),
        .v_waddr(v_waddr),
        .v_wdata(v_wdata),
        .dma_v_we(1'b0),
        .dma_v_dst(2'b0),
        .dma_v_addr(12'b0),
        .dma_v_data(128'b0),
        .out_raddr(12'b0),
        .out_rdata(unused_out_rdata),
        .causal(1'b0),
        .kv_tiles_ready(16'hffff)
    );

    assign dbg_start_accepted = u_dut.start_accepted;
    assign dbg_combined_scale = u_dut.combined_scale_reg;
    assign dbg_dequant_valid  = u_dut.dequant_valid[0];
    assign dbg_dequant_out    = u_dut.dequant_out[0];
    assign dbg_array_acc      = u_dut.array_acc[0];
    assign dbg_state          = u_dut._dbg_state;

endmodule
