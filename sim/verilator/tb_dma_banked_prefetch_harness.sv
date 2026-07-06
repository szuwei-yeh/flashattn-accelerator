// ============================================================
//  tb_dma_banked_prefetch_harness.sv — SIM harness:
//  flash_attn_top_dma_banked_prefetch + DRAM model.
//
//  Identical to tb_dma_banked_harness.sv but wraps the prefetch-variant top.
// ============================================================
`timescale 1ns/1ps

module tb_dma_banked_prefetch_harness #(
    parameter int SEQ_LEN    = 64,
    parameter int HEAD_DIM   = 16,
    parameter int TILE_SIZE  = 16,
    parameter int SRAM_DEPTH = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DRAM_DEPTH = 65536
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic        causal,
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    input  logic [15:0] rd_latency,

    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,

    input  logic [11:0]        out_raddr,
    output logic signed [31:0] out_rdata,

    // Runtime DMA scheduler config
    input  logic [15:0] cfg_seq_len,
    input  logic [31:0] cfg_q_base,
    input  logic [31:0] cfg_k_base,
    input  logic [31:0] cfg_v_base,
    output logic        cfg_error,

    // Performance counters
    output logic [31:0] perf_total_cycles,
    output logic [31:0] perf_dma_busy_cycles,
    output logic [31:0] perf_core_busy_cycles,
    output logic [31:0] perf_dma_bytes,
    output logic [15:0] perf_kv_tiles_loaded,
    output logic [31:0] perf_first_tile_wait_cycles,

    output logic [15:0] dbg_kv_tiles_ready
);
    logic [AXI_ADDR_W-1:0] araddr;
    logic [7:0]            arlen;
    logic [2:0]            arsize;
    logic [1:0]            arburst;
    logic                  arvalid, arready;
    logic [AXI_DATA_W-1:0] rdata;
    logic [1:0]            rresp;
    logic                  rlast, rvalid, rready;

    flash_attn_top_dma_banked_prefetch #(
        .SEQ_LEN(SEQ_LEN), .HEAD_DIM(HEAD_DIM), .TILE_SIZE(TILE_SIZE),
        .SRAM_DEPTH(SRAM_DEPTH), .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) u_dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .causal(causal), .scale_q(scale_q), .scale_k(scale_k), .scale_v(scale_v),
        .cfg_seq_len(cfg_seq_len), .cfg_q_base(cfg_q_base),
        .cfg_k_base(cfg_k_base), .cfg_v_base(cfg_v_base), .cfg_error(cfg_error),
        .out_raddr(out_raddr), .out_rdata(out_rdata),
        .m_araddr(araddr), .m_arlen(arlen), .m_arsize(arsize),
        .m_arburst(arburst), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast),
        .m_rvalid(rvalid), .m_rready(rready),
        .perf_total_cycles(perf_total_cycles),
        .perf_dma_busy_cycles(perf_dma_busy_cycles),
        .perf_core_busy_cycles(perf_core_busy_cycles),
        .perf_dma_bytes(perf_dma_bytes),
        .perf_kv_tiles_loaded(perf_kv_tiles_loaded),
        .perf_first_tile_wait_cycles(perf_first_tile_wait_cycles),
        .dbg_kv_tiles_ready(dbg_kv_tiles_ready)
    );

    axi_mem_model #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DRAM_DEPTH)
    ) u_dram (
        .clk(clk), .rst_n(rst_n), .rd_latency(rd_latency),
        .init_we(init_we), .init_addr(init_addr), .init_data(init_data),
        .s_araddr(araddr), .s_arlen(arlen), .s_arsize(arsize),
        .s_arburst(arburst), .s_arvalid(arvalid), .s_arready(arready),
        .s_rdata(rdata), .s_rresp(rresp), .s_rlast(rlast),
        .s_rvalid(rvalid), .s_rready(rready)
    );
endmodule
