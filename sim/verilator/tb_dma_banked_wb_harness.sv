// ============================================================
//  tb_dma_banked_wb_harness.sv — SIM harness: flash_attn_top_dma_banked_wb +
//  axi_mem_model_rw (read+write DRAM).
//
//  The single RW memory model carries both the read (AR/R) channels the read
//  DMA uses to fetch Q/K/V and the write (AW/W/B) channels the write-back DMA
//  uses to store O.  The C++ testbench backdoor-loads Q/K/V, runs the top, then
//  backdoor-reads the O region and compares against the golden expected.hex.
// ============================================================
`timescale 1ns/1ps

module tb_dma_banked_wb_harness #(
    parameter int SEQ_LEN    = 64,
    parameter int HEAD_DIM   = 16,
    parameter int TILE_SIZE  = 16,
    parameter int SRAM_DEPTH = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DRAM_DEPTH = 65536,
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8
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
    input  logic [15:0] wr_latency,

    // Backdoor DRAM load (Q/K/V) + readback (O checking)
    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,
    input  logic [AXI_ADDR_W-1:0] bd_raddr,
    output logic [7:0]            bd_rdata,

    // Runtime scheduler config
    input  logic [15:0] cfg_seq_len,
    input  logic [31:0] cfg_q_base,
    input  logic [31:0] cfg_k_base,
    input  logic [31:0] cfg_v_base,
    input  logic [31:0] cfg_o_base,
    output logic        cfg_error,

    // Read-side performance counters
    output logic [31:0] perf_total_cycles,
    output logic [31:0] perf_dma_busy_cycles,
    output logic [31:0] perf_core_busy_cycles,
    output logic [31:0] perf_dma_bytes,
    output logic [15:0] perf_kv_tiles_loaded,
    output logic [31:0] perf_first_tile_wait_cycles,

    // Write-back performance counters
    output logic [31:0] perf_wb_bytes,
    output logic [31:0] perf_wb_cycles,
    output logic [31:0] perf_wb_beats,

    output logic [15:0] dbg_kv_tiles_ready
);
    // Read channel (top read-master ↔ DRAM slave)
    logic [AXI_ADDR_W-1:0] araddr;
    logic [7:0]            arlen;
    logic [2:0]            arsize;
    logic [1:0]            arburst;
    logic                  arvalid, arready;
    logic [AXI_DATA_W-1:0] rdata;
    logic [1:0]            rresp;
    logic                  rlast, rvalid, rready;

    // Write channel (top write-master ↔ DRAM slave)
    logic [AXI_ADDR_W-1:0]     awaddr;
    logic [7:0]                awlen;
    logic [2:0]                awsize;
    logic [1:0]                awburst;
    logic                      awvalid, awready;
    logic [AXI_DATA_W-1:0]     wdata;
    logic [BYTES_PER_BEAT-1:0] wstrb;
    logic                      wlast, wvalid, wready;
    logic [1:0]                bresp;
    logic                      bvalid, bready;

    flash_attn_top_dma_banked_wb #(
        .SEQ_LEN(SEQ_LEN), .HEAD_DIM(HEAD_DIM), .TILE_SIZE(TILE_SIZE),
        .SRAM_DEPTH(SRAM_DEPTH), .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) u_dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .causal(causal), .scale_q(scale_q), .scale_k(scale_k), .scale_v(scale_v),
        .cfg_seq_len(cfg_seq_len), .cfg_q_base(cfg_q_base),
        .cfg_k_base(cfg_k_base), .cfg_v_base(cfg_v_base),
        .cfg_o_base(cfg_o_base), .cfg_error(cfg_error),
        .m_araddr(araddr), .m_arlen(arlen), .m_arsize(arsize),
        .m_arburst(arburst), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast),
        .m_rvalid(rvalid), .m_rready(rready),
        .m_awaddr(awaddr), .m_awlen(awlen), .m_awsize(awsize),
        .m_awburst(awburst), .m_awvalid(awvalid), .m_awready(awready),
        .m_wdata(wdata), .m_wstrb(wstrb), .m_wlast(wlast),
        .m_wvalid(wvalid), .m_wready(wready),
        .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready),
        .perf_total_cycles(perf_total_cycles),
        .perf_dma_busy_cycles(perf_dma_busy_cycles),
        .perf_core_busy_cycles(perf_core_busy_cycles),
        .perf_dma_bytes(perf_dma_bytes),
        .perf_kv_tiles_loaded(perf_kv_tiles_loaded),
        .perf_first_tile_wait_cycles(perf_first_tile_wait_cycles),
        .perf_wb_bytes(perf_wb_bytes),
        .perf_wb_cycles(perf_wb_cycles),
        .perf_wb_beats(perf_wb_beats),
        .dbg_kv_tiles_ready(dbg_kv_tiles_ready)
    );

    axi_mem_model_rw #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DRAM_DEPTH)
    ) u_dram (
        .clk(clk), .rst_n(rst_n),
        .rd_latency(rd_latency), .wr_latency(wr_latency),
        .init_we(init_we), .init_addr(init_addr), .init_data(init_data),
        .bd_raddr(bd_raddr), .bd_rdata(bd_rdata),
        .s_araddr(araddr), .s_arlen(arlen), .s_arsize(arsize),
        .s_arburst(arburst), .s_arvalid(arvalid), .s_arready(arready),
        .s_rdata(rdata), .s_rresp(rresp), .s_rlast(rlast),
        .s_rvalid(rvalid), .s_rready(rready),
        .s_awaddr(awaddr), .s_awlen(awlen), .s_awsize(awsize),
        .s_awburst(awburst), .s_awvalid(awvalid), .s_awready(awready),
        .s_wdata(wdata), .s_wstrb(wstrb), .s_wlast(wlast),
        .s_wvalid(wvalid), .s_wready(wready),
        .s_bresp(bresp), .s_bvalid(bvalid), .s_bready(bready)
    );
endmodule
