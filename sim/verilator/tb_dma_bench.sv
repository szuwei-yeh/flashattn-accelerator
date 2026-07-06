// ============================================================
//  tb_dma_bench.sv — SIM-ONLY harness: dma_bench_top + external DRAM model.
// ============================================================
`timescale 1ns/1ps

module tb_dma_bench #(
    parameter int NUM_BANKS  = 16,
    parameter int SP_DEPTH   = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DRAM_DEPTH = 65536,
    localparam int SP_ADDR_W = $clog2(SP_DEPTH),
    localparam int VEC_W     = NUM_BANKS * 8
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [15:0]           rd_latency,

    // Backdoor DRAM load
    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,

    // DMA load
    input  logic                  start,
    input  logic [AXI_ADDR_W-1:0] dram_base,
    input  logic [31:0]           tile_bytes,
    output logic                  dma_load_done,

    // Stripe drain
    input  logic                  sr_start,
    input  logic [15:0]           sr_num_rows,
    output logic                  sr_out_valid,
    output logic [VEC_W-1:0]      sr_out_data,
    output logic [15:0]           sr_out_row,
    output logic                  sr_busy,
    output logic                  sr_done,

    // Scalar drain
    input  logic                  ext_r_en,
    input  logic [SP_ADDR_W-1:0]  ext_r_addr,
    output logic [7:0]            ext_r_sdata
);
    logic [AXI_ADDR_W-1:0] araddr;
    logic [7:0]            arlen;
    logic [2:0]            arsize;
    logic [1:0]            arburst;
    logic                  arvalid, arready;
    logic [AXI_DATA_W-1:0] rdata;
    logic [1:0]            rresp;
    logic                  rlast, rvalid, rready;

    dma_bench_top #(
        .NUM_BANKS(NUM_BANKS), .SP_DEPTH(SP_DEPTH),
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .dram_base(dram_base), .tile_bytes(tile_bytes),
        .dma_load_done(dma_load_done),
        .sr_start(sr_start), .sr_num_rows(sr_num_rows),
        .sr_out_valid(sr_out_valid), .sr_out_data(sr_out_data),
        .sr_out_row(sr_out_row), .sr_busy(sr_busy), .sr_done(sr_done),
        .ext_r_en(ext_r_en), .ext_r_addr(ext_r_addr), .ext_r_sdata(ext_r_sdata),
        .m_araddr(araddr), .m_arlen(arlen), .m_arsize(arsize),
        .m_arburst(arburst), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast),
        .m_rvalid(rvalid), .m_rready(rready)
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
