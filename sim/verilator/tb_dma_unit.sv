// ============================================================
//  tb_dma_unit.sv — Verilated harness wiring dma_engine ↔ axi_mem_model.
//  Exposes descriptor input, backdoor DRAM load, and the scratchpad write
//  stream so the C++ testbench can check that bytes land correctly.
// ============================================================
`timescale 1ns/1ps

module tb_dma_unit #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DEPTH      = 65536
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [15:0] rd_latency,

    // Backdoor DRAM load
    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,

    // Descriptor
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [AXI_ADDR_W-1:0] desc_addr,
    input  logic [11:0]           desc_dst_addr,
    input  logic [31:0]           desc_len_bytes,
    input  logic [1:0]            desc_dst,
    output logic                  done,

    // Scratchpad write stream (observed by C++)
    output logic        w_we,
    output logic [1:0]  w_dst,
    output logic [11:0] w_addr,
    output logic [7:0]  w_data
);
    // AXI read channel between master (DMA) and slave (DRAM)
    logic [AXI_ADDR_W-1:0] araddr;
    logic [7:0]            arlen;
    logic [2:0]            arsize;
    logic [1:0]            arburst;
    logic                  arvalid, arready;
    logic [AXI_DATA_W-1:0] rdata;
    logic [1:0]            rresp;
    logic                  rlast, rvalid, rready;

    dma_engine #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)) u_dma (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_dst_addr(desc_dst_addr),
        .desc_len_bytes(desc_len_bytes), .desc_dst(desc_dst), .done(done),
        .m_araddr(araddr), .m_arlen(arlen), .m_arsize(arsize),
        .m_arburst(arburst), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast),
        .m_rvalid(rvalid), .m_rready(rready),
        .w_we(w_we), .w_dst(w_dst), .w_addr(w_addr), .w_data(w_data)
    );

    axi_mem_model #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DEPTH)) u_mem (
        .clk(clk), .rst_n(rst_n), .rd_latency(rd_latency),
        .init_we(init_we), .init_addr(init_addr), .init_data(init_data),
        .s_araddr(araddr), .s_arlen(arlen), .s_arsize(arsize),
        .s_arburst(arburst), .s_arvalid(arvalid), .s_arready(arready),
        .s_rdata(rdata), .s_rresp(rresp), .s_rlast(rlast),
        .s_rvalid(rvalid), .s_rready(rready)
    );
endmodule
