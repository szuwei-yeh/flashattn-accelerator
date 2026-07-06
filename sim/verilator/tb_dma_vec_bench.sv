// ============================================================
//  tb_dma_vec_bench.sv — SIM harness: dma_vec_bench_top + two DRAM models.
//  Backdoor-loads the same tile into both DRAMs so the scalar and vector fill
//  paths read identical data.
// ============================================================
`timescale 1ns/1ps

module tb_dma_vec_bench #(
    parameter int NUM_BANKS  = 16,
    parameter int SP_DEPTH   = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DRAM_DEPTH = 65536,
    localparam int SP_ADDR_W = $clog2(SP_DEPTH)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [15:0]           rd_latency,

    // Backdoor DRAM load (written to both DRAMs)
    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,

    input  logic [AXI_ADDR_W-1:0] dram_base,
    input  logic [31:0]           tile_bytes,

    // Path A
    input  logic                  start_a,
    output logic                  done_a,
    output logic                  o_a_we,
    input  logic [SP_ADDR_W-1:0]  r_a_addr,
    output logic [7:0]            r_a_sdata,

    // Path B
    input  logic                  start_b,
    output logic                  done_b,
    output logic                  o_b_we,
    input  logic [SP_ADDR_W-1:0]  r_b_addr,
    output logic [7:0]            r_b_sdata
);
    // AXI A
    logic [AXI_ADDR_W-1:0] a_araddr; logic [7:0] a_arlen; logic [2:0] a_arsize;
    logic [1:0] a_arburst; logic a_arvalid, a_arready;
    logic [AXI_DATA_W-1:0] a_rdata; logic [1:0] a_rresp; logic a_rlast, a_rvalid, a_rready;
    // AXI B
    logic [AXI_ADDR_W-1:0] b_araddr; logic [7:0] b_arlen; logic [2:0] b_arsize;
    logic [1:0] b_arburst; logic b_arvalid, b_arready;
    logic [AXI_DATA_W-1:0] b_rdata; logic [1:0] b_rresp; logic b_rlast, b_rvalid, b_rready;

    dma_vec_bench_top #(
        .NUM_BANKS(NUM_BANKS), .SP_DEPTH(SP_DEPTH),
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .dram_base(dram_base), .tile_bytes(tile_bytes),
        .start_a(start_a), .done_a(done_a), .o_a_we(o_a_we),
        .r_a_addr(r_a_addr), .r_a_sdata(r_a_sdata),
        .start_b(start_b), .done_b(done_b), .o_b_we(o_b_we),
        .r_b_addr(r_b_addr), .r_b_sdata(r_b_sdata),
        .a_araddr(a_araddr), .a_arlen(a_arlen), .a_arsize(a_arsize),
        .a_arburst(a_arburst), .a_arvalid(a_arvalid), .a_arready(a_arready),
        .a_rdata(a_rdata), .a_rresp(a_rresp), .a_rlast(a_rlast),
        .a_rvalid(a_rvalid), .a_rready(a_rready),
        .b_araddr(b_araddr), .b_arlen(b_arlen), .b_arsize(b_arsize),
        .b_arburst(b_arburst), .b_arvalid(b_arvalid), .b_arready(b_arready),
        .b_rdata(b_rdata), .b_rresp(b_rresp), .b_rlast(b_rlast),
        .b_rvalid(b_rvalid), .b_rready(b_rready)
    );

    axi_mem_model #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DRAM_DEPTH)) u_dram_a (
        .clk(clk), .rst_n(rst_n), .rd_latency(rd_latency),
        .init_we(init_we), .init_addr(init_addr), .init_data(init_data),
        .s_araddr(a_araddr), .s_arlen(a_arlen), .s_arsize(a_arsize),
        .s_arburst(a_arburst), .s_arvalid(a_arvalid), .s_arready(a_arready),
        .s_rdata(a_rdata), .s_rresp(a_rresp), .s_rlast(a_rlast),
        .s_rvalid(a_rvalid), .s_rready(a_rready)
    );

    axi_mem_model #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DRAM_DEPTH)) u_dram_b (
        .clk(clk), .rst_n(rst_n), .rd_latency(rd_latency),
        .init_we(init_we), .init_addr(init_addr), .init_data(init_data),
        .s_araddr(b_araddr), .s_arlen(b_arlen), .s_arsize(b_arsize),
        .s_arburst(b_arburst), .s_arvalid(b_arvalid), .s_arready(b_arready),
        .s_rdata(b_rdata), .s_rresp(b_rresp), .s_rlast(b_rlast),
        .s_rvalid(b_rvalid), .s_rready(b_rready)
    );
endmodule
