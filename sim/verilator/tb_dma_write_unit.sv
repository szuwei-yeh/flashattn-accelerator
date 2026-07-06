// ============================================================
//  tb_dma_write_unit.sv — Verilated harness wiring dma_write_engine ↔
//  axi_mem_model_rw over the AXI4 write channels (AW/W/B).
//
//  Exposes the descriptor input, the source-stream port (driven by C++), the
//  DRAM backdoor init/read ports (for sentinel preload + byte-exact checking),
//  and a few debug taps (W-beat fire, WLAST, AW fire) so the C++ testbench can
//  verify placement, WLAST-per-burst, and the single `done` pulse.
//
//  The read channel of the slave is unused here and tied off.
// ============================================================
`timescale 1ns/1ps

module tb_dma_write_unit #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int MAX_BURST  = 16,
    parameter int DEPTH      = 65536,
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [15:0] wr_latency,

    // Backdoor DRAM load (sentinel preload) + readback (checking)
    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,
    input  logic [AXI_ADDR_W-1:0] bd_raddr,
    output logic [7:0]            bd_rdata,

    // Descriptor
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [AXI_ADDR_W-1:0] desc_addr,
    input  logic [31:0]           desc_len_bytes,
    output logic                  done,

    // Source stream (driven by C++)
    input  logic                  src_valid,
    output logic                  src_ready,
    input  logic [AXI_DATA_W-1:0] src_data,

    // Debug taps (observed by C++)
    output logic        dbg_w_fire,     // W beat accepted this cycle
    output logic        dbg_w_last,     // WLAST on the accepted beat
    output logic        dbg_aw_fire     // AW accepted this cycle
);
    // AXI write channel between master (DMA) and slave (DRAM)
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

    assign dbg_w_fire  = wvalid & wready;
    assign dbg_w_last  = wlast;
    assign dbg_aw_fire = awvalid & awready;

    dma_write_engine #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .MAX_BURST(MAX_BURST)
    ) u_dma_wr (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_len_bytes(desc_len_bytes), .done(done),
        .src_valid(src_valid), .src_ready(src_ready), .src_data(src_data),
        .m_awaddr(awaddr), .m_awlen(awlen), .m_awsize(awsize),
        .m_awburst(awburst), .m_awvalid(awvalid), .m_awready(awready),
        .m_wdata(wdata), .m_wstrb(wstrb), .m_wlast(wlast),
        .m_wvalid(wvalid), .m_wready(wready),
        .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready)
    );

    axi_mem_model_rw #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DEPTH)
    ) u_mem (
        .clk(clk), .rst_n(rst_n),
        .rd_latency(16'd0), .wr_latency(wr_latency),
        .init_we(init_we), .init_addr(init_addr), .init_data(init_data),
        .bd_raddr(bd_raddr), .bd_rdata(bd_rdata),
        // read channel unused — tied off
        /* verilator lint_off PINCONNECTEMPTY */
        .s_araddr('0), .s_arlen('0), .s_arsize('0), .s_arburst('0),
        .s_arvalid(1'b0), .s_arready(),
        .s_rdata(), .s_rresp(), .s_rlast(), .s_rvalid(), .s_rready(1'b0),
        /* verilator lint_on PINCONNECTEMPTY */
        // write channel
        .s_awaddr(awaddr), .s_awlen(awlen), .s_awsize(awsize),
        .s_awburst(awburst), .s_awvalid(awvalid), .s_awready(awready),
        .s_wdata(wdata), .s_wstrb(wstrb), .s_wlast(wlast),
        .s_wvalid(wvalid), .s_wready(wready),
        .s_bresp(bresp), .s_bvalid(bvalid), .s_bready(bready)
    );
endmodule
