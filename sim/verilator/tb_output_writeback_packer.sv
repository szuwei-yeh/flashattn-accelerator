// ============================================================
//  tb_output_writeback_packer.sv — Stage WB-2 microbenchmark harness.
//
//  Wires the full post-output-buffer write-back datapath:
//    output-buffer-like SRAM (sram_1r1w) → output_writeback_packer
//      → dma_write_engine → axi_mem_model_rw (DRAM)
//
//  The C++ testbench preloads deterministic 32-bit words into the source SRAM
//  (its write port), kicks the packer + write descriptor, then backdoor-reads
//  DRAM to verify byte-exact placement.  Debug taps expose the W-beat fire and
//  the src valid/ready pair so the C++ can count bytes and observe back-pressure.
//
//  Reuses dma_write_engine.sv and axi_mem_model_rw.sv unchanged from WB-1.
// ============================================================
`timescale 1ns/1ps

module tb_output_writeback_packer #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int MAX_BURST  = 16,
    parameter int OUT_DEPTH  = 4096,          // source words (max = N*d = 64*64)
    parameter int DRAM_DEPTH = 65536,
    localparam int OUT_ADDR_W     = $clog2(OUT_DEPTH),
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [15:0] wr_latency,

    // Source-SRAM preload (output-buffer-like) write port
    input  logic                  obuf_we,
    input  logic [OUT_ADDR_W-1:0] obuf_waddr,
    input  logic [31:0]           obuf_wdata,

    // Control
    input  logic                  start,
    input  logic [15:0]           num_words,
    output logic                  busy,
    output logic                  pkr_done,

    // Write descriptor (to dma_write_engine)
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [AXI_ADDR_W-1:0] desc_addr,
    input  logic [31:0]           desc_len_bytes,
    output logic                  dma_done,

    // DRAM backdoor init (sentinel preload) + readback (checking)
    input  logic                  init_we,
    input  logic [AXI_ADDR_W-1:0] init_addr,
    input  logic [7:0]            init_data,
    input  logic [AXI_ADDR_W-1:0] bd_raddr,
    output logic [7:0]            bd_rdata,

    // Debug taps
    output logic dbg_w_fire,      // AXI W beat accepted this cycle
    output logic dbg_src_valid,   // packer presenting a beat
    output logic dbg_src_ready    // write engine accepting a beat
);
    // ── output-buffer-like source SRAM ───────────────────────────────
    logic                  out_re;
    logic [OUT_ADDR_W-1:0] out_raddr;
    logic [31:0]           out_rdata;

    sram_1r1w #(.DATA_WIDTH(32), .DEPTH(OUT_DEPTH)) u_outbuf (
        .clk(clk),
        .we(obuf_we), .waddr(obuf_waddr), .wdata(obuf_wdata),
        .re(out_re), .raddr(out_raddr), .rdata(out_rdata)
    );

    // ── Packer ────────────────────────────────────────────────────────
    logic                  src_valid, src_ready;
    logic [AXI_DATA_W-1:0] src_data;

    output_writeback_packer #(
        .DATA_W(32), .AXI_DATA_W(AXI_DATA_W), .OUT_ADDR_W(OUT_ADDR_W)
    ) u_packer (
        .clk(clk), .rst_n(rst_n),
        .start(start), .num_words(num_words), .busy(busy), .done(pkr_done),
        .out_re(out_re), .out_raddr(out_raddr), .out_rdata(out_rdata),
        .src_valid(src_valid), .src_ready(src_ready), .src_data(src_data)
    );

    // ── Write master + DRAM model ─────────────────────────────────────
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

    assign dbg_w_fire    = wvalid & wready;
    assign dbg_src_valid = src_valid;
    assign dbg_src_ready = src_ready;

    dma_write_engine #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .MAX_BURST(MAX_BURST)
    ) u_dma_wr (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_len_bytes(desc_len_bytes), .done(dma_done),
        .src_valid(src_valid), .src_ready(src_ready), .src_data(src_data),
        .m_awaddr(awaddr), .m_awlen(awlen), .m_awsize(awsize),
        .m_awburst(awburst), .m_awvalid(awvalid), .m_awready(awready),
        .m_wdata(wdata), .m_wstrb(wstrb), .m_wlast(wlast),
        .m_wvalid(wvalid), .m_wready(wready),
        .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready)
    );

    axi_mem_model_rw #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .DEPTH(DRAM_DEPTH)
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
