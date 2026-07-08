// ============================================================
//  dma_bench_top.sv — Stage-2 microbenchmark (isolated; no core changes)
//
//  Path:  axi_mem_model (DRAM) → dma_engine (AXI4 read master, 1 byte/cyc)
//         → banked_scratchpad → { scalar read port | stripe_reader 16 B/cyc }
//
//  Demonstrates the banked scratchpad's read bandwidth: the same DMA-loaded
//  tile can be drained one byte/cycle (legacy-style) or one 16-byte stripe/
//  cycle (banked).  The DMA still writes byte-by-byte (Stage-2 keeps DMA
//  write-out unchanged); the bandwidth win is on the tile-load read path.
// ============================================================
`timescale 1ns/1ps

module dma_bench_top #(
    parameter int NUM_BANKS  = 16,
    parameter int SP_DEPTH   = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    localparam int SP_ADDR_W = $clog2(SP_DEPTH),
    localparam int VEC_W     = NUM_BANKS * 8
)(
    input  logic clk,
    input  logic rst_n,

    // DMA tile-load trigger
    input  logic                  start,
    input  logic [AXI_ADDR_W-1:0] dram_base,
    input  logic [31:0]           tile_bytes,
    output logic                  dma_load_done,   // level: high once tile resident

    // Stripe-reader (banked 16 B/cyc) drain control
    input  logic                  sr_start,
    input  logic [15:0]           sr_num_rows,
    output logic                  sr_out_valid,
    output logic [VEC_W-1:0]      sr_out_data,
    output logic [15:0]           sr_out_row,
    output logic                  sr_busy,
    output logic                  sr_done,

    // Scalar (1 B/cyc) drain — driven externally for the comparison path
    input  logic                  ext_r_en,
    input  logic [SP_ADDR_W-1:0]  ext_r_addr,
    output logic [7:0]            ext_r_sdata,

    // ── AXI4 read-master ports (to external DRAM) ────────────────────
    output logic [AXI_ADDR_W-1:0] m_araddr,
    output logic [7:0]            m_arlen,
    output logic [2:0]            m_arsize,
    output logic [1:0]            m_arburst,
    output logic                  m_arvalid,
    input  logic                  m_arready,
    input  logic [AXI_DATA_W-1:0] m_rdata,
    input  logic [1:0]            m_rresp,
    input  logic                  m_rlast,
    input  logic                  m_rvalid,
    output logic                  m_rready
);

    // ── DMA engine ───────────────────────────────────────────────────
    logic                  desc_valid, desc_ready, dma_done;
    logic [AXI_ADDR_W-1:0] desc_addr;
    logic [31:0]           desc_len_bytes;
    logic                  w_we;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [1:0]            w_dst;   // single scratchpad here — dst ignored
    /* verilator lint_on UNUSEDSIGNAL */
    logic [11:0]           w_addr;
    logic [7:0]            w_data;

    dma_engine #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)) u_dma (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_dst_addr(12'd0),
        .desc_len_bytes(desc_len_bytes), .desc_dst(2'd1), .done(dma_done),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize),
        .m_arburst(m_arburst), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_rvalid(m_rvalid), .m_rready(m_rready),
        .w_we(w_we), .w_dst(w_dst), .w_addr(w_addr), .w_data(w_data)
    );

    // ── DMA-load FSM: one descriptor on `start` ──────────────────────
    typedef enum logic [1:0] { LD_IDLE, LD_ISSUE, LD_WAIT, LD_DONE } ld_t;
    ld_t ld_state;

    assign desc_valid     = (ld_state == LD_ISSUE);
    assign desc_addr      = dram_base;
    assign desc_len_bytes = tile_bytes;
    assign dma_load_done  = (ld_state == LD_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ld_state <= LD_IDLE;
        else begin
            case (ld_state)
                LD_IDLE:  if (start)      ld_state <= LD_ISSUE;
                LD_ISSUE: if (desc_ready) ld_state <= LD_WAIT;
                LD_WAIT:  if (dma_done)   ld_state <= LD_DONE;
                LD_DONE:  ;   // stays resident until reset
                default:  ld_state <= LD_IDLE;
            endcase
        end
    end

    // ── Stripe reader ────────────────────────────────────────────────
    logic                 sr_r_en, sr_r_vec;
    logic [SP_ADDR_W-1:0] sr_r_addr;
    logic [VEC_W-1:0]     sp_r_vdata;

    stripe_reader #(.NUM_BANKS(NUM_BANKS), .ADDR_W(SP_ADDR_W)) u_sr (
        .clk(clk), .rst_n(rst_n),
        .start(sr_start), .base_addr({SP_ADDR_W{1'b0}}), .num_rows(sr_num_rows),
        .r_en(sr_r_en), .r_vec(sr_r_vec), .r_addr(sr_r_addr), .r_vdata(sp_r_vdata),
        .out_valid(sr_out_valid), .out_data(sr_out_data), .out_row(sr_out_row),
        .busy(sr_busy), .done(sr_done)
    );

    // ── Scratchpad read-port mux: stripe reader vs external scalar ───
    logic                 sp_r_en, sp_r_vec;
    logic [SP_ADDR_W-1:0] sp_r_addr;
    logic [7:0]           sp_r_sdata;

    assign sp_r_en   = sr_busy ? sr_r_en   : ext_r_en;
    assign sp_r_vec  = sr_busy ? sr_r_vec  : 1'b0;
    assign sp_r_addr = sr_busy ? sr_r_addr : ext_r_addr;
    assign ext_r_sdata = sp_r_sdata;

    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SP_DEPTH)) u_sp (
        .clk(clk),
        .w_en(w_we), .w_vec(1'b0), .w_addr(SP_ADDR_W'(w_addr)),
        .w_sdata(w_data), .w_vdata({VEC_W{1'b0}}),
        .r_en(sp_r_en), .r_vec(sp_r_vec), .r_addr(sp_r_addr),
        .r_sdata(sp_r_sdata), .r_vdata(sp_r_vdata)
    );

endmodule
