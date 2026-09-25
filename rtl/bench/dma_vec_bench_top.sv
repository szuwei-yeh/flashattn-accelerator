// ============================================================
//  dma_vec_bench_top.sv — Stage-3A microbenchmark: byte-serial vs vector DMA fill
//
//  Two independent fill paths into two banked_scratchpads, for a side-by-side
//  comparison of how the DMA WRITES into the scratchpad:
//    A) dma_engine      → scalar (1 byte/cycle) writes   (existing path)
//    B) dma_engine_vec  → vector (16 byte/stripe) writes  (Stage-3A)
//
//  Each path has its own AXI master (wired to its own DRAM model in the
//  harness) and its own load FSM.  Scratchpad read ports are exposed for
//  read-back verification; the per-path scratchpad write-enables are exposed
//  so the C++ testbench can count scratchpad write cycles.
//
//  No flash_attn_core; no change to flash_attn_top_dma / dma_engine.
// ============================================================
`timescale 1ns/1ps

module dma_vec_bench_top #(
    parameter int NUM_BANKS  = 16,
    parameter int SP_DEPTH   = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    localparam int SP_ADDR_W = $clog2(SP_DEPTH),
    localparam int VEC_W     = NUM_BANKS * 8
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [AXI_ADDR_W-1:0] dram_base,
    input  logic [31:0]           tile_bytes,

    // Path A (scalar) control + observability
    input  logic                  start_a,
    output logic                  done_a,
    output logic                  o_a_we,         // scalar scratchpad write enable
    input  logic [SP_ADDR_W-1:0]  r_a_addr,
    output logic [7:0]            r_a_sdata,

    // Path B (vector) control + observability
    input  logic                  start_b,
    output logic                  done_b,
    output logic                  o_b_we,         // vector scratchpad write enable
    input  logic [SP_ADDR_W-1:0]  r_b_addr,
    output logic [7:0]            r_b_sdata,

    // AXI master A (scalar engine → DRAM A)
    output logic [AXI_ADDR_W-1:0] a_araddr,
    output logic [7:0]            a_arlen,
    output logic [2:0]            a_arsize,
    output logic [1:0]            a_arburst,
    output logic                  a_arvalid,
    input  logic                  a_arready,
    input  logic [AXI_DATA_W-1:0] a_rdata,
    input  logic [1:0]            a_rresp,
    input  logic                  a_rlast,
    input  logic                  a_rvalid,
    output logic                  a_rready,

    // AXI master B (vector engine → DRAM B)
    output logic [AXI_ADDR_W-1:0] b_araddr,
    output logic [7:0]            b_arlen,
    output logic [2:0]            b_arsize,
    output logic [1:0]            b_arburst,
    output logic                  b_arvalid,
    input  logic                  b_arready,
    input  logic [AXI_DATA_W-1:0] b_rdata,
    input  logic [1:0]            b_rresp,
    input  logic                  b_rlast,
    input  logic                  b_rvalid,
    output logic                  b_rready
);

    // ================================================================
    // Path A — scalar dma_engine → banked_scratchpad (byte writes)
    // ================================================================
    logic                  a_desc_valid, a_desc_ready, a_dma_done;
    logic                  a_w_we;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [1:0]            a_w_dst;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [11:0]           a_w_addr;
    logic [7:0]            a_w_data;

    dma_engine #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)) u_dma_a (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(a_desc_valid), .desc_ready(a_desc_ready),
        .desc_addr(dram_base), .desc_dst_addr(12'd0),
        .desc_len_bytes(tile_bytes), .desc_dst(2'd1), .done(a_dma_done),
        .m_araddr(a_araddr), .m_arlen(a_arlen), .m_arsize(a_arsize),
        .m_arburst(a_arburst), .m_arvalid(a_arvalid), .m_arready(a_arready),
        .m_rdata(a_rdata), .m_rresp(a_rresp), .m_rlast(a_rlast),
        .m_rvalid(a_rvalid), .m_rready(a_rready),
        .w_we(a_w_we), .w_dst(a_w_dst), .w_addr(a_w_addr), .w_data(a_w_data)
    );

    /* verilator lint_off PINCONNECTEMPTY */
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SP_DEPTH)) u_sp_a (
        .clk(clk),
        .w_en(a_w_we), .w_vec(1'b0), .w_addr(SP_ADDR_W'(a_w_addr)),
        .w_sdata(a_w_data), .w_vdata({VEC_W{1'b0}}),
        .r_en(1'b1), .r_vec(1'b0), .r_addr(r_a_addr),
        .r_sdata(r_a_sdata), .r_vdata()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    assign o_a_we = a_w_we;

    // ================================================================
    // Path B — vector dma_engine_vec → banked_scratchpad (stripe writes)
    // ================================================================
    logic                  b_desc_valid, b_desc_ready, b_dma_done;
    logic                  b_w_en, b_w_vec;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [1:0]            b_w_dst;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [11:0]           b_w_addr;
    logic [VEC_W-1:0]      b_w_vdata;

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_dma_error; // Legacy wrapper: success-only interface, reset on failure.
    /* verilator lint_on UNUSEDSIGNAL */
    dma_engine_vec #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .NUM_BANKS(NUM_BANKS)) u_dma_b (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(b_desc_valid), .desc_ready(b_desc_ready),
        .desc_addr(dram_base), .desc_dst_addr(12'd0),
        .desc_len_bytes(tile_bytes), .desc_dst(2'd1), .done(b_dma_done), .error(unused_dma_error),
        .m_araddr(b_araddr), .m_arlen(b_arlen), .m_arsize(b_arsize),
        .m_arburst(b_arburst), .m_arvalid(b_arvalid), .m_arready(b_arready),
        .m_rdata(b_rdata), .m_rresp(b_rresp), .m_rlast(b_rlast),
        .m_rvalid(b_rvalid), .m_rready(b_rready),
        .w_en(b_w_en), .w_vec(b_w_vec), .w_dst(b_w_dst),
        .w_addr(b_w_addr), .w_vdata(b_w_vdata)
    );

    /* verilator lint_off PINCONNECTEMPTY */
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SP_DEPTH)) u_sp_b (
        .clk(clk),
        .w_en(b_w_en), .w_vec(b_w_vec), .w_addr(SP_ADDR_W'(b_w_addr)),
        .w_sdata(8'b0), .w_vdata(b_w_vdata),
        .r_en(1'b1), .r_vec(1'b0), .r_addr(r_b_addr),
        .r_sdata(r_b_sdata), .r_vdata()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    assign o_b_we = b_w_en;

    // ================================================================
    // Two independent load FSMs (one descriptor each on start)
    // ================================================================
    typedef enum logic [1:0] { LD_IDLE, LD_ISSUE, LD_WAIT, LD_DONE } ld_t;
    ld_t a_state, b_state;

    assign a_desc_valid = (a_state == LD_ISSUE);
    assign done_a       = (a_state == LD_DONE);
    assign b_desc_valid = (b_state == LD_ISSUE);
    assign done_b       = (b_state == LD_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_state <= LD_IDLE;
            b_state <= LD_IDLE;
        end else begin
            case (a_state)
                LD_IDLE:  if (start_a)       a_state <= LD_ISSUE;
                LD_ISSUE: if (a_desc_ready)  a_state <= LD_WAIT;
                LD_WAIT:  if (a_dma_done)    a_state <= LD_DONE;
                LD_DONE:  ;
                default:  a_state <= LD_IDLE;
            endcase
            case (b_state)
                LD_IDLE:  if (start_b)       b_state <= LD_ISSUE;
                LD_ISSUE: if (b_desc_ready)  b_state <= LD_WAIT;
                LD_WAIT:  if (b_dma_done)    b_state <= LD_DONE;
                LD_DONE:  ;
                default:  b_state <= LD_IDLE;
            endcase
        end
    end

endmodule
