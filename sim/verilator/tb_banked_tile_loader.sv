// ============================================================
//  tb_banked_tile_loader.sv — harness: banked_tile_loader + 3 banked_scratchpads.
//  Testbench fills Q/K/V scratchpads via a scalar fill port, then triggers the
//  loader and observes the tile-register write stream (wr_en/wr_index/*_stripe).
// ============================================================
`timescale 1ns/1ps

module tb_banked_tile_loader #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 16,
    parameter int NUM_BANKS = 16,
    parameter int SP_DEPTH  = 4096,
    localparam int SP_ADDR_W = $clog2(SP_DEPTH),
    localparam int VEC_W     = NUM_BANKS * 8,
    localparam int TILE_ELEMS = TILE_SIZE * HEAD_DIM,
    localparam int IDX_W      = $clog2(TILE_ELEMS)
)(
    input  logic clk,
    input  logic rst_n,

    // Scalar fill port (testbench preloads the scratchpads)
    input  logic                 f_we,
    input  logic [1:0]           f_sel,    // 0=Q, 1=K, 2=V
    input  logic [SP_ADDR_W-1:0] f_addr,
    input  logic [7:0]           f_data,

    // Loader control
    input  logic                 start,
    input  logic                 mode,     // 0=Q, 1=KV
    input  logic [SP_ADDR_W-1:0] base_addr,
    output logic                 busy,
    output logic                 done,

    // Tile-register write stream (observed by C++)
    output logic             wr_en,
    output logic [IDX_W-1:0] wr_index,
    output logic [VEC_W-1:0] q_stripe,
    output logic [VEC_W-1:0] k_stripe,
    output logic [VEC_W-1:0] v_stripe
);
    // Loader ↔ scratchpad read wiring
    logic                 q_r_en, k_r_en, v_r_en;
    logic [SP_ADDR_W-1:0] q_r_addr, k_r_addr, v_r_addr;
    logic [VEC_W-1:0]     q_r_vdata, k_r_vdata, v_r_vdata;

    banked_tile_loader #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM),
        .NUM_BANKS(NUM_BANKS), .SP_ADDR_W(SP_ADDR_W)
    ) u_loader (
        .clk(clk), .rst_n(rst_n),
        .start(start), .mode(mode), .base_addr(base_addr),
        .q_r_en(q_r_en), .q_r_addr(q_r_addr), .q_r_vdata(q_r_vdata),
        .k_r_en(k_r_en), .k_r_addr(k_r_addr), .k_r_vdata(k_r_vdata),
        .v_r_en(v_r_en), .v_r_addr(v_r_addr), .v_r_vdata(v_r_vdata),
        .wr_en(wr_en), .wr_index(wr_index),
        .q_stripe(q_stripe), .k_stripe(k_stripe), .v_stripe(v_stripe),
        .busy(busy), .done(done)
    );

    // Per-scratchpad scalar write enables (fill path)
    logic q_we, k_we, v_we;
    assign q_we = f_we & (f_sel == 2'd0);
    assign k_we = f_we & (f_sel == 2'd1);
    assign v_we = f_we & (f_sel == 2'd2);

    /* verilator lint_off PINCONNECTEMPTY */
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SP_DEPTH)) u_sp_q (
        .clk(clk),
        .w_en(q_we), .w_vec(1'b0), .w_addr(f_addr), .w_sdata(f_data), .w_vdata({VEC_W{1'b0}}),
        .r_en(q_r_en), .r_vec(1'b1), .r_addr(q_r_addr), .r_sdata(), .r_vdata(q_r_vdata)
    );
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SP_DEPTH)) u_sp_k (
        .clk(clk),
        .w_en(k_we), .w_vec(1'b0), .w_addr(f_addr), .w_sdata(f_data), .w_vdata({VEC_W{1'b0}}),
        .r_en(k_r_en), .r_vec(1'b1), .r_addr(k_r_addr), .r_sdata(), .r_vdata(k_r_vdata)
    );
    banked_scratchpad #(.NUM_BANKS(NUM_BANKS), .DATA_WIDTH(8), .DEPTH(SP_DEPTH)) u_sp_v (
        .clk(clk),
        .w_en(v_we), .w_vec(1'b0), .w_addr(f_addr), .w_sdata(f_data), .w_vdata({VEC_W{1'b0}}),
        .r_en(v_r_en), .r_vec(1'b1), .r_addr(v_r_addr), .r_sdata(), .r_vdata(v_r_vdata)
    );
    /* verilator lint_on PINCONNECTEMPTY */

endmodule
