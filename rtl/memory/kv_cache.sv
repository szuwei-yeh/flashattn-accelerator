// ============================================================
//  kv_cache.sv — KV Cache for Autoregressive Inference
//
//  Stores K and V vectors indexed by token position.
//  Uses two sram_1r1w instances (one for K, one for V).
//
//  Write semantics: append-only.
//  cache_len increments only when write_ptr == cache_len
//  (i.e., sequential append starting from 0).
//
//  Read latency: 1 cycle (registered SRAM output).
// ============================================================
`timescale 1ns/1ps

module kv_cache #(
    parameter int MAX_SEQ_LEN = 256,
    parameter int HEAD_DIM    = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Write port: append new K/V at write_ptr
    input  logic                    write_en,
    input  logic [7:0]              write_ptr,
    input  logic [HEAD_DIM*8-1:0]   k_new_flat,   // HEAD_DIM INT8 packed LSB-first
    input  logic [HEAD_DIM*8-1:0]   v_new_flat,

    // Read port: 1-cycle registered latency
    input  logic [7:0]              read_addr,
    output logic [HEAD_DIM*8-1:0]   k_out_flat,
    output logic [HEAD_DIM*8-1:0]   v_out_flat,

    // Status
    output logic [8:0]              cache_len     // number of valid tokens stored
);

    // ---- K cache SRAM ----------------------------------------
    sram_1r1w #(
        .DATA_WIDTH(HEAD_DIM * 8),
        .DEPTH     (MAX_SEQ_LEN)
    ) u_k_sram (
        .clk  (clk),
        .we   (write_en),
        .waddr(write_ptr),
        .wdata(k_new_flat),
        .re   (1'b1),
        .raddr(read_addr),
        .rdata(k_out_flat)
    );

    // ---- V cache SRAM ----------------------------------------
    sram_1r1w #(
        .DATA_WIDTH(HEAD_DIM * 8),
        .DEPTH     (MAX_SEQ_LEN)
    ) u_v_sram (
        .clk  (clk),
        .we   (write_en),
        .waddr(write_ptr),
        .wdata(v_new_flat),
        .re   (1'b1),
        .raddr(read_addr),
        .rdata(v_out_flat)
    );

    // ---- cache_len counter -----------------------------------
    // Increments only on sequential append: write_ptr == cache_len
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cache_len <= 9'd0;
        end else if (write_en &&
                     (write_ptr == cache_len[7:0]) &&
                     (cache_len < 9'(MAX_SEQ_LEN))) begin
            cache_len <= cache_len + 9'd1;
        end
    end

endmodule
