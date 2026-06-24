/// sta-blackbox
// ============================================================
//  output_buffer.sv — Accumulation / Rescale / Normalise Buffer
//
//  Three operating modes (mutually exclusive):
//    accum_en  : new = old + data_in
//    rescale_en: new = (old * rescale_q88) >> 8   [Q8.8 multiply]
//    norm_en   : new = (old << 8) / norm_divisor  [integer divide]
//
//  All modes share the 1-cycle SRAM read-latency pipeline.
//  External read (re_ext) takes priority on the read port.
// ============================================================
`timescale 1ns/1ps

module output_buffer #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_WIDTH = 32,
    /* verilator lint_on UNUSEDPARAM */
    parameter DEPTH      = 4096,
    localparam ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // Accumulate: new = old + data_in
    input  logic                  accum_en,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic signed [31:0]    data_in,

    // Rescale: new = (old * rescale_q88) >> 8
    input  logic                  rescale_en,
    input  logic [ADDR_WIDTH-1:0] rescale_addr,
    input  logic [15:0]           rescale_q88,

    // Normalise: new = (old << 8) / norm_divisor
    input  logic                  norm_en,
    input  logic [ADDR_WIDTH-1:0] norm_addr,
    input  logic [31:0]           norm_divisor,

    // External read (1-cycle latency)
    input  logic                  re_ext,
    input  logic [ADDR_WIDTH-1:0] raddr_ext,
    output logic signed [31:0]    rdata_ext
);

    // ── Pipeline stage: delay write-side signals 1 cycle ─────
    logic                  accum_en_d,    rescale_en_d,    norm_en_d;
    logic [ADDR_WIDTH-1:0] accum_addr_d,  rescale_addr_d,  norm_addr_d;
    logic signed [31:0]    data_in_d;
    logic [15:0]           rescale_q88_d;
    logic [31:0]           norm_divisor_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_en_d     <= 1'b0;
            accum_addr_d   <= '0;
            data_in_d      <= '0;
            rescale_en_d   <= 1'b0;
            rescale_addr_d <= '0;
            rescale_q88_d  <= 16'h0100;
            norm_en_d      <= 1'b0;
            norm_addr_d    <= '0;
            norm_divisor_d <= 32'd1;
        end else begin
            accum_en_d     <= accum_en;
            accum_addr_d   <= addr;
            data_in_d      <= data_in;
            rescale_en_d   <= rescale_en;
            rescale_addr_d <= rescale_addr;
            rescale_q88_d  <= rescale_q88;
            norm_en_d      <= norm_en;
            norm_addr_d    <= norm_addr;
            norm_divisor_d <= norm_divisor;
        end
    end

    // ── SRAM read-address mux ─────────────────────────────────
    logic [ADDR_WIDTH-1:0] raddr_mux;
    always_comb begin
        if      (re_ext)     raddr_mux = raddr_ext;
        else if (rescale_en) raddr_mux = rescale_addr;
        else if (norm_en)    raddr_mux = norm_addr;
        else                 raddr_mux = addr;
    end

    // ── SRAM ──────────────────────────────────────────────────
    logic signed [31:0] old_data;

    // ── Compute new write value ───────────────────────────────
    // Rescale: 48-bit signed multiply then >>8
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [47:0] rescale_wide;
    /* verilator lint_on UNUSEDSIGNAL */
    assign rescale_wide = $signed({{16{old_data[31]}}, old_data})
                        * $signed({32'b0, rescale_q88_d});

    // Normalise: (old<<8) / norm_divisor  (signed integer divide)
    // Both operands promoted to 40 bits; result truncated to 32 bits.
    logic signed [39:0] norm_numer;
    assign norm_numer = {old_data, 8'b0};   // 40-bit = old_data << 8

    logic signed [31:0] new_val;
    always_comb begin
        if (rescale_en_d)
            new_val = $signed(rescale_wide[39:8]);           // >>8
        else if (norm_en_d)
            new_val = (norm_divisor_d != 32'd0)
                      ? 32'($signed(norm_numer) / $signed({8'b0, norm_divisor_d}))
                      : old_data;
        else
            new_val = old_data + data_in_d;                  // accum
    end

    // ── SRAM write enables and address ───────────────────────
    logic                  we;
    logic [ADDR_WIDTH-1:0] waddr;

    assign we    = accum_en_d | rescale_en_d | norm_en_d;
    assign waddr = rescale_en_d ? rescale_addr_d :
                   norm_en_d    ? norm_addr_d     :
                                  accum_addr_d;

    sram_1r1w #(
        .DATA_WIDTH(32),
        .DEPTH     (DEPTH)
    ) u_sram_o (
        .clk   (clk),
        .we    (we),
        .waddr (waddr),
        .wdata (new_val),
        .re    (1'b1),
        .raddr (raddr_mux),
        .rdata (old_data)
    );

    assign rdata_ext = old_data;

    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (accum_en && re_ext)
            $error("output_buffer: accum_en and re_ext conflict!");
        if ((int'(accum_en) + int'(rescale_en) + int'(norm_en)) > 1)
            $error("output_buffer: multiple write modes active!");
    end
    // synthesis translate_on

endmodule
