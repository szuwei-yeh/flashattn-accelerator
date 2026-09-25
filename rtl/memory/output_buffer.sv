// ============================================================
//  output_buffer.sv — Accumulation / Rescale / Normalise Buffer
//
//  Operating modes:
//    accum_en                    : new = old + data_in
//    rescale_en                  : new = (old * rescale_q88) >> 8
//    accum_en && rescale_en      : new = ((old * rescale_q88) >> 8) + data_in
//    norm_en                     : new = (old << 8) / norm_divisor
//
//  The fused mode deliberately truncates the rescale result to the same 32-bit
//  intermediate value that the standalone rescale pass writes to SRAM before
//  performing the existing 32-bit accumulation.  This preserves the original
//  two-pass numerical semantics exactly.
//
//  All modes share the 1-cycle SRAM read-latency pipeline.
//  External read (re_ext) takes priority on the read port.
// ============================================================
`timescale 1ns/1ps

module output_buffer #(
    /* verilator lint_off UNUSEDPARAM */
    parameter int DATA_WIDTH = 32,
    /* verilator lint_on UNUSEDPARAM */
    parameter int DEPTH      = 4096,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                  clk,
    input  logic                  rst_n,
    // First K/V tile owns a fresh output row. Ignore prior SRAM contents on
    // updates (including X power-up contents); normalization still reads SRAM.
    input  logic                  first_tile,

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
    logic                  first_tile_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            first_tile_d   <= 1'b0;
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
            first_tile_d   <= first_tile;
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

    logic signed [31:0] rescaled_val;
    logic signed [31:0] new_val;
    assign rescaled_val = $signed(rescale_wide[39:8]);

    always_comb begin
        if (rescale_en_d && accum_en_d)
            new_val = first_tile_d ? data_in_d : rescaled_val + data_in_d;
        else if (rescale_en_d)
            new_val = first_tile_d ? 32'sd0 : rescaled_val;
        else if (norm_en_d)
            new_val = (norm_divisor_d != 32'd0)
                      ? 32'($signed(norm_numer) / $signed({8'b0, norm_divisor_d}))
                      : old_data;
        else
            new_val = first_tile_d ? data_in_d : old_data + data_in_d;
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
        if (norm_en && (accum_en || rescale_en))
            $error("output_buffer: norm_en conflicts with update mode!");
        if (accum_en && rescale_en && (addr != rescale_addr))
            $error("output_buffer: fused update addresses must match!");
    end
    // synthesis translate_on

endmodule
