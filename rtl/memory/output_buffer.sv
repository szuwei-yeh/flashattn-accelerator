// ============================================================
//  output_buffer.sv — Accumulation Buffer for Output O
//
//  Data type : INT32 (handles matmul accumulation)
//  Capacity  : up to 64×64 × 4 B = 16 KB
//
//  Fix (v2) — two bugs corrected:
//
//  Bug 1 — Read-before-write hazard:
//    sram_1r1w has 1-cycle registered read latency.  The original code
//    applied accum_en and addr in the same cycle as the read, so
//    new_sum = (stale rdata) + data_in was written — always wrong.
//    Fix: pipeline accum_en / addr / data_in by 1 cycle so the write
//    happens exactly when old_data is valid.
//
//  Bug 2 — Port conflict between accumulation read and external read:
//    The single SRAM read port is shared between the accumulation path
//    (raddr = addr) and the external read path (raddr = raddr_ext).
//    Asserting both simultaneously corrupts the accumulation base value.
//    Fix: external read is prioritised in the mux; a simulation-only
//    assertion fires if both are active at the same time.
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

    // --- Accumulation port (write-side) ---
    input  logic                  accum_en,           // Pulse: accumulate data_in at addr
    input  logic [ADDR_WIDTH-1:0] addr,               // Accumulation address
    input  logic signed [31:0]    data_in,            // Value to add

    // --- External read port ---
    input  logic                  re_ext,             // External read enable
    input  logic [ADDR_WIDTH-1:0] raddr_ext,          // External read address
    output logic signed [31:0]    rdata_ext           // External read data (1-cycle latency)
);

    // --------------------------------------------------------
    // Pipeline stage: delay accum_en / addr / data_in by 1 cycle
    // so that the write arrives when old_data (SRAM read output) is valid.
    // --------------------------------------------------------
    logic                  accum_en_d;
    logic [ADDR_WIDTH-1:0] addr_d;
    logic signed [31:0]    data_in_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_en_d <= 1'b0;
            addr_d     <= '0;
            data_in_d  <= '0;
        end else begin
            accum_en_d <= accum_en;
            addr_d     <= addr;
            data_in_d  <= data_in;
        end
    end

    // --------------------------------------------------------
    // SRAM read address mux:
    //   - External read takes priority (re_ext overrides).
    //   - When accumulating, point read at the accumulation address
    //     so old_data is ready 1 cycle later for the write.
    // --------------------------------------------------------
    logic [ADDR_WIDTH-1:0] raddr_mux;
    assign raddr_mux = re_ext ? raddr_ext : addr;

    // --------------------------------------------------------
    // Accumulation: new value = old SRAM content + new data_in
    // Both sides are now aligned: old_data reflects addr_d (cycle-1 addr)
    // and data_in_d is the correspondingly delayed input.
    // --------------------------------------------------------
    logic signed [31:0] old_data;
    logic signed [31:0] new_sum;
    assign new_sum = old_data + data_in_d;

    sram_1r1w #(
        .DATA_WIDTH(32),
        .DEPTH     (DEPTH)
    ) u_sram_o (
        .clk   (clk),
        // Write: delayed by 1 cycle relative to accum_en input
        .we    (accum_en_d),
        .waddr (addr_d),
        .wdata (new_sum),
        // Read: always enabled; address shared between accum and external paths
        .re    (1'b1),
        .raddr (raddr_mux),
        .rdata (old_data)
    );

    // External read data is the raw SRAM output (1-cycle latency from re_ext)
    assign rdata_ext = old_data;

    // --------------------------------------------------------
    // Simulation-only assertion: catch simultaneous accum + external read.
    // When re_ext=1 the read port points at raddr_ext, so old_data no longer
    // reflects the accumulation address — new_sum_d would be corrupted.
    // --------------------------------------------------------
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (accum_en && re_ext)
            $error("output_buffer: accum_en and re_ext asserted simultaneously — port conflict!");
    end
    // synthesis translate_on

endmodule
