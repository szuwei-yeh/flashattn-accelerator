// ============================================================
//  addr_gen.sv — Address Generator for SRAM and Main Memory
//
//  Responsibilities:
//  1. Compute the Global Offset for fetching data from external AXI memory.
//  2. Generate the sequential read/write addresses for internal SRAMs.
//
//  Fix (v2): All 16-bit × 16-bit multiplies are now cast to 32-bit before
//  multiplication to prevent silent truncation (e.g. 1024 × 64 = 65536
//  overflows 16-bit).  Verilator -Wall no longer raises WIDTHEXPAND here.
// ============================================================
`timescale 1ns/1ps

module addr_gen #(
    parameter int MAX_SRAM_DEPTH = 4096, // Supports max 64x64 tile size
    localparam int ADDR_W = $clog2(MAX_SRAM_DEPTH)
)(
    input  logic              clk,
    input  logic              rst_n,

    // === Software Configuration Parameters ===
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [15:0]       seq_len,   // Sequence length N (e.g., 256)
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [15:0]       head_dim,  // Head dimension d (e.g., 64)
    input  logic [15:0]       tile_size, // Tile size (e.g., 16 or 64)

    // === Tile Coordinates (Driven by Tile Controller FSM) ===
    input  logic [15:0]       tile_row,  // Outer loop j (Q tile start index)
    input  logic [15:0]       tile_col,  // Inner loop i (K/V tile start index)

    // === Output 1: Global Address Offsets (For AXI Memory Fetch) ===
    output logic [31:0]       q_global_offset,
    output logic [31:0]       k_global_offset,
    output logic [31:0]       v_global_offset,

    // === Output 2: Local SRAM Sequential Address Counter ===
    input  logic              cnt_en,    // Enable SRAM read/write address counting
    input  logic              cnt_clr,   // Clear/reset the counter
    output logic [ADDR_W-1:0] sram_addr, // Generated internal SRAM address (0 to total_elements-1)
    output logic              cnt_done   // Flag indicating the current tile has finished counting
);

    // --------------------------------------------------------
    // 1. Global Offset Calculation (Combinational Logic)
    //
    // FIX: cast operands to 32-bit BEFORE multiplying.
    // Without the cast, SV computes 16-bit × 16-bit = 16-bit (truncated)
    // and then zero-extends to 32-bit — silently wrong for large inputs.
    // --------------------------------------------------------
    assign q_global_offset = 32'(tile_row)  * 32'(head_dim);
    assign k_global_offset = 32'(tile_col)  * 32'(head_dim);
    assign v_global_offset = 32'(tile_col)  * 32'(head_dim);

    // --------------------------------------------------------
    // 2. SRAM Sequential Address Counter (Sequential Logic)
    // Counts from 0 to (tile_size * head_dim) - 1
    //
    // FIX: same 32-bit cast applied to total_elements.
    // --------------------------------------------------------
    logic [31:0] total_elements;
    assign total_elements = 32'(tile_size) * 32'(head_dim);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sram_addr <= '0;
        end else if (cnt_clr) begin
            sram_addr <= '0;
        end else if (cnt_en) begin
            if (32'(sram_addr) < total_elements - 32'd1) begin
                sram_addr <= sram_addr + 1'b1;
            end
        end
    end

    // Flag when the counter reaches the last element and counting is enabled
    assign cnt_done = (32'(sram_addr) == total_elements - 32'd1) && cnt_en;

endmodule
