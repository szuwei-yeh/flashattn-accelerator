// ============================================================
//  flash_attn_top_dma.sv — FlashAttention single-head top with DMA front-end
//
//  Adds the top two boxes of the memory-hierarchy diagram:
//      DDR/HBM (AXI4 slave, external) → DMA (AXI4 master, here) → scratchpad
//  The DMA *pulls* Q and then K/V tiles from external memory into the core's
//  Q/K/V scratchpad SRAMs.  A small scheduler streams KV tiles in order and
//  raises `kv_tiles_ready`; the core stalls on the residency gate only for the
//  first tile, so subsequent DRAM latency is hidden behind systolic compute.
//
//  DRAM image layout (byte addresses), all matrices SEQ_LEN×HEAD_DIM, INT8:
//      Q : [Q_BASE .. Q_BASE+MAT_BYTES)
//      K : [K_BASE .. K_BASE+MAT_BYTES)
//      V : [V_BASE .. V_BASE+MAT_BYTES)
//
//  AXI4 read-master ports are exposed at the boundary (synthesizable); the
//  external DRAM (axi_mem_model) is connected in simulation by the harness.
// ============================================================
`timescale 1ns/1ps

module flash_attn_top_dma #(
    parameter int SEQ_LEN    = 64,
    parameter int HEAD_DIM   = 16,
    parameter int TILE_SIZE  = 16,
    parameter int SRAM_DEPTH = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    localparam int MAT_BYTES  = SEQ_LEN * HEAD_DIM,
    localparam int TILE_BYTES = TILE_SIZE * HEAD_DIM,
    localparam int NUM_TILES  = SEQ_LEN / TILE_SIZE,
    localparam int Q_BASE     = 0,
    localparam int K_BASE     = MAT_BYTES,
    localparam int V_BASE     = 2 * MAT_BYTES
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic        causal,
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // Output buffer read port (drained after done)
    input  logic [11:0]        out_raddr,
    output logic signed [31:0] out_rdata,

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
    output logic                  m_rready,

    // Observability
    output logic [15:0] dbg_kv_tiles_ready
);

    // ====================================================================
    // DMA engine (AXI4 read master)
    // ====================================================================
    logic                  desc_valid, desc_ready, dma_done;
    logic [AXI_ADDR_W-1:0] desc_addr;
    logic [11:0]           desc_dst_addr;
    logic [31:0]           desc_len_bytes;
    logic [1:0]            desc_dst;

    logic        w_we;
    logic [1:0]  w_dst;
    logic [11:0] w_addr;
    logic [7:0]  w_data;

    dma_engine #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) u_dma (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_dst_addr(desc_dst_addr),
        .desc_len_bytes(desc_len_bytes), .desc_dst(desc_dst), .done(dma_done),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize),
        .m_arburst(m_arburst), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_rvalid(m_rvalid), .m_rready(m_rready),
        .w_we(w_we), .w_dst(w_dst), .w_addr(w_addr), .w_data(w_data)
    );

    // ====================================================================
    // DMA scheduler — Q bulk load, then K/V tiles in order
    // ====================================================================
    typedef enum logic [2:0] {
        SCH_IDLE, SCH_Q_ISSUE, SCH_Q_WAIT,
        SCH_K_ISSUE, SCH_K_WAIT, SCH_V_ISSUE, SCH_V_WAIT, SCH_DONE
    } sch_t;
    sch_t sch_state;

    logic [15:0] tile_idx;        // KV tile being fetched
    logic [15:0] kv_tiles_ready;  // monotonic count of resident KV tiles
    logic        core_start;

    logic [31:0] tile_off;
    assign tile_off = 32'(tile_idx) * 32'(TILE_BYTES);

    assign dbg_kv_tiles_ready = kv_tiles_ready;

    // ── Descriptor outputs: combinational from state (valid/ready handshake) ──
    always_comb begin
        desc_valid     = (sch_state == SCH_Q_ISSUE)
                      || (sch_state == SCH_K_ISSUE)
                      || (sch_state == SCH_V_ISSUE);
        // defaults (Q load)
        desc_addr      = AXI_ADDR_W'(Q_BASE);
        desc_dst_addr  = 12'd0;
        desc_len_bytes = 32'(MAT_BYTES);
        desc_dst       = 2'd0;
        unique case (sch_state)
            SCH_K_ISSUE: begin
                desc_addr      = AXI_ADDR_W'(K_BASE) + AXI_ADDR_W'(tile_off);
                desc_dst_addr  = tile_off[11:0];
                desc_len_bytes = 32'(TILE_BYTES);
                desc_dst       = 2'd1;
            end
            SCH_V_ISSUE: begin
                desc_addr      = AXI_ADDR_W'(V_BASE) + AXI_ADDR_W'(tile_off);
                desc_dst_addr  = tile_off[11:0];
                desc_len_bytes = 32'(TILE_BYTES);
                desc_dst       = 2'd2;
            end
            default: ; // Q load defaults above
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sch_state      <= SCH_IDLE;
            tile_idx       <= '0;
            kv_tiles_ready <= '0;
            core_start     <= 1'b0;
        end else begin
            core_start <= 1'b0;   // 1-cycle pulse

            case (sch_state)
                SCH_IDLE: begin
                    tile_idx       <= '0;
                    kv_tiles_ready <= '0;
                    if (start) sch_state <= SCH_Q_ISSUE;
                end

                // Bulk-load the whole Q matrix into the Q scratchpad
                SCH_Q_ISSUE: if (desc_ready) sch_state <= SCH_Q_WAIT;
                SCH_Q_WAIT:  if (dma_done)   sch_state <= SCH_K_ISSUE;

                // Stream K tile `tile_idx`
                SCH_K_ISSUE: if (desc_ready) sch_state <= SCH_K_WAIT;
                SCH_K_WAIT:  if (dma_done)   sch_state <= SCH_V_ISSUE;

                // Stream V tile `tile_idx`
                SCH_V_ISSUE: if (desc_ready) sch_state <= SCH_V_WAIT;
                SCH_V_WAIT: begin
                    if (dma_done) begin
                        // tile (K,V) pair is now fully resident
                        kv_tiles_ready <= kv_tiles_ready + 16'd1;
                        if (tile_idx == 16'd0)
                            core_start <= 1'b1;        // kick off compute
                        if (int'(tile_idx) == NUM_TILES - 1) begin
                            sch_state <= SCH_DONE;
                        end else begin
                            tile_idx  <= tile_idx + 16'd1;
                            sch_state <= SCH_K_ISSUE;
                        end
                    end
                end

                SCH_DONE: ;   // all tiles streamed; core finishes from scratchpad

                default: sch_state <= SCH_IDLE;
            endcase
        end
    end

    // ====================================================================
    // Per-destination scratchpad write enables
    // ====================================================================
    logic q_we_c, k_we_c, v_we_c;
    assign q_we_c = w_we & (w_dst == 2'd0);
    assign k_we_c = w_we & (w_dst == 2'd1);
    assign v_we_c = w_we & (w_dst == 2'd2);

    // ====================================================================
    // FlashAttention core (compute); DMA fills its Q/K/V scratchpads
    // ====================================================================
    /* verilator lint_off PINCONNECTEMPTY */
    flash_attn_core #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM),
        .SEQ_LEN(SEQ_LEN), .SRAM_DEPTH(SRAM_DEPTH)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .done(done),
        .mode(1'b0), .kv_len(16'b0),
        .scale_q(scale_q), .scale_k(scale_k), .scale_v(scale_v),
        .q_we(q_we_c), .q_waddr(w_addr), .q_wdata(w_data),
        .k_we(k_we_c), .k_waddr(w_addr), .k_wdata(w_data),
        .v_we(v_we_c), .v_waddr(w_addr), .v_wdata(w_data),
        .out_raddr(out_raddr), .out_rdata(out_rdata),
        .causal(causal),
        .kv_tiles_ready(kv_tiles_ready),
        // KV cache unused in DMA prefill top
        .kc_write_en(1'b0), .kc_write_ptr(8'b0),
        .kc_k_flat('0), .kc_v_flat('0), .kc_read_addr(8'b0),
        .kc_k_out(), .kc_v_out(), .kc_cache_len()
    );
    /* verilator lint_on PINCONNECTEMPTY */

endmodule
