// ============================================================
//  flash_attn_top_dma_banked_wb.sv — DMA-fed banked top WITH output write-back
//                                     (Stage WB-3, full DRAM round-trip)
//
//  Extends flash_attn_top_dma_banked.sv with the WB-1/WB-2 write path so the
//  complete datapath is proven end to end:
//
//    DRAM ─(AR/R)→ dma_engine_vec ─→ banked scratchpads ─→ flash_attn_core_banked
//         ─(compute)→ output_buffer ─→ output_writeback_packer ─→ dma_write_engine
//         ─(AW/W/B)→ DRAM
//
//  Read side is identical to flash_attn_top_dma_banked (same open-loop Q-then-KV
//  scheduler, same residency gate, same read-side performance counters).  After
//  the core finishes, a small write-back FSM starts the packer and issues one
//  write descriptor (O @ cfg_o_base, SEQ_LEN*HEAD_DIM int32 words); top-level
//  `done` asserts only after the write DMA's final B response.
//
//  Additive: does not modify flash_attn_top_dma_banked.sv, flash_attn_core_banked.sv,
//  output_buffer.sv, dma_engine_vec.sv, dma_write_engine.sv, or
//  output_writeback_packer.sv.  Reuses all of them unchanged.
// ============================================================
`timescale 1ns/1ps

module flash_attn_top_dma_banked_wb #(
    parameter int SEQ_LEN    = 64,
    parameter int HEAD_DIM   = 16,
    parameter int TILE_SIZE  = 16,
    parameter int SRAM_DEPTH = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    localparam int TILE_BYTES = TILE_SIZE * HEAD_DIM,
    localparam int LOG2_TILE  = $clog2(TILE_SIZE),
    localparam int VEC_W      = 16 * 8,
    localparam int O_WORDS    = SEQ_LEN * HEAD_DIM,   // output words (int32 each)
    localparam int O_BYTES    = O_WORDS * 4,
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic        causal,
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // ── Runtime scheduler config (sampled at accepted start) ──────────
    input  logic [15:0]           cfg_seq_len,
    input  logic [31:0]           cfg_q_base,
    input  logic [31:0]           cfg_k_base,
    input  logic [31:0]           cfg_v_base,
    input  logic [31:0]           cfg_o_base,   // output write-back DRAM base
    output logic                  cfg_error,

    // ── AXI4 read-master ports (to external DRAM) ─────────────────────
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

    // ── AXI4 write-master ports (to external DRAM) ────────────────────
    output logic [AXI_ADDR_W-1:0]     m_awaddr,
    output logic [7:0]                m_awlen,
    output logic [2:0]                m_awsize,
    output logic [1:0]                m_awburst,
    output logic                      m_awvalid,
    input  logic                      m_awready,
    output logic [AXI_DATA_W-1:0]     m_wdata,
    output logic [BYTES_PER_BEAT-1:0] m_wstrb,
    output logic                      m_wlast,
    output logic                      m_wvalid,
    input  logic                      m_wready,
    input  logic [1:0]                m_bresp,
    input  logic                      m_bvalid,
    output logic                      m_bready,

    // ── Read-side performance counters (preserved) ────────────────────
    output logic [31:0] perf_total_cycles,
    output logic [31:0] perf_dma_busy_cycles,
    output logic [31:0] perf_core_busy_cycles,
    output logic [31:0] perf_dma_bytes,
    output logic [15:0] perf_kv_tiles_loaded,
    output logic [31:0] perf_first_tile_wait_cycles,

    // ── Write-back performance counters (new) ─────────────────────────
    output logic [31:0] perf_wb_bytes,
    output logic [31:0] perf_wb_cycles,
    output logic [31:0] perf_wb_beats,

    output logic [15:0] dbg_kv_tiles_ready
);

    // ── Config validation (combinational, from live inputs) ──────────
    assign cfg_error = (cfg_seq_len == 16'd0)
                    || (cfg_seq_len > 16'(SEQ_LEN))
                    || ((cfg_seq_len & 16'(TILE_SIZE - 1)) != 16'd0);

    // ── Read DMA engine (vector) ──────────────────────────────────────
    logic                  desc_valid, desc_ready, dma_done;
    logic [AXI_ADDR_W-1:0] desc_addr;
    logic [11:0]           desc_dst_addr;
    logic [31:0]           desc_len_bytes;
    logic [1:0]            desc_dst;

    logic              w_en, w_vec;
    logic [1:0]        w_dst;
    logic [11:0]       w_addr;
    logic [VEC_W-1:0]  w_vdata;

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_dma_error; // Legacy wrapper: success-only interface, reset on failure.
    /* verilator lint_on UNUSEDSIGNAL */
    dma_engine_vec #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .NUM_BANKS(16)
    ) u_dma (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_dst_addr(desc_dst_addr),
        .desc_len_bytes(desc_len_bytes), .desc_dst(desc_dst), .done(dma_done), .error(unused_dma_error),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize),
        .m_arburst(m_arburst), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_rvalid(m_rvalid), .m_rready(m_rready),
        .w_en(w_en), .w_vec(w_vec), .w_dst(w_dst), .w_addr(w_addr), .w_vdata(w_vdata)
    );

    // ── Read scheduler (identical to flash_attn_top_dma_banked) ───────
    typedef enum logic [2:0] {
        SCH_IDLE, SCH_Q_ISSUE, SCH_Q_WAIT,
        SCH_K_ISSUE, SCH_K_WAIT, SCH_V_ISSUE, SCH_V_WAIT, SCH_DONE
    } sch_t;
    sch_t sch_state;

    logic [15:0] tile_idx;
    logic [15:0] kv_tiles_ready;
    logic        core_start;

    logic [31:0] r_q_base, r_k_base, r_v_base;
    logic [15:0] r_num_tiles;
    logic [31:0] r_q_len_bytes;

    logic start_accepted;
    assign start_accepted = (sch_state == SCH_IDLE) && start && !cfg_error;

    logic [31:0] tile_off;
    assign tile_off = 32'(tile_idx) * 32'(TILE_BYTES);
    assign dbg_kv_tiles_ready = kv_tiles_ready;

    always_comb begin
        desc_valid     = (sch_state == SCH_Q_ISSUE)
                      || (sch_state == SCH_K_ISSUE)
                      || (sch_state == SCH_V_ISSUE);
        desc_addr      = r_q_base[AXI_ADDR_W-1:0];
        desc_dst_addr  = 12'd0;
        desc_len_bytes = r_q_len_bytes;
        desc_dst       = 2'd0;
        unique case (sch_state)
            SCH_K_ISSUE: begin
                desc_addr      = r_k_base[AXI_ADDR_W-1:0] + AXI_ADDR_W'(tile_off);
                desc_dst_addr  = tile_off[11:0];
                desc_len_bytes = 32'(TILE_BYTES);
                desc_dst       = 2'd1;
            end
            SCH_V_ISSUE: begin
                desc_addr      = r_v_base[AXI_ADDR_W-1:0] + AXI_ADDR_W'(tile_off);
                desc_dst_addr  = tile_off[11:0];
                desc_len_bytes = 32'(TILE_BYTES);
                desc_dst       = 2'd2;
            end
            default: ;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sch_state      <= SCH_IDLE;
            tile_idx       <= '0;
            kv_tiles_ready <= '0;
            core_start     <= 1'b0;
            r_q_base       <= '0;
            r_k_base       <= '0;
            r_v_base       <= '0;
            r_num_tiles    <= '0;
            r_q_len_bytes  <= '0;
        end else begin
            core_start <= 1'b0;
            case (sch_state)
                SCH_IDLE: begin
                    tile_idx       <= '0;
                    kv_tiles_ready <= '0;
                    if (start && !cfg_error) begin
                        r_q_base      <= cfg_q_base;
                        r_k_base      <= cfg_k_base;
                        r_v_base      <= cfg_v_base;
                        r_num_tiles   <= cfg_seq_len >> LOG2_TILE;
                        r_q_len_bytes <= 32'(cfg_seq_len >> LOG2_TILE) * 32'(TILE_BYTES);
                        sch_state     <= SCH_Q_ISSUE;
                    end
                end
                SCH_Q_ISSUE: if (desc_ready) sch_state <= SCH_Q_WAIT;
                SCH_Q_WAIT:  if (dma_done)   sch_state <= SCH_K_ISSUE;
                SCH_K_ISSUE: if (desc_ready) sch_state <= SCH_K_WAIT;
                SCH_K_WAIT:  if (dma_done)   sch_state <= SCH_V_ISSUE;
                SCH_V_ISSUE: if (desc_ready) sch_state <= SCH_V_WAIT;
                SCH_V_WAIT: begin
                    if (dma_done) begin
                        kv_tiles_ready <= kv_tiles_ready + 16'd1;
                        if (tile_idx == 16'd0) core_start <= 1'b1;
                        if (tile_idx == r_num_tiles - 16'd1) begin
                            sch_state <= SCH_DONE;
                        end else begin
                            tile_idx  <= tile_idx + 16'd1;
                            sch_state <= SCH_K_ISSUE;
                        end
                    end
                end
                SCH_DONE: ;
                default: sch_state <= SCH_IDLE;
            endcase
        end
    end

    // ── Banked core; DMA fills its scratchpads via the vector write port ──
    logic              core_done;
    logic [11:0]       o_raddr;      // packer → core output read address
    logic signed [31:0] o_rdata;     // core → packer output read data

    flash_attn_core_banked #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM),
        .SEQ_LEN(SEQ_LEN), .SRAM_DEPTH(SRAM_DEPTH)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .done(core_done),
        .mode(1'b0), .kv_len(16'b0),
        .scale_q(scale_q), .scale_k(scale_k), .scale_v(scale_v),
        .q_we(1'b0), .q_waddr(12'b0), .q_wdata(8'b0),
        .k_we(1'b0), .k_waddr(12'b0), .k_wdata(8'b0),
        .v_we(1'b0), .v_waddr(12'b0), .v_wdata(8'b0),
        .dma_v_we(w_en), .dma_v_dst(w_dst), .dma_v_addr(w_addr), .dma_v_data(w_vdata),
        .out_raddr(o_raddr), .out_rdata(o_rdata),
        .causal(causal),
        .kv_tiles_ready(kv_tiles_ready)
    );

    // ── Output write-back: packer drains output_buffer → write DMA ────
    logic                  pkr_start, pkr_busy, pkr_done, pkr_out_re;
    logic                  src_valid, src_ready;
    logic [AXI_DATA_W-1:0] src_data;
    logic                  wdesc_valid, wdesc_ready, wb_done;

    output_writeback_packer #(
        .DATA_W(32), .AXI_DATA_W(AXI_DATA_W), .OUT_ADDR_W(12)
    ) u_packer (
        .clk(clk), .rst_n(rst_n),
        .start(pkr_start), .num_words(16'(O_WORDS)),
        .busy(pkr_busy), .done(pkr_done),
        .out_re(pkr_out_re), .out_raddr(o_raddr), .out_rdata(o_rdata),
        .src_valid(src_valid), .src_ready(src_ready), .src_data(src_data)
    );

    dma_write_engine #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .MAX_BURST(16)
    ) u_dma_wr (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(wdesc_valid), .desc_ready(wdesc_ready),
        .desc_addr(cfg_o_base[AXI_ADDR_W-1:0]), .desc_len_bytes(32'(O_BYTES)),
        .done(wb_done),
        .src_valid(src_valid), .src_ready(src_ready), .src_data(src_data),
        .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize),
        .m_awburst(m_awburst), .m_awvalid(m_awvalid), .m_awready(m_awready),
        .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_wvalid(m_wvalid), .m_wready(m_wready),
        .m_bresp(m_bresp), .m_bvalid(m_bvalid), .m_bready(m_bready)
    );

    // ── Write-back control FSM ────────────────────────────────────────
    //  On core done: pulse the packer start + issue one write descriptor.
    //  Top-level `done` asserts after the write DMA's final B response.
    typedef enum logic [1:0] { WB_IDLE, WB_ISSUE, WB_RUN, WB_FIN } wb_t;
    wb_t wb_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_state    <= WB_IDLE;
            pkr_start   <= 1'b0;
            wdesc_valid <= 1'b0;
            done        <= 1'b0;
        end else begin
            pkr_start <= 1'b0;   // default: 1-cycle pulse
            done      <= 1'b0;   // default: 1-cycle pulse
            case (wb_state)
                WB_IDLE: begin
                    if (core_done) begin
                        pkr_start   <= 1'b1;   // seen by packer next cycle (WB_ISSUE)
                        wdesc_valid <= 1'b1;
                        wb_state    <= WB_ISSUE;
                    end
                end
                WB_ISSUE: begin
                    if (wdesc_ready) begin     // write engine accepted the descriptor
                        wdesc_valid <= 1'b0;
                        wb_state    <= WB_RUN;
                    end
                end
                WB_RUN: begin
                    if (wb_done) begin
                        done     <= 1'b1;      // full round-trip complete
                        wb_state <= WB_FIN;
                    end
                end
                WB_FIN: ;                       // one-shot; holds until reset
                default: wb_state <= WB_IDLE;
            endcase
        end
    end

    logic wb_active;
    assign wb_active = (wb_state == WB_ISSUE) || (wb_state == WB_RUN);

    // ── Read-side performance counters (preserved from base top) ──────
    logic dma_busy;
    assign dma_busy = (sch_state != SCH_IDLE) && (sch_state != SCH_DONE);

    logic run_active, core_active, waiting_first;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_cycles           <= 32'd0;
            perf_dma_busy_cycles        <= 32'd0;
            perf_core_busy_cycles       <= 32'd0;
            perf_dma_bytes              <= 32'd0;
            perf_kv_tiles_loaded        <= 16'd0;
            perf_first_tile_wait_cycles <= 32'd0;
            run_active                  <= 1'b0;
            core_active                 <= 1'b0;
            waiting_first               <= 1'b0;
        end else if (start_accepted) begin
            perf_total_cycles           <= 32'd0;
            perf_dma_busy_cycles        <= 32'd0;
            perf_core_busy_cycles       <= 32'd0;
            perf_dma_bytes              <= 32'd0;
            perf_kv_tiles_loaded        <= 16'd0;
            perf_first_tile_wait_cycles <= 32'd0;
            run_active                  <= 1'b1;
            core_active                 <= 1'b0;
            waiting_first               <= 1'b1;
        end else begin
            if (run_active) begin
                perf_total_cycles <= perf_total_cycles + 32'd1;
                if (dma_busy)      perf_dma_busy_cycles <= perf_dma_busy_cycles + 32'd1;
                if (waiting_first) perf_first_tile_wait_cycles <= perf_first_tile_wait_cycles + 32'd1;
                if (w_en)          perf_dma_bytes <= perf_dma_bytes + 32'd16;
                if ((sch_state == SCH_V_WAIT) && dma_done)
                    perf_kv_tiles_loaded <= perf_kv_tiles_loaded + 16'd1;
                if (done)          run_active <= 1'b0;   // top done (after write-back)
            end

            // core-busy window uses the CORE's done (compute only)
            if (core_start)      core_active <= 1'b1;
            else if (core_done)  core_active <= 1'b0;
            if (core_active)     perf_core_busy_cycles <= perf_core_busy_cycles + 32'd1;

            if (core_start)      waiting_first <= 1'b0;
        end
    end

    // ── Write-back performance counters (new) ─────────────────────────
    logic wb_w_fire;
    assign wb_w_fire = m_wvalid & m_wready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_wb_bytes  <= 32'd0;
            perf_wb_cycles <= 32'd0;
            perf_wb_beats  <= 32'd0;
        end else if (start_accepted) begin
            perf_wb_bytes  <= 32'd0;
            perf_wb_cycles <= 32'd0;
            perf_wb_beats  <= 32'd0;
        end else begin
            if (wb_active)  perf_wb_cycles <= perf_wb_cycles + 32'd1;
            if (wb_w_fire) begin
                perf_wb_beats <= perf_wb_beats + 32'd1;
                perf_wb_bytes <= perf_wb_bytes + 32'(BYTES_PER_BEAT);
            end
        end
    end

    // ── Suppress unused warnings ──────────────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused;
    assign _unused = &{ w_vec, pkr_busy, pkr_done, pkr_out_re, 1'b0 };
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
