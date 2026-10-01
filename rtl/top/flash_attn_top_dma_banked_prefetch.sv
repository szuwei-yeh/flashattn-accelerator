// ============================================================
//  flash_attn_top_dma_banked_prefetch.sv — DMA-fed banked top, prefetch variant.
//
//  Instantiates flash_attn_core_banked_prefetch (banked core WITH KV
//  double-buffering) behind the vector DMA/read-side scheduler. Transaction
//  configuration is captured at accepted start and runtime-short sequences are
//  rejected because compute extent is compile-time fixed.
//
//  Does not touch flash_attn_top_dma_banked.sv, flash_attn_core_banked.sv,
//  tile_controller_banked.sv, or the dma_banked_top_N64 target.
//
//  Latency hiding still holds: the KV prefetch only launches for a tile that the
//  residency gate reports resident (next_tile_col/TILE < kv_tiles_ready), so a
//  not-yet-streamed tile falls back to the normal residency-stalled reload —
//  correctness stays latency-independent, exactly as in the non-prefetch top.
// ============================================================
`timescale 1ns/1ps

module flash_attn_top_dma_banked_prefetch #(
    parameter int SEQ_LEN    = 64,
    parameter int HEAD_DIM   = 16,
    parameter int TILE_SIZE  = 16,
    parameter int SRAM_DEPTH = 4096,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    localparam int TILE_BYTES = TILE_SIZE * HEAD_DIM,
    localparam int LOG2_TILE  = $clog2(TILE_SIZE),
    localparam int VEC_W      = 16 * 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,

    input  logic        causal,
    // All transaction-level configuration is sampled on accepted start.
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // ── Runtime DMA scheduler config (sampled at accepted start) ──────
    input  logic [15:0]           cfg_seq_len,
    input  logic [31:0]           cfg_q_base,
    input  logic [31:0]           cfg_k_base,
    input  logic [31:0]           cfg_v_base,
    output logic                  cfg_error,
    output logic                  dma_error, // sticky failure; common AXI reset required

    input  logic [11:0]        out_raddr,
    output logic signed [31:0] out_rdata,

    // AXI4 read-master ports (to external DRAM)
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

    // ── RTL-visible performance counters (reset at run start, hold after done) ──
    output logic [31:0] perf_total_cycles,
    output logic [31:0] perf_dma_busy_cycles,
    output logic [31:0] perf_core_busy_cycles,
    output logic [31:0] perf_dma_bytes,
    output logic [15:0] perf_kv_tiles_loaded,
    output logic [31:0] perf_first_tile_wait_cycles,

    output logic [15:0] dbg_kv_tiles_ready
);

    // ── Config validation (combinational, from live inputs) ──────────
    // The core is compile-time sized; runtime-short sequences are rejected.
    // Vector DMA descriptors operate on aligned 16-byte scratchpad stripes.
    // Validate each whole matrix before tiled K/V address additions can wrap.
    localparam logic [31:0] LAST_MATRIX_BASE =
        32'hFFFF_FFFF - 32'(SEQ_LEN * HEAD_DIM - 1);
    assign cfg_error = (cfg_seq_len != 16'(SEQ_LEN))
                    || ((cfg_seq_len & 16'(TILE_SIZE - 1)) != 16'd0)
                    || (cfg_q_base[3:0] != 4'b0)
                    || (cfg_k_base[3:0] != 4'b0)
                    || (cfg_v_base[3:0] != 4'b0)
                    || (cfg_q_base > LAST_MATRIX_BASE)
                    || (cfg_k_base > LAST_MATRIX_BASE)
                    || (cfg_v_base > LAST_MATRIX_BASE);

    initial begin : p_parameter_guard
        if (TILE_SIZE != 16)
            $fatal(1, "flash_attn_top_dma_banked_prefetch: TILE_SIZE must be 16");
        if ((HEAD_DIM % TILE_SIZE) != 0)
            $fatal(1, "flash_attn_top_dma_banked_prefetch: HEAD_DIM must be a TILE_SIZE multiple");
        if (!((HEAD_DIM == 16) || (HEAD_DIM == 64)))
            $fatal(1, "flash_attn_top_dma_banked_prefetch: supported HEAD_DIM values are 16 and 64");
        if ((SEQ_LEN == 0) || ((SEQ_LEN % TILE_SIZE) != 0))
            $fatal(1, "flash_attn_top_dma_banked_prefetch: SEQ_LEN must be a non-zero TILE_SIZE multiple");
        if ((SEQ_LEN * HEAD_DIM) > SRAM_DEPTH)
            $fatal(1, "flash_attn_top_dma_banked_prefetch: SEQ_LEN*HEAD_DIM exceeds SRAM_DEPTH");
        if (SRAM_DEPTH != 4096)
            $fatal(1, "flash_attn_top_dma_banked_prefetch: supported SRAM_DEPTH is 4096 (fixed 12-bit internal interfaces)");
        if (AXI_ADDR_W != 32)
            $fatal(1, "flash_attn_top_dma_banked_prefetch: AXI_ADDR_W must be 32");
        if (AXI_DATA_W != 64)
            $fatal(1, "flash_attn_top_dma_banked_prefetch: AXI_DATA_W must be 64");
    end

    // ── Vector DMA engine ─────────────────────────────────────────────
    logic                  desc_valid, desc_ready, dma_done;
    logic [AXI_ADDR_W-1:0] desc_addr;
    logic [11:0]           desc_dst_addr;
    logic [31:0]           desc_len_bytes;
    logic [1:0]            desc_dst;

    logic              w_en, w_vec;
    logic [1:0]        w_dst;
    logic [11:0]       w_addr;
    logic [VEC_W-1:0]  w_vdata;

    dma_engine_vec #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W), .NUM_BANKS(16)
    ) u_dma (
        .clk(clk), .rst_n(rst_n),
        .desc_valid(desc_valid), .desc_ready(desc_ready),
        .desc_addr(desc_addr), .desc_dst_addr(desc_dst_addr),
        .desc_len_bytes(desc_len_bytes), .desc_dst(desc_dst), .done(dma_done), .error(dma_error),
        .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize),
        .m_arburst(m_arburst), .m_arvalid(m_arvalid), .m_arready(m_arready),
        .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast),
        .m_rvalid(m_rvalid), .m_rready(m_rready),
        .w_en(w_en), .w_vec(w_vec), .w_dst(w_dst), .w_addr(w_addr), .w_vdata(w_vdata)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused_wvec; assign _unused_wvec = w_vec;   // always 1
    /* verilator lint_on UNUSEDSIGNAL */

    // ── Scheduler ────────────────────────────────────────────────────
    typedef enum logic [2:0] {
        SCH_IDLE, SCH_Q_ISSUE, SCH_Q_WAIT,
        SCH_K_ISSUE, SCH_K_WAIT, SCH_V_ISSUE, SCH_V_WAIT, SCH_DONE
    } sch_t;
    sch_t sch_state;

    logic [15:0] tile_idx;
    logic [15:0] kv_tiles_ready;
    logic        core_start;

    // Latched runtime config (sampled at accepted start)
    logic [31:0] r_q_base, r_k_base, r_v_base;
    logic [15:0] r_num_tiles;
    logic [31:0] r_q_len_bytes;
    logic signed [15:0] r_scale_q, r_scale_k, r_scale_v;
    logic               r_causal;

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
            r_scale_q      <= '0;
            r_scale_k      <= '0;
            r_scale_v      <= '0;
            r_causal       <= 1'b0;
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
                        r_scale_q     <= scale_q;
                        r_scale_k     <= scale_k;
                        r_scale_v     <= scale_v;
                        r_causal      <= causal;
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
                // Keep the single-shot interface contract; reset re-arms the
                // controller and first-tile overwrite initializes output data.
                SCH_DONE: ;
                default: sch_state <= SCH_IDLE;
            endcase
        end
    end

    // ── Performance counters ─────────────────────────────────────────
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
                if (done || dma_error) run_active <= 1'b0;
            end

            if (core_start)      core_active <= 1'b1;
            else if (done || dma_error) core_active <= 1'b0;
            if (core_active)     perf_core_busy_cycles <= perf_core_busy_cycles + 32'd1;

            if (core_start)      waiting_first <= 1'b0;
        end
    end

    logic core_done;
    // Errors invalidate all partial results, even if causal compute can finish
    // without the failed later tile. The host observes dma_error, never success.
    assign done = core_done && !dma_error;

    // ── Banked prefetch core; DMA fills its scratchpads via the vector port ──
    flash_attn_core_banked_prefetch #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM),
        .SEQ_LEN(SEQ_LEN), .SRAM_DEPTH(SRAM_DEPTH)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .done(core_done),
        .mode(1'b0), .kv_len(16'b0),
        .scale_q(r_scale_q), .scale_k(r_scale_k), .scale_v(r_scale_v),
        // scalar preload ports unused (DMA fills via vector port)
        .q_we(1'b0), .q_waddr(12'b0), .q_wdata(8'b0),
        .k_we(1'b0), .k_waddr(12'b0), .k_wdata(8'b0),
        .v_we(1'b0), .v_waddr(12'b0), .v_wdata(8'b0),
        // vector DMA fill port
        .dma_v_we(w_en), .dma_v_dst(w_dst), .dma_v_addr(w_addr), .dma_v_data(w_vdata),
        .out_raddr(out_raddr), .out_rdata(out_rdata),
        .causal(r_causal),
        .kv_tiles_ready(kv_tiles_ready)
    );

endmodule
