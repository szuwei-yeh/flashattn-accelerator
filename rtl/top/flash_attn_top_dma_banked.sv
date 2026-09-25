// ============================================================
//  flash_attn_top_dma_banked.sv — DMA-fed banked single-head top (Stage 3B-3)
//
//  Combines the Stage-3A vector DMA with the Stage-3B banked core:
//      DRAM → dma_engine_vec (16-byte stripe writes) → banked scratchpads
//           → banked_tile_loader → tile registers → systolic array
//
//  Same open-loop scheduler as flash_attn_top_dma.sv (bulk-load Q, then stream
//  KV tiles in order, bumping kv_tiles_ready), but the DMA writes 16-byte
//  stripes (w_vec=1) into flash_attn_core_banked's vector fill port.
//
//  Runtime-configurable DMA scheduler (compile-time HEAD_DIM/TILE_SIZE):
//    - cfg_seq_len  : number of KV tiles streamed = cfg_seq_len / TILE_SIZE.
//                     Must be > 0, <= SEQ_LEN, and a multiple of TILE_SIZE,
//                     else cfg_error is asserted and the scheduler will not run.
//    - cfg_q/k/v_base : DRAM source byte base addresses for Q / K / V.
//    Scratchpad destination layout is unchanged (Q @ 0, KV tile @ tile_off).
//    Config is sampled when `start` is accepted and held for the run.
//    With cfg_seq_len=SEQ_LEN and the default bases, behaviour is identical to
//    the previous fixed scheduler.
//    NOTE: the compute core's sequence extent is still the compile-time SEQ_LEN
//    (it has no runtime seq_len input); the valid use is cfg_seq_len == SEQ_LEN.
//    Shorter cfg_seq_len relocates/limits the DMA stream but the core would
//    still iterate over SEQ_LEN — running fewer tiles correctly would need a
//    runtime-length core (future work).
//
//  Also exposes RTL-visible performance counters (see ports). They are reset
//  when a run starts and hold their values after `done`.
//
//  Does not touch flash_attn_top_dma.sv, flash_attn_core.sv, dma_engine.sv,
//  flash_attn_core_banked.sv, or tile_controller_banked.sv.
// ============================================================
`timescale 1ns/1ps

module flash_attn_top_dma_banked #(
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
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // ── Runtime DMA scheduler config (sampled at accepted start) ──────
    input  logic [15:0]           cfg_seq_len,
    input  logic [31:0]           cfg_q_base,
    input  logic [31:0]           cfg_k_base,
    input  logic [31:0]           cfg_v_base,
    output logic                  cfg_error,   // invalid cfg_seq_len → won't run

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
    output logic [31:0] perf_total_cycles,            // accepted start → done
    output logic [31:0] perf_dma_busy_cycles,         // scheduler active / DMA transferring
    output logic [31:0] perf_core_busy_cycles,        // core_start → core done
    output logic [31:0] perf_dma_bytes,               // += 16 per vector scratchpad write
    output logic [15:0] perf_kv_tiles_loaded,         // completed (K,V) tile pairs
    output logic [31:0] perf_first_tile_wait_cycles,  // start → core_start

    output logic [15:0] dbg_kv_tiles_ready
);

    // ── Config validation (combinational, from live inputs) ──────────
    assign cfg_error = (cfg_seq_len == 16'd0)
                    || (cfg_seq_len > 16'(SEQ_LEN))
                    || ((cfg_seq_len & 16'(TILE_SIZE - 1)) != 16'd0);

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
    logic [31:0] r_q_len_bytes;     // = r_num_tiles * TILE_BYTES

    logic start_accepted;
    assign start_accepted = (sch_state == SCH_IDLE) && start && !cfg_error;

    logic [31:0] tile_off;
    assign tile_off = 32'(tile_idx) * 32'(TILE_BYTES);
    assign dbg_kv_tiles_ready = kv_tiles_ready;

    // Descriptor: Q bulk-load (runtime length / base), then KV tiles.
    always_comb begin
        desc_valid     = (sch_state == SCH_Q_ISSUE)
                      || (sch_state == SCH_K_ISSUE)
                      || (sch_state == SCH_V_ISSUE);
        desc_addr      = r_q_base[AXI_ADDR_W-1:0];   // Q source base
        desc_dst_addr  = 12'd0;                       // Q dest @ 0 (unchanged)
        desc_len_bytes = r_q_len_bytes;
        desc_dst       = 2'd0;
        unique case (sch_state)
            SCH_K_ISSUE: begin
                desc_addr      = r_k_base[AXI_ADDR_W-1:0] + AXI_ADDR_W'(tile_off);
                desc_dst_addr  = tile_off[11:0];      // KV dest @ tile_off (unchanged)
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
                        // sample config for this run
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
            // reset all counters at the start of a run
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
                if (done)          run_active <= 1'b0;
            end

            // core-busy window (independent of run_active edge timing)
            if (core_start)      core_active <= 1'b1;
            else if (done)       core_active <= 1'b0;
            if (core_active)     perf_core_busy_cycles <= perf_core_busy_cycles + 32'd1;

            if (core_start)      waiting_first <= 1'b0;
        end
    end

    // ── Banked core; DMA fills its scratchpads via the vector write port ──
    flash_attn_core_banked #(
        .TILE_SIZE(TILE_SIZE), .HEAD_DIM(HEAD_DIM),
        .SEQ_LEN(SEQ_LEN), .SRAM_DEPTH(SRAM_DEPTH)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .done(done),
        .mode(1'b0), .kv_len(16'b0),
        .scale_q(scale_q), .scale_k(scale_k), .scale_v(scale_v),
        // scalar preload ports unused (DMA fills via vector port)
        .q_we(1'b0), .q_waddr(12'b0), .q_wdata(8'b0),
        .k_we(1'b0), .k_waddr(12'b0), .k_wdata(8'b0),
        .v_we(1'b0), .v_waddr(12'b0), .v_wdata(8'b0),
        // vector DMA fill port
        .dma_v_we(w_en), .dma_v_dst(w_dst), .dma_v_addr(w_addr), .dma_v_data(w_vdata),
        .out_raddr(out_raddr), .out_rdata(out_rdata),
        .causal(causal),
        .kv_tiles_ready(kv_tiles_ready)
    );

endmodule
