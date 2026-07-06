// ============================================================
//  dma_write_engine.sv — Synthesizable AXI4 write-master DMA
//
//  The write-side mirror of dma_engine.sv / dma_engine_vec.sv: it drains a
//  streaming source (src_valid/src_ready/src_data, one AXI_DATA_W beat/cycle)
//  and writes it to a contiguous byte region in external memory (DDR/HBM) over
//  the AXI4 write channels (AW/W/B).
//
//  One descriptor describes a single transfer:
//      desc_addr      — destination byte base address in DRAM
//      desc_len_bytes — transfer length in bytes (multiple of BYTES_PER_BEAT)
//
//  The descriptor is split into one or more INCR bursts of up to MAX_BURST
//  beats.  Every beat is a full bus-width write (WSTRB all ones — full-beat
//  writes only, for now).  Source beats are consumed exactly on accepted W
//  handshakes, so the engine naturally back-pressures a slow source or slave.
//  `done` pulses for one cycle after the final burst's B response is accepted.
//
//  Kept as a SEPARATE module (not merged into dma_engine_vec.sv) so the proven
//  read path is untouched; the two engines can later run concurrently on the
//  independent AXI read/write channel groups.
// ============================================================
`timescale 1ns/1ps

module dma_write_engine #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int MAX_BURST  = 16,           // beats per AXI burst
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8,
    localparam int BEAT_LOG2      = $clog2(BYTES_PER_BEAT)
)(
    input  logic clk,
    input  logic rst_n,

    // ── Descriptor input ──────────────────────────────────────────────
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [AXI_ADDR_W-1:0] desc_addr,       // DRAM byte base (dest)
    input  logic [31:0]           desc_len_bytes,   // multiple of BYTES_PER_BEAT
    output logic                  done,             // 1-cycle pulse per descriptor

    // ── Source stream (one AXI_DATA_W-bit beat per accepted handshake) ─
    input  logic                  src_valid,
    output logic                  src_ready,
    input  logic [AXI_DATA_W-1:0] src_data,

    // ── AXI4 write address channel (master) ──────────────────────────
    output logic [AXI_ADDR_W-1:0] m_awaddr,
    output logic [7:0]            m_awlen,     // beats - 1
    output logic [2:0]            m_awsize,    // log2(bytes/beat)
    output logic [1:0]            m_awburst,   // 2'b01 = INCR
    output logic                  m_awvalid,
    input  logic                  m_awready,

    // ── AXI4 write data channel (master) ─────────────────────────────
    output logic [AXI_DATA_W-1:0]     m_wdata,
    output logic [BYTES_PER_BEAT-1:0] m_wstrb,
    output logic                      m_wlast,
    output logic                      m_wvalid,
    input  logic                      m_wready,

    // ── AXI4 write response channel (master) ─────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]            m_bresp,     // accepted, not checked (assume OKAY)
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                  m_bvalid,
    output logic                  m_bready
);

    typedef enum logic [1:0] { S_IDLE, S_AW, S_W, S_B } state_t;
    state_t state;

    logic [AXI_ADDR_W-1:0] cur_addr;          // DRAM write pointer
    logic [31:0]           total_beats_left;
    logic [8:0]            burst_beats_left;   // beats remaining in current burst

    // Beats requested in the current AW (combinational min(remaining, MAX_BURST))
    logic [8:0] req_beats;
    always_comb begin
        if (total_beats_left >= 32'(MAX_BURST)) req_beats = 9'(MAX_BURST);
        else                                    req_beats = total_beats_left[8:0];
    end

    // ── Combinational outputs ────────────────────────────────────────
    always_comb begin
        // AW channel
        m_awaddr  = cur_addr;
        m_awlen   = 8'(req_beats - 9'd1);
        m_awsize  = 3'(BEAT_LOG2);
        m_awburst = 2'b01;                        // INCR
        m_awvalid = (state == S_AW);
        // W channel — pass the source beat straight through; full-beat writes
        m_wdata   = src_data;
        m_wstrb   = {BYTES_PER_BEAT{1'b1}};
        m_wlast   = (burst_beats_left == 9'd1);
        m_wvalid  = (state == S_W) & src_valid;
        src_ready = (state == S_W) & m_wready;
        // B channel
        m_bready  = (state == S_B);
        // Descriptor handshake
        desc_ready = (state == S_IDLE);
    end

    // ── FSM ──────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            cur_addr         <= '0;
            total_beats_left <= '0;
            burst_beats_left <= '0;
            done             <= 1'b0;
        end else begin
            done <= 1'b0;   // default: 1-cycle pulse

            case (state)

                S_IDLE: begin
                    if (desc_valid) begin
                        cur_addr         <= desc_addr;
                        // bytes → beats (len is a multiple of BYTES_PER_BEAT)
                        total_beats_left <= desc_len_bytes >> BEAT_LOG2;
                        state            <= S_AW;
                    end
                end

                S_AW: begin
                    if (m_awready) begin
                        burst_beats_left <= req_beats;
                        state            <= S_W;
                    end
                end

                S_W: begin
                    if (m_wvalid && m_wready) begin      // beat accepted
                        cur_addr         <= cur_addr + AXI_ADDR_W'(BYTES_PER_BEAT);
                        total_beats_left <= total_beats_left - 32'd1;
                        burst_beats_left <= burst_beats_left - 9'd1;
                        if (burst_beats_left == 9'd1)     // last beat of this burst
                            state <= S_B;                 // wait for B response
                    end
                end

                S_B: begin
                    if (m_bvalid) begin
                        if (total_beats_left == 32'd0) begin
                            done  <= 1'b1;                // descriptor complete
                            state <= S_IDLE;
                        end else begin
                            state <= S_AW;                // issue next burst
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (desc_valid && (state == S_IDLE)) begin
            if ((desc_len_bytes % BYTES_PER_BEAT) != 0)
                $error("dma_write_engine: desc_len_bytes not a multiple of %0d", BYTES_PER_BEAT);
            if (desc_len_bytes == 32'd0)
                $error("dma_write_engine: desc_len_bytes is zero");
        end
    end
    // synthesis translate_on

endmodule
