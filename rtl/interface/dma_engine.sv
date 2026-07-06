// ============================================================
//  dma_engine.sv — Synthesizable AXI4 read-master DMA
//
//  Pulls a contiguous byte region from external memory (DDR/HBM) over the
//  AXI4 read channels (AR/R) and writes it, one INT8 byte per cycle, into an
//  on-chip scratchpad SRAM (the flash_attn_core Q/K/V buffers).
//
//  One descriptor describes a single transfer:
//      desc_addr      — source byte address in DRAM
//      desc_dst_addr  — destination byte address in the target scratchpad SRAM
//      desc_len_bytes — transfer length in bytes (multiple of BYTES_PER_BEAT)
//      desc_dst       — which scratchpad to write (0=Q, 1=K, 2=V)
//
//  A descriptor is split into one or more INCR bursts of up to MAX_BURST
//  beats.  Each beat carries BYTES_PER_BEAT INT8 elements; because the
//  scratchpad is byte-wide (1 write/cycle) the engine throttles RREADY and
//  unpacks each beat over BYTES_PER_BEAT cycles.  `done` pulses for one cycle
//  when the whole descriptor has landed.
// ============================================================
`timescale 1ns/1ps

module dma_engine #(
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
    input  logic [AXI_ADDR_W-1:0] desc_addr,
    input  logic [11:0]           desc_dst_addr,
    input  logic [31:0]           desc_len_bytes,
    input  logic [1:0]            desc_dst,
    output logic                  done,        // 1-cycle pulse per descriptor

    // ── AXI4 read address channel (master) ───────────────────────────
    output logic [AXI_ADDR_W-1:0] m_araddr,
    output logic [7:0]            m_arlen,     // beats - 1
    output logic [2:0]            m_arsize,    // log2(bytes/beat)
    output logic [1:0]            m_arburst,   // 2'b01 = INCR
    output logic                  m_arvalid,
    input  logic                  m_arready,

    // ── AXI4 read data channel (master) ──────────────────────────────
    input  logic [AXI_DATA_W-1:0] m_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]            m_rresp,     // accepted, not checked
    input  logic                  m_rlast,     // beat count is tracked locally
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                  m_rvalid,
    output logic                  m_rready,

    // ── Scratchpad write-out (byte-wide) ─────────────────────────────
    output logic        w_we,
    output logic [1:0]  w_dst,
    output logic [11:0] w_addr,
    output logic [7:0]  w_data
);

    typedef enum logic [1:0] { S_IDLE, S_AR, S_R, S_WRITE } state_t;
    state_t state;

    logic [AXI_ADDR_W-1:0] cur_addr;        // DRAM read pointer
    logic [11:0]           wr_ptr;          // scratchpad write pointer
    logic [1:0]            dst;
    logic [31:0]           total_beats_left;
    logic [8:0]            burst_beats_left; // beats remaining in current burst

    logic [AXI_DATA_W-1:0] beat_buf;
    logic [BEAT_LOG2:0]    byte_idx;         // 0 .. BYTES_PER_BEAT-1

    // Beats requested in the current AR (combinational min(remaining, MAX_BURST))
    logic [8:0] req_beats;
    always_comb begin
        if (total_beats_left >= 32'(MAX_BURST)) req_beats = 9'(MAX_BURST);
        else                                    req_beats = total_beats_left[8:0];
    end

    // ── Combinational outputs ────────────────────────────────────────
    always_comb begin
        // AR channel
        m_araddr  = cur_addr;
        m_arlen   = 8'(req_beats - 9'd1);
        m_arsize  = 3'(BEAT_LOG2);
        m_arburst = 2'b01;                  // INCR
        m_arvalid = (state == S_AR);
        // R channel — only accept a beat when ready to start unpacking it
        m_rready  = (state == S_R);
        // Scratchpad write
        w_we   = (state == S_WRITE);
        w_dst  = dst;
        w_addr = wr_ptr;
        w_data = beat_buf[8*byte_idx +: 8];
        // Descriptor handshake
        desc_ready = (state == S_IDLE);
    end

    // ── FSM ──────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            cur_addr         <= '0;
            wr_ptr           <= '0;
            dst              <= '0;
            total_beats_left <= '0;
            burst_beats_left <= '0;
            beat_buf         <= '0;
            byte_idx         <= '0;
            done             <= 1'b0;
        end else begin
            done <= 1'b0;   // default: pulse

            case (state)

                S_IDLE: begin
                    if (desc_valid) begin
                        cur_addr         <= desc_addr;
                        wr_ptr           <= desc_dst_addr;
                        dst              <= desc_dst;
                        // bytes → beats (len is a multiple of BYTES_PER_BEAT)
                        total_beats_left <= desc_len_bytes >> BEAT_LOG2;
                        state            <= S_AR;
                    end
                end

                S_AR: begin
                    if (m_arready) begin
                        burst_beats_left <= req_beats;
                        state            <= S_R;
                    end
                end

                S_R: begin
                    if (m_rvalid) begin
                        beat_buf <= m_rdata;
                        byte_idx <= '0;
                        state    <= S_WRITE;
                    end
                end

                S_WRITE: begin
                    if (int'(byte_idx) == BYTES_PER_BEAT - 1) begin
                        // Last byte of this beat — advance counters.
                        wr_ptr           <= wr_ptr + 12'd1;
                        cur_addr         <= cur_addr + AXI_ADDR_W'(BYTES_PER_BEAT);
                        total_beats_left <= total_beats_left - 32'd1;
                        burst_beats_left <= burst_beats_left - 9'd1;
                        if (burst_beats_left == 9'd1) begin
                            // Finished the last beat of this burst.
                            if (total_beats_left == 32'd1) begin
                                done  <= 1'b1;       // descriptor complete
                                state <= S_IDLE;
                            end else begin
                                state <= S_AR;        // issue next burst
                            end
                        end else begin
                            state <= S_R;             // next beat in burst
                        end
                    end else begin
                        wr_ptr   <= wr_ptr + 12'd1;
                        byte_idx <= byte_idx + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
