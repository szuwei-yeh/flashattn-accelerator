// ============================================================
//  dma_engine_vec.sv — AXI4 read-master DMA with VECTOR scratchpad write-out
//
//  Same AXI4 read front-end as dma_engine.sv (AR/R, INCR bursts), but instead
//  of unpacking each 64-bit beat into 8 byte-writes (1 B/cyc, RREADY throttled),
//  it collects BEATS_PER_STRIPE consecutive beats into one NUM_BANKS-byte stripe
//  and issues a single vector write (w_vec=1) into a banked_scratchpad.
//
//  RREADY is NOT throttled, so the transfer is AXI-bound (~BYTES_PER_BEAT/cyc)
//  and the scratchpad sees one 16-byte write per stripe instead of 16 writes.
//
//  Stage-3A constraints (asserted in simulation, documented here):
//    - AXI_DATA_W * BEATS_PER_STRIPE == VEC_W   (here 64*2 == 128)
//    - desc_dst_addr is NUM_BANKS-aligned (low LANE_LOG2 bits = 0)
//    - desc_len_bytes is a multiple of STRIPE_BYTES (= NUM_BANKS)
//  Tile transfers always satisfy these (TILE_BYTES = TILE_SIZE*HEAD_DIM, both
//  multiples of 16).
// ============================================================
`timescale 1ns/1ps

module dma_engine_vec #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int NUM_BANKS  = 16,
    parameter int MAX_BURST  = 16,
    localparam int BYTES_PER_BEAT   = AXI_DATA_W / 8,
    localparam int BEAT_LOG2        = $clog2(BYTES_PER_BEAT),
    localparam int VEC_W            = NUM_BANKS * 8,
    localparam int BEATS_PER_STRIPE = VEC_W / AXI_DATA_W,   // 128/64 = 2
    localparam int STRIPE_BYTES     = VEC_W / 8             // 16
)(
    input  logic clk,
    input  logic rst_n,

    // ── Descriptor input ──────────────────────────────────────────────
    input  logic                  desc_valid,
    output logic                  desc_ready,
    input  logic [AXI_ADDR_W-1:0] desc_addr,       // DRAM byte base
    input  logic [11:0]           desc_dst_addr,    // scratchpad byte base (16-aligned)
    input  logic [31:0]           desc_len_bytes,   // multiple of STRIPE_BYTES
    input  logic [1:0]            desc_dst,
    output logic                  done,

    // ── AXI4 read address channel (master) ───────────────────────────
    output logic [AXI_ADDR_W-1:0] m_araddr,
    output logic [7:0]            m_arlen,
    output logic [2:0]            m_arsize,
    output logic [1:0]            m_arburst,
    output logic                  m_arvalid,
    input  logic                  m_arready,

    // ── AXI4 read data channel (master) ──────────────────────────────
    input  logic [AXI_DATA_W-1:0] m_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]            m_rresp,
    input  logic                  m_rlast,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                  m_rvalid,
    output logic                  m_rready,

    // ── Vector scratchpad write-out ──────────────────────────────────
    output logic              w_en,
    output logic              w_vec,    // always 1
    output logic [1:0]        w_dst,
    output logic [11:0]       w_addr,   // 16-byte aligned stripe address
    output logic [VEC_W-1:0]  w_vdata
);

    typedef enum logic [1:0] { S_IDLE, S_AR, S_R } state_t;
    state_t state;

    logic [AXI_ADDR_W-1:0] cur_addr;          // DRAM read pointer
    logic [11:0]           stripe_addr;        // scratchpad stripe write pointer
    logic [1:0]            dst;
    logic [31:0]           total_beats_left;
    logic [8:0]            burst_beats_left;

    // Beat-within-stripe accumulator (low beats registered; top beat is current)
    logic [AXI_DATA_W*(BEATS_PER_STRIPE-1)-1:0] stripe_lo;  // here: 1 beat (64b)
    logic                  beat_in_stripe;     // 0 or 1 (BEATS_PER_STRIPE=2)

    // Beats requested in current AR
    logic [8:0] req_beats;
    always_comb begin
        if (total_beats_left >= 32'(MAX_BURST)) req_beats = 9'(MAX_BURST);
        else                                    req_beats = total_beats_left[8:0];
    end

    // The completing (second) beat of a stripe forms the vector write this cycle
    logic stripe_complete;
    assign stripe_complete = (state == S_R) & m_rvalid & (beat_in_stripe == 1'b1);

    always_comb begin
        m_araddr  = cur_addr;
        m_arlen   = 8'(req_beats - 9'd1);
        m_arsize  = 3'(BEAT_LOG2);
        m_arburst = 2'b01;                 // INCR
        m_arvalid = (state == S_AR);
        m_rready  = (state == S_R);        // no throttling

        // Vector write: {current beat, registered low beat}
        w_en    = stripe_complete;
        w_vec   = 1'b1;
        w_dst   = dst;
        w_addr  = stripe_addr;
        w_vdata = {m_rdata, stripe_lo};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            cur_addr         <= '0;
            stripe_addr      <= '0;
            dst              <= '0;
            total_beats_left <= '0;
            burst_beats_left <= '0;
            stripe_lo        <= '0;
            beat_in_stripe   <= 1'b0;
            done             <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    beat_in_stripe <= 1'b0;
                    if (desc_valid) begin
                        cur_addr         <= desc_addr;
                        stripe_addr      <= desc_dst_addr;
                        dst              <= desc_dst;
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
                        // accumulate beat
                        if (beat_in_stripe == 1'b0) begin
                            stripe_lo      <= m_rdata;
                            beat_in_stripe <= 1'b1;
                        end else begin
                            // stripe_complete: vector write emitted combinationally
                            stripe_addr    <= stripe_addr + 12'(STRIPE_BYTES);
                            beat_in_stripe <= 1'b0;
                        end

                        // advance burst / descriptor counters
                        cur_addr         <= cur_addr + AXI_ADDR_W'(BYTES_PER_BEAT);
                        total_beats_left <= total_beats_left - 32'd1;
                        burst_beats_left <= burst_beats_left - 9'd1;
                        if (burst_beats_left == 9'd1) begin
                            if (total_beats_left == 32'd1) begin
                                done  <= 1'b1;
                                state <= S_IDLE;
                            end else begin
                                state <= S_AR;
                            end
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign desc_ready = (state == S_IDLE);

    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (desc_valid && (state == S_IDLE)) begin
            if (desc_dst_addr[$clog2(NUM_BANKS)-1:0] != '0)
                $error("dma_engine_vec: desc_dst_addr not %0d-byte aligned", NUM_BANKS);
            if ((desc_len_bytes % STRIPE_BYTES) != 0)
                $error("dma_engine_vec: desc_len_bytes not a multiple of %0d", STRIPE_BYTES);
        end
    end
    // synthesis translate_on

endmodule
