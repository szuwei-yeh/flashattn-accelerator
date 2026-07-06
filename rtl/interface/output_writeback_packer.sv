// ============================================================
//  output_writeback_packer.sv — output-buffer drain → 64-bit source stream
//
//  Stage WB-2 datapath adapter.  Reads consecutive 32-bit result words from an
//  output-buffer-like read port (1-cycle synchronous read, same timing as
//  sram_1r1w / output_buffer.sv's re_ext/raddr_ext/rdata_ext) and packs two
//  consecutive words into one AXI_DATA_W (=64-bit) beat:
//
//      word[2k]   → src_data[31:0]     (low  half, lower DRAM address)
//      word[2k+1] → src_data[63:32]    (high half, higher DRAM address)
//
//  The packed beats are presented as a valid/ready source stream to
//  dma_write_engine.  A beat is held stable while `src_ready` is low, so a slow
//  or bursty consumer (e.g. the write engine between AXI bursts, or under write
//  latency) back-pressures the packer cleanly with no data loss.
//
//  Control: pulse `start` with `num_words` (must be even).  `busy` is high for
//  the whole drain; `done` pulses for one cycle after the last beat is accepted.
//
//  Synthesizable; no dependence on the output buffer's internals — it only uses
//  the read port, so it drops straight onto output_buffer.re_ext/raddr_ext in
//  WB-3 without touching that module.
// ============================================================
`timescale 1ns/1ps

module output_writeback_packer #(
    parameter int DATA_W     = 32,
    parameter int AXI_DATA_W = 64,
    parameter int OUT_ADDR_W = 12,           // output-buffer address width
    parameter int NW_W       = 16,           // num_words counter width
    localparam int WORDS_PER_BEAT = AXI_DATA_W / DATA_W   // = 2
)(
    input  logic clk,
    input  logic rst_n,

    // ── Control ───────────────────────────────────────────────────────
    input  logic              start,
    input  logic [NW_W-1:0]   num_words,     // must be even
    output logic              busy,
    output logic              done,          // 1-cycle pulse

    // ── Output-buffer read port (1-cycle synchronous read) ────────────
    output logic                  out_re,
    output logic [OUT_ADDR_W-1:0] out_raddr,
    input  logic [DATA_W-1:0]     out_rdata,

    // ── Packed source stream to dma_write_engine ──────────────────────
    output logic                  src_valid,
    input  logic                  src_ready,
    output logic [AXI_DATA_W-1:0] src_data
);
    // WORDS_PER_BEAT is 2 in this design; the low/high capture below assumes it.
    // (kept as a localparam for documentation / lint symmetry)
    /* verilator lint_off UNUSEDPARAM */
    localparam int _WPB = WORDS_PER_BEAT;
    /* verilator lint_on UNUSEDPARAM */

    typedef enum logic [2:0] {
        P_IDLE, P_ISSUE_LO, P_ISSUE_HI, P_FORM, P_SEND, P_DONE
    } state_t;
    state_t state;

    logic [NW_W-1:0]     word_idx;      // index of the low word of the current pair
    logic [NW_W-1:0]     total_words;
    logic [DATA_W-1:0]   lo_word;
    logic [AXI_DATA_W-1:0] beat_data;

    // ── Combinational outputs ────────────────────────────────────────
    always_comb begin
        out_re    = (state == P_ISSUE_LO) || (state == P_ISSUE_HI);
        // lo address in P_ISSUE_LO, hi (word_idx+1) in P_ISSUE_HI
        out_raddr = (state == P_ISSUE_HI) ? (word_idx[OUT_ADDR_W-1:0] + OUT_ADDR_W'(1))
                                          :  word_idx[OUT_ADDR_W-1:0];
        src_valid = (state == P_SEND);
        src_data  = beat_data;
        busy      = (state != P_IDLE);
        done      = (state == P_DONE);
    end

    // ── FSM ──────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= P_IDLE;
            word_idx    <= '0;
            total_words <= '0;
            lo_word     <= '0;
            beat_data   <= '0;
        end else begin
            case (state)
                P_IDLE: begin
                    word_idx <= '0;
                    if (start) begin
                        total_words <= num_words;
                        state <= (num_words == '0) ? P_DONE : P_ISSUE_LO;
                    end
                end

                // Issue read of the low word (out_re/out_raddr driven combinationally).
                P_ISSUE_LO: state <= P_ISSUE_HI;

                // out_rdata now holds the low word; capture it and issue the high read.
                P_ISSUE_HI: begin
                    lo_word <= out_rdata;
                    state   <= P_FORM;
                end

                // out_rdata now holds the high word; assemble the 64-bit beat.
                P_FORM: begin
                    beat_data <= {out_rdata, lo_word};
                    state     <= P_SEND;
                end

                // Present the beat; hold until the write engine accepts it.
                P_SEND: begin
                    if (src_ready) begin
                        word_idx <= word_idx + NW_W'(2);
                        if ((word_idx + NW_W'(2)) >= total_words) state <= P_DONE;
                        else                                      state <= P_ISSUE_LO;
                    end
                end

                P_DONE: state <= P_IDLE;   // done asserted for exactly this cycle

                default: state <= P_IDLE;
            endcase
        end
    end

    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (start && (state == P_IDLE) && (num_words[0] != 1'b0))
            $error("output_writeback_packer: num_words (%0d) must be even", num_words);
    end
    // synthesis translate_on

endmodule
