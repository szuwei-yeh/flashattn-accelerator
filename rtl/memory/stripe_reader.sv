// ============================================================
//  stripe_reader.sv — drains a banked_scratchpad at one 16-byte stripe/cycle
//
//  Given a base address and a row count, issues one aligned vector (stripe)
//  read per cycle to a banked_scratchpad and streams the NUM_BANKS-byte rows
//  out.  This is the fast tile-load path: a 16×16 INT8 tile (256 B) drains in
//  NUM_BANKS beats instead of 256 byte-reads.
//
//  The scratchpad has a 1-cycle read latency, so out_valid/out_row are aligned
//  to r_vdata by issuing addresses combinationally and registering the
//  valid/row tag one cycle.
// ============================================================
`timescale 1ns/1ps

module stripe_reader #(
    parameter int NUM_BANKS  = 16,
    parameter int DATA_WIDTH = 8,
    parameter int ADDR_W     = 12,
    localparam int LANE_LOG2 = $clog2(NUM_BANKS),
    localparam int VEC_W     = NUM_BANKS * DATA_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    input  logic               start,
    input  logic [ADDR_W-1:0]  base_addr,   // NUM_BANKS-aligned
    input  logic [15:0]        num_rows,

    // To banked_scratchpad read port
    output logic               r_en,
    output logic               r_vec,
    output logic [ADDR_W-1:0]  r_addr,
    input  logic [VEC_W-1:0]   r_vdata,

    // Streamed output (one stripe per out_valid)
    output logic               out_valid,
    output logic [VEC_W-1:0]   out_data,
    output logic [15:0]        out_row,
    output logic               busy,
    output logic               done
);
    typedef enum logic [1:0] { S_IDLE, S_ISSUE, S_DRAIN } state_t;
    state_t state;

    logic [15:0]       issue_cnt;
    logic [15:0]       num_rows_l;
    logic [ADDR_W-1:0] base_l;

    // Combinational read issue
    assign r_en   = (state == S_ISSUE);
    assign r_vec  = 1'b1;
    assign r_addr = base_l + ADDR_W'(issue_cnt << LANE_LOG2);

    // Output data is the scratchpad's registered read result (aligned by the
    // 1-cycle valid pipeline below).
    assign out_data = r_vdata;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            issue_cnt  <= '0;
            num_rows_l <= '0;
            base_l     <= '0;
            out_valid  <= 1'b0;
            out_row    <= '0;
            busy       <= 1'b0;
            done       <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        base_l     <= base_addr;
                        num_rows_l <= num_rows;
                        issue_cnt  <= '0;
                        busy       <= 1'b1;
                        state      <= S_ISSUE;
                    end
                end

                S_ISSUE: begin
                    busy      <= 1'b1;
                    // a read is being issued this cycle; its data is valid next cycle
                    out_valid <= 1'b1;
                    out_row   <= issue_cnt;
                    issue_cnt <= issue_cnt + 16'd1;
                    if (issue_cnt == num_rows_l - 16'd1)
                        state <= S_DRAIN;
                end

                S_DRAIN: begin
                    // last issued stripe's data lands this cycle (out_valid still set
                    // from the final S_ISSUE cycle); then we are done.
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
