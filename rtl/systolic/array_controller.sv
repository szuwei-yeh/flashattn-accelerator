// ============================================================
//  array_controller.sv
//
//  Timing fix: COMPUTE_CYCLES = 3*SIZE-2
//
//  Push-in phase : cycle_cnt 0..SIZE-1     → feed real data
//  Drain phase   : cycle_cnt SIZE..3*SIZE-3 → feed zeros
//                  (data flows through shift-regs to far-corner PE)
// ============================================================
`timescale 1ns/1ps

module array_controller #(
    parameter int SIZE = 16
)(
    input  logic        clk,
    input  logic        rst_n,

    input  logic signed [7:0] a_flat [SIZE*SIZE-1:0],
    input  logic signed [7:0] b_flat [SIZE*SIZE-1:0],

    input  logic        start,
    output logic        busy,
    output logic        done,

    output logic signed [31:0] acc [SIZE*SIZE-1:0]
);

    logic signed [7:0] a_sr [SIZE*SIZE-1:0];
    logic signed [7:0] b_sr [SIZE*SIZE-1:0];
    logic signed [7:0] a_skewed [SIZE-1:0];
    logic signed [7:0] b_skewed [SIZE-1:0];

    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        CLEAR   = 2'b01,
        COMPUTE = 2'b10,
        FINISH  = 2'b11
    } state_t;

    state_t     state;
    logic [5:0] cycle_cnt;
    logic [4:0] feed_col;

    localparam int COMPUTE_CYCLES = 3 * SIZE - 1;

    logic en_sr;
    logic clear_pe;
    logic in_push_phase;

    assign en_sr         = (state == COMPUTE) || (state == CLEAR);
    assign clear_pe      = (state == CLEAR);
    // in_push_phase: true during the first SIZE cycles of COMPUTE
    assign in_push_phase = (state == COMPUTE) && (cycle_cnt < 6'(SIZE));

    // ── FSM ─────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            cycle_cnt <= '0;
            feed_col  <= '0;
            busy      <= '0;
            done      <= '0;
        end else begin
            done <= '0;
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= CLEAR;
                        busy  <= '1;
                    end
                end
                CLEAR: begin
                    cycle_cnt <= '0;
                    feed_col  <= '0;
                    state     <= COMPUTE;
                end
                COMPUTE: begin
                    
                    if (in_push_phase && feed_col < 5'(SIZE - 1))
                        feed_col <= feed_col + 1;

                    if (cycle_cnt == 6'(COMPUTE_CYCLES - 1))
                        state <= FINISH;
                    else
                        cycle_cnt <= cycle_cnt + 1;
                end
                FINISH: begin
                    done  <= '1;
                    busy  <= '0;
                    state <= IDLE;
                end
            endcase
        end
    end

    // ── Shift registers ──────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < SIZE*SIZE; i++) begin
                a_sr[i] <= '0;
                b_sr[i] <= '0;
            end
        end else if (en_sr) begin
            for (int row = 0; row < SIZE; row++) begin
                
                if (clear_pe || !in_push_phase) begin
                    a_sr[row*SIZE + 0] <= 8'sd0;
                    b_sr[row*SIZE + 0] <= 8'sd0;
                end else begin
                    a_sr[row*SIZE + 0] <= a_flat[row*SIZE + int'(feed_col)];
                    b_sr[row*SIZE + 0] <= b_flat[int'(feed_col)*SIZE + row];
                end
                
                // Stages 1..SIZE-1: always shift (zero on clear)
                for (int s = 1; s < SIZE; s++) begin
                    a_sr[row*SIZE + s] <= clear_pe ? 8'sd0 : a_sr[row*SIZE + s - 1];
                    b_sr[row*SIZE + s] <= clear_pe ? 8'sd0 : b_sr[row*SIZE + s - 1];
                end
            end
        end
    end

    // Tap: row i taps delay stage i
    genvar gi;
    generate
        for (gi = 0; gi < SIZE; gi++) begin : tap
            assign a_skewed[gi] = a_sr[gi*SIZE + gi];
            assign b_skewed[gi] = b_sr[gi*SIZE + gi];
        end
    endgenerate

    // ── Systolic array ───────────────────────────────────────
    systolic_array #(.SIZE(SIZE)) u_array (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en_sr),
        .clear (clear_pe),
        .a_in  (a_skewed),
        .b_in  (b_skewed),
        .acc   (acc)
    );

endmodule
