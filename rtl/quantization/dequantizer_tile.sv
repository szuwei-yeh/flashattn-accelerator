// Time-share unchanged full-precision dequantizers over a 16x16 score tile.
// valid_in is a one-cycle request while !busy. data_in and combined_scale must
// remain stable until busy clears (the core retains QK accumulators until PV).
// Scores are published atomically via tile_ready after every batch retires.
`timescale 1ns/1ps
module dequantizer_tile #(
    parameter int LANES = 32,
    localparam int ELEMENTS = 256,
    localparam int GROUPS = ELEMENTS / LANES,
    localparam int GROUP_W = $clog2(GROUPS)
)(
    input logic clk,
    input logic rst_n,
    input logic valid_in,
    input logic signed [31:0] data_in [ELEMENTS-1:0],
    input logic signed [31:0] combined_scale,
    output logic signed [15:0] data_out [ELEMENTS-1:0],
    output logic busy,
    output logic done,
    output logic tile_ready
);
    initial begin
        if (!(LANES == 16 || LANES == 32))
            $fatal(1, "dequantizer_tile: LANES must be 16 or 32");
    end

    logic issue_active, retire_valid;
    logic [GROUP_W-1:0] issue_group, retire_group;
    logic signed [15:0] lane_result [LANES-1:0];
    /* verilator lint_off UNUSEDSIGNAL */
    logic [LANES-1:0] lane_valid;
    /* verilator lint_on UNUSEDSIGNAL */

    for (genvar lane = 0; lane < LANES; lane++) begin : gen_lane
        dequantizer u_deq (
            .clk(clk), .rst_n(rst_n), .valid_in(issue_active),
            .data_in(data_in[int'(issue_group)*LANES + lane]),
            .combined_scale(combined_scale),
            .valid_out(lane_valid[lane]), .data_out(lane_result[lane])
        );
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            tile_ready <= 1'b0;
            issue_active <= 1'b0;
            retire_valid <= 1'b0;
            issue_group <= '0;
            retire_group <= '0;
        end else begin
            done <= 1'b0;
            retire_valid <= issue_active;
            retire_group <= issue_group;
            if (valid_in && !busy) begin
                busy <= 1'b1;
                tile_ready <= 1'b0;
                issue_active <= 1'b1;
                issue_group <= '0;
            end else if (issue_active) begin
                if (issue_group == GROUP_W'(GROUPS-1))
                    issue_active <= 1'b0;
                else
                    issue_group <= issue_group + 1'b1;
            end
            if (retire_valid) begin
                for (int lane = 0; lane < LANES; lane++)
                    data_out[int'(retire_group)*LANES + lane] <= lane_result[lane];
                if (retire_group == GROUP_W'(GROUPS-1)) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    tile_ready <= 1'b1;
                end
            end
        end
    end

    // synthesis translate_off
    always @(posedge clk or negedge rst_n) begin
        if (rst_n && valid_in && busy)
            $fatal(1, "dequantizer_tile: overlapping tile requests");
    end
    // synthesis translate_on
endmodule
