// ============================================================
//  banked_tile_loader.sv — fills Q/K/V tile registers from a banked_scratchpad
//  using NUM_BANKS-byte stripe reads (one stripe/cycle) instead of byte-serial.
//
//  Two modes:
//    MODE_Q  (0): read the Q scratchpad, emit q_stripe.
//    MODE_KV (1): read the K and V scratchpads concurrently (same address
//                 sequence), emit k_stripe and v_stripe together.
//
//  A tile is a contiguous flat block in matrix/scratchpad space, so loading is
//  a flat stripe counter:
//    NUM_STRIPES        = TILE_SIZE * HEAD_DIM / NUM_BANKS   (params, not HEAD_DIM)
//    scratchpad address = base_addr      + stripe_idx * NUM_BANKS   (full matrix)
//    tile-reg write idx = stripe_idx * NUM_BANKS                    (local tile)
//  These two are deliberately separate: Q/K/V live at a tile's global offset in
//  scratchpad space, but Q_reg/K_reg/V_reg are local tile buffers starting at 0.
//
//  The loader is layout-agnostic: it writes NUM_BANKS contiguous tile-register
//  elements per stripe, preserving row-major order. It knows NOTHING about K^T;
//  K stays row-major in K_reg and transposition remains in the slicing mux.
//
//  Output timing: the scratchpad has a 1-cycle read latency. `wr_en`/`wr_index`
//  are registered so they line up with `*_stripe` (= the scratchpad's registered
//  vector read result) on the cycle the data is valid. The consumer does:
//      if (wr_en) for (i=0; i<NUM_BANKS; i++)
//          reg[wr_index + i] <= signed'(stripe[i*8 +: 8]);
// ============================================================
`timescale 1ns/1ps

module banked_tile_loader #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 16,
    parameter int NUM_BANKS = 16,
    parameter int SP_ADDR_W = 12,                       // scratchpad byte-addr width
    localparam int TILE_ELEMS  = TILE_SIZE * HEAD_DIM,
    localparam int NUM_STRIPES = TILE_ELEMS / NUM_BANKS,
    localparam int VEC_W       = NUM_BANKS * 8,
    localparam int IDX_W       = $clog2(TILE_ELEMS)     // tile-register index width
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                 start,
    input  logic                 mode,        // 0 = Q load, 1 = KV load
    input  logic [SP_ADDR_W-1:0] base_addr,   // tile global base in scratchpad space

    // Banked-scratchpad vector read ports (1-cycle latency)
    output logic                 q_r_en,
    output logic [SP_ADDR_W-1:0] q_r_addr,
    input  logic [VEC_W-1:0]     q_r_vdata,
    output logic                 k_r_en,
    output logic [SP_ADDR_W-1:0] k_r_addr,
    input  logic [VEC_W-1:0]     k_r_vdata,
    output logic                 v_r_en,
    output logic [SP_ADDR_W-1:0] v_r_addr,
    input  logic [VEC_W-1:0]     v_r_vdata,

    // Tile-register write stream
    output logic             wr_en,
    output logic [IDX_W-1:0] wr_index,         // local tile index = stripe_idx*NUM_BANKS
    output logic [VEC_W-1:0] q_stripe,
    output logic [VEC_W-1:0] k_stripe,
    output logic [VEC_W-1:0] v_stripe,

    output logic busy,
    output logic done
);
    localparam logic MODE_Q  = 1'b0;
    localparam logic MODE_KV = 1'b1;

    typedef enum logic [1:0] { S_IDLE, S_ISSUE, S_DRAIN } state_t;
    state_t state;

    logic [15:0]           issue_cnt;
    logic                  mode_l;
    logic [SP_ADDR_W-1:0]  base_l;

    // Combinational read issue (one stripe per cycle while issuing)
    logic                 issuing;
    logic [SP_ADDR_W-1:0] cur_addr;
    assign issuing  = (state == S_ISSUE);
    assign cur_addr = base_l + SP_ADDR_W'(issue_cnt * NUM_BANKS);

    assign q_r_en   = issuing & (mode_l == MODE_Q);
    assign k_r_en   = issuing & (mode_l == MODE_KV);
    assign v_r_en   = issuing & (mode_l == MODE_KV);
    assign q_r_addr = cur_addr;
    assign k_r_addr = cur_addr;
    assign v_r_addr = cur_addr;

    // Stripe data is the scratchpad's registered read result (aligned to wr_en)
    assign q_stripe = q_r_vdata;
    assign k_stripe = k_r_vdata;
    assign v_stripe = v_r_vdata;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            issue_cnt <= '0;
            mode_l    <= MODE_Q;
            base_l    <= '0;
            wr_en     <= 1'b0;
            wr_index  <= '0;
            busy      <= 1'b0;
            done      <= 1'b0;
        end else begin
            wr_en <= 1'b0;
            done  <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        base_l    <= base_addr;
                        mode_l    <= mode;
                        issue_cnt <= '0;
                        busy      <= 1'b1;
                        state     <= S_ISSUE;
                    end
                end

                S_ISSUE: begin
                    busy      <= 1'b1;
                    // a stripe read is issued this cycle; its data is valid next
                    // cycle, so register the matching write strobe + index.
                    wr_en     <= 1'b1;
                    wr_index  <= IDX_W'(issue_cnt * NUM_BANKS);
                    issue_cnt <= issue_cnt + 16'd1;
                    if (issue_cnt == 16'(NUM_STRIPES - 1))
                        state <= S_DRAIN;
                end

                S_DRAIN: begin
                    // last stripe's data lands this cycle (wr_en still set from the
                    // final S_ISSUE cycle); then done.
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
