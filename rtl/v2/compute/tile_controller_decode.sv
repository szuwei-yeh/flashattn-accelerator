// ============================================================
//  Tile/phase controller for foreground-loaded d16/d64 attention.
//
//  The banked tile loader is reusable; this V2 controller selects
//  foreground-only operation and never launches KV prefetch.
//  Q and paired K/V loads share that loader and cannot overlap.
//
//  Foreground loading:
//    - Load Q before processing the query pass.
//    - Wait for complete resident K/V pairs before loading a KV tile.
//    - Reload K/V when advancing to the next eligible tile.
//    - Skip above-diagonal tiles when causal masking permits.
//
//  Shared-array compute sequence:
//    - Accumulate all QK feature chunks before publishing scores.
//    - Wait for softmax weights before starting PV.
//    - Follow each PV output-feature chunk with a fused output update.
//    - Normalize output chunks after the final KV tile.
//
//  Retained prefetch interface (inactive in V2):
//    - pf_start would request a next-tile load into shadow registers.
//    - pf_rdy indicates a complete shadow tile.
//    - kv_swap_banks would copy shadow K/V into active registers.
//    - The launch guard is constant false; pf_pending stays clear.
//    - Prefetch-consume/wait states are retained but not entered.
//    - Simulation guards reject prefetch or bank-swap activity.
//
//  Fused output update:
//    - Rescale prior output and add PV in one SRAM traversal.
//    - Preserve rescale truncation before adding scaled PV values.
//    - Completion must allow registered output writes to retire.
// ============================================================
`timescale 1ns/1ps

/* verilator lint_off UNUSEDPARAM */
module tile_controller_decode_m2c3 #(
    parameter int TILE_SIZE = 16,
    parameter int HEAD_DIM  = 64,
    parameter int SEQ_LEN   = 64,
    localparam int NUM_CHUNKS = HEAD_DIM / TILE_SIZE
)(
/* verilator lint_on UNUSEDPARAM */
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    output logic start_accepted,  // high only while S_IDLE accepts start

    input  logic        mode,       // 0 = prefill, 1 = decode
    input  logic [15:0] kv_len,     // runtime KV length (decode mode)

    output logic [15:0] tile_row,
    output logic [15:0] tile_col,
    output logic        cnt_en,
    output logic        cnt_clr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        cnt_done,
    /* verilator lint_on UNUSEDSIGNAL */

    // Foreground stripe-loader handshake (Q load, first KV load, non-prefetched
    // reloads). Identical to tile_controller_banked.sv.
    output logic        ld_start,   // 1-cycle pulse to start a foreground tile load
    output logic        ld_mode,    // 0 = Q load, 1 = KV load
    input  logic        ld_done,    // FOREGROUND loader finished (core de-muxes)

    // Retained KV prefetch handshake; inactive in this controller.
    output logic        pf_start,       // 1-cycle pulse: prefetch next KV tile → shadow
    output logic [15:0] pf_tile_col,    // tile_col of the tile being prefetched
    input  logic        pf_rdy,         // shadow holds a complete next tile
    output logic        kv_swap_banks,  // 1-cycle pulse: copy shadow → active

    output logic        array_start,
    output logic        array_no_clear,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        array_done,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        softmax_tile_start,
    output logic        softmax_tile_valid,
    output logic        softmax_tile_last,
    input  logic        exp_out_valid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic        softmax_out_valid,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        fused_update_en,
    output logic        norm_en,
    output logic        short_cnt_mode,
    output logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] k_chunk,
    output logic        pv_done,

    input  logic        causal,

    // DMA streaming interlock (same semantics as tile_controller_banked). Tie to
    // 16'hFFFF when K/V is fully preloaded.
    input  logic [15:0] kv_tiles_ready,

    output logic [3:0]  dbg_state
);

    localparam logic MODE_Q  = 1'b0;
    localparam logic MODE_KV = 1'b1;

    typedef enum logic [3:0] {
        S_IDLE             = 4'd0,
        S_LOAD_Q           = 4'd1,
        S_LOAD_KV          = 4'd2,
        S_MATMUL_QK        = 4'd3,
        S_UPDATE_SOFTMAX   = 4'd4,
        S_MATMUL_PV        = 4'd6,
        S_FUSED_UPDATE     = 4'd7,
        S_CHECK_INNER      = 4'd8,
        S_NORMALIZE        = 4'd9,
        S_CHECK_OUTER      = 4'd10,
        S_DONE             = 4'd11,
        S_PF_WAIT          = 4'd12    // NEW: wait for shadow prefetch, then swap
    } state_t;

    state_t state;
    logic   array_started;
    logic   load_started;     // one-shot guard for foreground ld_start
    logic   pf_pending;       // a prefetch was launched, not yet consumed by swap
    logic [$clog2(NUM_CHUNKS > 1 ? NUM_CHUNKS : 2)-1:0] chunk_cnt;

    assign dbg_state = 4'(state);
    assign start_accepted = (state == S_IDLE) && start;

    logic is_first_kv;
    logic is_last_kv;
    logic [15:0] effective_kv_len;
    logic [15:0] next_tile_col;
    logic        next_tile_valid;      // next tile exists and is not above-diagonal

    localparam int LOG2_T = $clog2(TILE_SIZE);
    logic        tile_resident;
    logic        next_tile_resident;

    assign is_first_kv        = (tile_col == 16'b0);
    assign effective_kv_len   = mode ? kv_len : 16'(SEQ_LEN);
    assign is_last_kv         = ((tile_col + 16'(TILE_SIZE)) >= effective_kv_len);
    assign next_tile_col      = tile_col + 16'(TILE_SIZE);
    // Valid to prefetch: there is a next tile AND it is not above the diagonal.
    assign next_tile_valid    = !is_last_kv && (!causal || next_tile_col <= tile_row);
    // Residency: a tile is consumable once its index < the resident count.
    assign tile_resident      = ((tile_col      >> LOG2_T) < kv_tiles_ready);
    assign next_tile_resident = ((next_tile_col >> LOG2_T) < kv_tiles_ready);
    // The tile currently being prefetched is always the next tile column.
    assign pf_tile_col        = next_tile_col;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            tile_row           <= '0;
            tile_col           <= '0;
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            ld_start           <= 1'b0;
            ld_mode            <= MODE_Q;
            load_started       <= 1'b0;
            pf_start           <= 1'b0;
            pf_pending         <= 1'b0;
            kv_swap_banks      <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            array_started      <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            fused_update_en    <= 1'b0;
            norm_en            <= 1'b0;
            short_cnt_mode     <= 1'b0;
            k_chunk            <= '0;
            pv_done            <= 1'b0;
            chunk_cnt          <= '0;
        end else begin
            // Default: de-assert all pulses every cycle
            done               <= 1'b0;
            cnt_en             <= 1'b0;
            cnt_clr            <= 1'b0;
            ld_start           <= 1'b0;
            pf_start           <= 1'b0;
            kv_swap_banks      <= 1'b0;
            array_start        <= 1'b0;
            array_no_clear     <= 1'b0;
            softmax_tile_start <= 1'b0;
            softmax_tile_valid <= 1'b0;
            softmax_tile_last  <= 1'b0;
            fused_update_en    <= 1'b0;
            norm_en            <= 1'b0;
            pv_done            <= 1'b0;
            // short_cnt_mode, k_chunk, ld_mode, pf_pending are level signals — hold

            case (state)

                S_IDLE: begin
                    short_cnt_mode <= 1'b0;
                    pf_pending     <= 1'b0;    // no prefetch outstanding at start
                    if (start_accepted) begin
                        tile_row      <= '0;
                        tile_col      <= '0;
                        array_started <= 1'b0;
                        load_started  <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        cnt_clr       <= 1'b1;
                        state         <= S_LOAD_Q;
                    end
                end

                // Stripe-load the Q tile (16 bytes/cycle) — foreground load.
                S_LOAD_Q: begin
                    short_cnt_mode <= 1'b0;
                    if (!load_started) begin
                        ld_start     <= 1'b1;
                        ld_mode      <= MODE_Q;
                        load_started <= 1'b1;
                    end
                    if (ld_done) begin
                        load_started <= 1'b0;
                        cnt_clr      <= 1'b1;
                        state        <= S_LOAD_KV;
                    end
                end

                // Stripe-load the KV tile (K and V concurrently) — foreground load.
                // Residency gate: stall until the tile is resident.  Reached only
                // when NO prefetch supplied this tile (pf_pending was 0).
                S_LOAD_KV: begin
                    short_cnt_mode <= 1'b0;
                    if (tile_resident && !load_started) begin
                        ld_start     <= 1'b1;
                        ld_mode      <= MODE_KV;
                        load_started <= 1'b1;
                    end
                    if (ld_done) begin
                        load_started  <= 1'b0;
                        cnt_clr       <= 1'b1;
                        array_started <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        state         <= S_MATMUL_QK;
                    end
                end

                S_MATMUL_QK: begin
                    if (!array_started) begin
                        array_no_clear <= (chunk_cnt != '0);
                        array_start    <= 1'b1;
                        array_started  <= 1'b1;
                    end
                    if (array_done) begin
                        array_started <= 1'b0;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_MATMUL_QK;
                        end else begin
                            chunk_cnt <= '0;
                            k_chunk   <= '0;
                            state     <= S_UPDATE_SOFTMAX;
                        end
                    end
                end

                // Publish scores to softmax; start PV when weights are ready.
                // The retained prefetch launch remains disabled.
                S_UPDATE_SOFTMAX: begin
                    softmax_tile_valid <= 1'b1;
                    softmax_tile_start <= is_first_kv;
                    softmax_tile_last  <= is_last_kv;
                    if (exp_out_valid) begin
                        softmax_tile_valid <= 1'b0;
                        array_started      <= 1'b0;
                        cnt_clr            <= 1'b1;
                        // Retained residency-qualified prefetch branch.
                        // The constant-false guard keeps V2 foreground-only.
                        if (1'b0 && next_tile_valid && next_tile_resident) begin // foreground only
                            pf_start   <= 1'b1;
                            pf_pending <= 1'b1;
                        end
                        state              <= S_MATMUL_PV;
                    end
                end

                S_MATMUL_PV: begin
                    if (!array_started) begin
                        array_start   <= 1'b1;
                        array_started <= 1'b1;
                    end
                    if (array_done) begin
                        array_started  <= 1'b0;
                        short_cnt_mode <= 1'b1;
                        cnt_clr        <= 1'b1;
                        state          <= S_FUSED_UPDATE;
                    end
                end

                S_FUSED_UPDATE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    fused_update_en <= 1'b1;
                    if (cnt_done) begin
                        cnt_en          <= 1'b0;
                        fused_update_en <= 1'b0;
                        cnt_clr         <= 1'b1;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_MATMUL_PV;
                        end else begin
                            pv_done        <= 1'b1;
                            short_cnt_mode <= 1'b0;
                            chunk_cnt      <= '0;
                            k_chunk        <= '0;
                            state          <= S_CHECK_INNER;
                        end
                    end
                end

                // Advance to the next inner tile.  Three mutually-exclusive cases:
                //   (a) causal above-diagonal → spin (skip), no prefetch was fired;
                //   (b) prefetch pending      → wait for shadow, then swap;
                //   (c) no prefetch           → foreground reload via S_LOAD_KV.
                // tile_col is advanced exactly ONCE here; S_PF_WAIT never re-advances.
                S_CHECK_INNER: begin
                    if (!is_last_kv) begin
                        tile_col <= next_tile_col;
                        cnt_clr  <= 1'b1;
                        if (causal && next_tile_col > tile_row) begin
                            state <= S_CHECK_INNER;   // skip above-diagonal tile
                        end else if (pf_pending) begin
                            state <= S_PF_WAIT;       // consume the prefetch
                        end else begin
                            state <= S_LOAD_KV;       // no prefetch: normal reload
                        end
                    end else begin
                        cnt_clr <= 1'b1;
                        state   <= S_NORMALIZE;
                    end
                end

                // Wait for the shadow prefetch to complete, then swap shadow→active
                // and skip S_LOAD_KV.  tile_col was already advanced in S_CHECK_INNER;
                // staying here does not change it (correct stall).
                S_PF_WAIT: begin
                    if (pf_rdy) begin
                        kv_swap_banks <= 1'b1;
                        pf_pending    <= 1'b0;
                        array_started <= 1'b0;
                        chunk_cnt     <= '0;
                        k_chunk       <= '0;
                        state         <= S_MATMUL_QK;
                    end
                    // else: stay in S_PF_WAIT until the shadow buffer is ready
                end

                S_NORMALIZE: begin
                    short_cnt_mode <= 1'b1;
                    cnt_en         <= 1'b1;
                    norm_en        <= 1'b1;
                    if (cnt_done) begin
                        cnt_en  <= 1'b0;
                        norm_en <= 1'b0;
                        cnt_clr <= 1'b1;
                        if (int'(chunk_cnt) < NUM_CHUNKS - 1) begin
                            chunk_cnt <= chunk_cnt + 1'b1;
                            k_chunk   <= chunk_cnt + 1'b1;
                            state     <= S_NORMALIZE;
                        end else begin
                            short_cnt_mode <= 1'b0;
                            chunk_cnt      <= '0;
                            k_chunk        <= '0;
                            state          <= S_CHECK_OUTER;
                        end
                    end
                end

                S_CHECK_OUTER: begin
                    if (!mode && ((tile_row + 16'(TILE_SIZE)) < 16'(SEQ_LEN))) begin
                        tile_row <= tile_row + 16'(TILE_SIZE);
                        tile_col <= '0;
                        cnt_clr  <= 1'b1;
                        state    <= S_LOAD_Q;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
