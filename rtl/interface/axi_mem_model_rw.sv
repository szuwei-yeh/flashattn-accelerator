// ============================================================
//  axi_mem_model_rw.sv — Behavioral AXI4-slave DRAM model, READ+WRITE (SIM ONLY)
//
//  Additive superset of axi_mem_model.sv: the AR/R read side is byte-for-byte
//  identical (same programmable `rd_latency`, same INCR streaming, same backdoor
//  `init_*` preload), and this file ADDS the AXI4 write channels (AW/W/B) so a
//  DMA write master can stream results back into the same byte-addressed store.
//
//  Read side (unchanged from axi_mem_model.sv):
//    - AR/R with a programmable first-beat `rd_latency`, INCR beats streamed
//      back-to-back while RREADY is high.
//  Write side (new):
//    - AW/W/B, single outstanding write burst, INCR assumed.
//    - Full byte-addressed backing store; WSTRB honoured per byte lane.
//    - Programmable `wr_latency` inserted before the B (write-response) beat,
//      modelling write-completion latency (0 = respond immediately after WLAST).
//    - W data is accepted at full rate (WREADY high in the data phase); the
//      exposed latency shows up on the B channel.
//
//  Backdoor ports for the testbench: `init_*` preloads (write) DRAM; `bd_raddr`/
//  `bd_rdata` reads it back combinationally for byte-exact checking.
//
//  NOT synthesizable / not on the accelerator's synthesis path — this is the
//  external memory the DMA talks to.
// ============================================================
`timescale 1ns/1ps

module axi_mem_model_rw #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DEPTH      = 65536,        // bytes of DRAM modeled
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8,
    localparam int ADDR_W         = $clog2(DEPTH)
)(
    input  logic clk,
    input  logic rst_n,

    // Programmable channel latencies (cycles)
    input  logic [15:0] rd_latency,
    input  logic [15:0] wr_latency,

    // ── Backdoor load port (testbench preloads DRAM image) ────────────
    input  logic                  init_we,
    /* verilator lint_off UNUSEDSIGNAL */     // only low ADDR_W bits indexed
    input  logic [AXI_ADDR_W-1:0] init_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [7:0]            init_data,

    // ── Backdoor read port (testbench checks written bytes) ───────────
    /* verilator lint_off UNUSEDSIGNAL */     // only low ADDR_W bits indexed
    input  logic [AXI_ADDR_W-1:0] bd_raddr,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [7:0]            bd_rdata,

    // ── AXI4 read address channel (slave) ─────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */     // low ADDR_W addr bits; AR meta unused
    input  logic [AXI_ADDR_W-1:0] s_araddr,
    input  logic [7:0]            s_arlen,
    input  logic [2:0]            s_arsize,    // ignored (BYTES_PER_BEAT fixed)
    input  logic [1:0]            s_arburst,   // INCR assumed
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                  s_arvalid,
    output logic                  s_arready,

    // ── AXI4 read data channel (slave) ───────────────────────────────
    output logic [AXI_DATA_W-1:0] s_rdata,
    output logic [1:0]            s_rresp,
    output logic                  s_rlast,
    output logic                  s_rvalid,
    input  logic                  s_rready,

    // ── AXI4 write address channel (slave) ────────────────────────────
    /* verilator lint_off UNUSEDSIGNAL */     // low ADDR_W addr bits; AW meta unused
    input  logic [AXI_ADDR_W-1:0] s_awaddr,
    input  logic [7:0]            s_awlen,
    input  logic [2:0]            s_awsize,    // ignored (BYTES_PER_BEAT fixed)
    input  logic [1:0]            s_awburst,   // INCR assumed
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                  s_awvalid,
    output logic                  s_awready,

    // ── AXI4 write data channel (slave) ──────────────────────────────
    input  logic [AXI_DATA_W-1:0]   s_wdata,
    input  logic [BYTES_PER_BEAT-1:0] s_wstrb,
    /* verilator lint_off UNUSEDSIGNAL */     // beat count tracked locally
    input  logic                    s_wlast,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                    s_wvalid,
    output logic                    s_wready,

    // ── AXI4 write response channel (slave) ──────────────────────────
    output logic [1:0]            s_bresp,
    output logic                  s_bvalid,
    input  logic                  s_bready
);

    logic [7:0] mem [0:DEPTH-1];

    // Backdoor combinational read (testbench checking)
    assign bd_rdata = mem[bd_raddr[ADDR_W-1:0]];

    // =====================================================================
    //  READ side (AR/R) — identical to axi_mem_model.sv
    // =====================================================================
    typedef enum logic [1:0] { S_IDLE, S_LAT, S_DATA } rstate_t;
    rstate_t rstate;

    logic [ADDR_W-1:0] raddr_ptr;
    logic [8:0]        rbeats_left;    // up to 256
    logic [15:0]       rlat_cnt;

    // Combinational read data: pack BYTES_PER_BEAT consecutive bytes / beat
    always_comb begin
        for (int b = 0; b < BYTES_PER_BEAT; b++)
            s_rdata[8*b +: 8] = mem[(raddr_ptr + ADDR_W'(b)) & ADDR_W'(DEPTH-1)];
    end

    assign s_rresp   = 2'b00;                       // OKAY
    assign s_arready = (rstate == S_IDLE);
    assign s_rvalid  = (rstate == S_DATA);
    assign s_rlast   = (rstate == S_DATA) && (rbeats_left == 9'd1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate      <= S_IDLE;
            raddr_ptr   <= '0;
            rbeats_left <= '0;
            rlat_cnt    <= '0;
        end else begin
            case (rstate)
                S_IDLE: begin
                    if (s_arvalid) begin
                        raddr_ptr   <= s_araddr[ADDR_W-1:0];
                        rbeats_left <= 9'(s_arlen) + 9'd1;
                        rlat_cnt    <= rd_latency;
                        rstate      <= (rd_latency == 16'd0) ? S_DATA : S_LAT;
                    end
                end

                S_LAT: begin
                    if (rlat_cnt <= 16'd1) rstate <= S_DATA;
                    else                   rlat_cnt <= rlat_cnt - 16'd1;
                end

                S_DATA: begin
                    if (s_rready) begin
                        raddr_ptr   <= raddr_ptr + ADDR_W'(BYTES_PER_BEAT);
                        rbeats_left <= rbeats_left - 9'd1;
                        if (rbeats_left == 9'd1) rstate <= S_IDLE;
                    end
                end

                default: rstate <= S_IDLE;
            endcase
        end
    end

    // =====================================================================
    //  WRITE side (AW/W/B) — new
    // =====================================================================
    typedef enum logic [1:0] { W_IDLE, W_DATA, W_LAT, W_RESP } wstate_t;
    wstate_t wstate;

    logic [ADDR_W-1:0] waddr_ptr;
    logic [8:0]        wbeats_left;
    logic [15:0]       wlat_cnt;

    assign s_awready = (wstate == W_IDLE);
    assign s_wready  = (wstate == W_DATA);          // full-rate accept
    assign s_bvalid  = (wstate == W_RESP);
    assign s_bresp   = 2'b00;                        // OKAY

    logic w_fire;
    assign w_fire = (wstate == W_DATA) && s_wvalid;

    // Backing-store writes: backdoor init has priority, else an accepted W beat
    // writes BYTES_PER_BEAT lanes (WSTRB-gated) at the current write pointer.
    always_ff @(posedge clk) begin
        if (init_we) begin
            mem[init_addr[ADDR_W-1:0]] <= init_data;
        end else if (w_fire) begin
            for (int b = 0; b < BYTES_PER_BEAT; b++)
                if (s_wstrb[b])
                    mem[(waddr_ptr + ADDR_W'(b)) & ADDR_W'(DEPTH-1)] <= s_wdata[8*b +: 8];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate      <= W_IDLE;
            waddr_ptr   <= '0;
            wbeats_left <= '0;
            wlat_cnt    <= '0;
        end else begin
            case (wstate)
                W_IDLE: begin
                    if (s_awvalid) begin
                        waddr_ptr   <= s_awaddr[ADDR_W-1:0];
                        wbeats_left <= 9'(s_awlen) + 9'd1;
                        wstate      <= W_DATA;
                    end
                end

                W_DATA: begin
                    if (s_wvalid) begin
                        waddr_ptr   <= waddr_ptr + ADDR_W'(BYTES_PER_BEAT);
                        wbeats_left <= wbeats_left - 9'd1;
                        if (wbeats_left == 9'd1) begin      // last beat of burst
                            wlat_cnt <= wr_latency;
                            wstate   <= (wr_latency == 16'd0) ? W_RESP : W_LAT;
                        end
                    end
                end

                W_LAT: begin
                    if (wlat_cnt <= 16'd1) wstate <= W_RESP;
                    else                   wlat_cnt <= wlat_cnt - 16'd1;
                end

                W_RESP: begin
                    if (s_bready) wstate <= W_IDLE;
                end

                default: wstate <= W_IDLE;
            endcase
        end
    end

    // synthesis translate_off
    // AXI requires WLAST on the final beat of the burst; check the master obeys.
    always_ff @(posedge clk) begin
        if (w_fire && (wbeats_left == 9'd1) && !s_wlast)
            $error("axi_mem_model_rw: WLAST not asserted on final burst beat");
        if (w_fire && (wbeats_left != 9'd1) && s_wlast)
            $error("axi_mem_model_rw: WLAST asserted early (beats_left=%0d)", wbeats_left);
    end
    // synthesis translate_on

endmodule
