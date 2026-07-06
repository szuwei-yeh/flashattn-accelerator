// ============================================================
//  axi_mem_model.sv — Behavioral AXI4-slave DRAM model (SIMULATION ONLY)
//
//  Models off-chip DDR/HBM behind the AXI4 read channels (AR/R).  A read
//  burst incurs a programmable `rd_latency` (row-activation latency) before
//  the first beat, then streams INCR beats back-to-back (1 beat / cycle while
//  RREADY is high).  RREADY may be throttled by the master — RVALID/RDATA are
//  held stable until the handshake completes.
//
//  Backing store is loaded out-of-band through the backdoor write port
//  (init_we/init_addr/init_data) by the testbench during setup.
//
//  NOT synthesizable / not on the accelerator's synthesis path — this is the
//  external memory the DMA talks to.
// ============================================================
`timescale 1ns/1ps

module axi_mem_model #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int DEPTH      = 65536,        // bytes of DRAM modeled
    localparam int BYTES_PER_BEAT = AXI_DATA_W / 8,
    localparam int ADDR_W         = $clog2(DEPTH)
)(
    input  logic clk,
    input  logic rst_n,

    // Programmable first-beat read latency (cycles)
    input  logic [15:0] rd_latency,

    // ── Backdoor load port (testbench preloads DRAM image) ────────────
    input  logic                  init_we,
    /* verilator lint_off UNUSEDSIGNAL */     // only low ADDR_W bits indexed
    input  logic [AXI_ADDR_W-1:0] init_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [7:0]            init_data,

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
    input  logic                  s_rready
);

    logic [7:0] mem [0:DEPTH-1];

    // Backdoor preload
    always_ff @(posedge clk) begin
        if (init_we) mem[init_addr[ADDR_W-1:0]] <= init_data;
    end

    typedef enum logic [1:0] { S_IDLE, S_LAT, S_DATA } state_t;
    state_t state;

    logic [ADDR_W-1:0] raddr_ptr;
    logic [8:0]        beats_left;     // up to 256
    logic [15:0]       lat_cnt;

    // Combinational read data: pack BYTES_PER_BEAT consecutive bytes / beat
    always_comb begin
        for (int b = 0; b < BYTES_PER_BEAT; b++)
            s_rdata[8*b +: 8] = mem[(raddr_ptr + ADDR_W'(b)) & ADDR_W'(DEPTH-1)];
    end

    assign s_rresp   = 2'b00;                       // OKAY
    assign s_arready = (state == S_IDLE);
    assign s_rvalid  = (state == S_DATA);
    assign s_rlast   = (state == S_DATA) && (beats_left == 9'd1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            raddr_ptr  <= '0;
            beats_left <= '0;
            lat_cnt    <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (s_arvalid) begin
                        raddr_ptr  <= s_araddr[ADDR_W-1:0];
                        beats_left <= 9'(s_arlen) + 9'd1;
                        lat_cnt    <= rd_latency;
                        state      <= (rd_latency == 16'd0) ? S_DATA : S_LAT;
                    end
                end

                S_LAT: begin
                    if (lat_cnt <= 16'd1) state <= S_DATA;
                    else                  lat_cnt <= lat_cnt - 16'd1;
                end

                S_DATA: begin
                    if (s_rready) begin
                        raddr_ptr  <= raddr_ptr + ADDR_W'(BYTES_PER_BEAT);
                        beats_left <= beats_left - 9'd1;
                        if (beats_left == 9'd1) state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
