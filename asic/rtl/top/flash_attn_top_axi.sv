// ============================================================
//  flash_attn_top_axi.sv — FlashAttention 4-head AXI wrapper (with GQA)
//
//  4 parallel flash_attn_core instances (one per Q-head) + AXI4-Stream.
//
//  GQA (Grouped-Query Attention):
//    GQA_RATIO = 1 → MHA: each head has its own K/V (default)
//    GQA_RATIO = 2 → GQA: 4 Q-heads, 2 KV-heads (LLaMA 2/3 style)
//      Q-heads 0,1 share KV-head 0; Q-heads 2,3 share KV-head 1.
//    The K/V AXI stream is NUM_KV_HEADS heads wide (halved for GQA_RATIO=2),
//    reducing KV cache bandwidth by GQA_RATIO.
//
//  Parameters:
//    SEQ_LEN      = sequence length (e.g. 64)
//    HEAD_DIM     = total Q head dimension (e.g. 64 = 4 heads × 16)
//    NUM_HEADS    = number of Q-heads (4)
//    GQA_RATIO    = Q-heads per KV-head (1=MHA, 2=GQA, 4=MQA)
//    TILE_SIZE    = systolic array size (16)
//    SRAM_DEPTH   = SRAM depth per core (4096 supports up to SEQ_LEN=256)
//    PER_HEAD_DIM = HEAD_DIM / NUM_HEADS (= 16)
// ============================================================
`timescale 1ns/1ps

module flash_attn_top_axi #(
    parameter int SEQ_LEN      = 64,
    parameter int HEAD_DIM     = 64,
    parameter int NUM_HEADS    = 4,
    parameter int GQA_RATIO    = 1,                     // 1=MHA, 2=GQA, 4=MQA
    parameter int TILE_SIZE    = 16,
    parameter int SRAM_DEPTH   = 4096,
    parameter int PER_HEAD_DIM = HEAD_DIM / NUM_HEADS,  // = 16
    localparam int NUM_KV_HEADS = NUM_HEADS / GQA_RATIO // = 2 for GQA_RATIO=2
)(
    input  logic clk,
    input  logic rst_n,

    // AXI4-Stream slave — Q, K, V input (8-bit / INT8)
    input  logic [7:0]  s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,

    // Quantization scales (shared across all heads)
    input  logic signed [15:0] scale_q,
    input  logic signed [15:0] scale_k,
    input  logic signed [15:0] scale_v,

    // AXI4-Stream master — attention output (32-bit / INT32)
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,

    output logic done   // pulses high for 1 cycle when all 4 cores finish
);

    // ====================================================================
    // AXI4-Stream slave — loads Q/K/V into per-head SRAMs
    // ====================================================================
    logic        slv_we;
    logic [11:0] slv_waddr;
    logic [7:0]  slv_wdata;
    logic [1:0]  slv_head_sel;
    logic [1:0]  slv_mat_sel;
    logic        load_done;

    axi4_stream_slave #(
        .SEQ_LEN(SEQ_LEN),
        .HEAD_DIM(HEAD_DIM),
        .NUM_HEADS(NUM_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS),  // GQA: K/V stream is NUM_KV_HEADS-wide
        .PER_HEAD_DIM(PER_HEAD_DIM)
    ) u_slave (
        .clk(clk), .rst_n(rst_n),
        .s_tdata(s_axis_tdata), .s_tvalid(s_axis_tvalid),
        .s_tready(s_axis_tready), .s_tlast(s_axis_tlast),
        .we(slv_we), .waddr(slv_waddr), .wdata(slv_wdata),
        .head_sel(slv_head_sel), .mat_sel(slv_mat_sel),
        .load_done(load_done)
    );

    // ====================================================================
    // Per-head write enable decode
    //   slv_head_sel cycles 0..NUM_HEADS-1 for Q, 0..NUM_KV_HEADS-1 for K/V.
    //   Q: each core gets its own Q data (head_sel == h).
    //   K/V: GQA groups — core h receives KV-head (h/GQA_RATIO).
    //        For GQA_RATIO=2: cores 0,1 share KV-head 0; cores 2,3 share KV-head 1.
    //        For GQA_RATIO=1 (MHA): each core gets its own K/V (head_sel == h).
    // ====================================================================
    logic [NUM_HEADS-1:0] core_q_we, core_k_we, core_v_we;

    always_comb begin
        for (int h = 0; h < NUM_HEADS; h++) begin
            core_q_we[h] = slv_we & (slv_head_sel == 2'(h))            & (slv_mat_sel == 2'd0);
            core_k_we[h] = slv_we & (slv_head_sel == 2'(h/GQA_RATIO)) & (slv_mat_sel == 2'd1);
            core_v_we[h] = slv_we & (slv_head_sel == 2'(h/GQA_RATIO)) & (slv_mat_sel == 2'd2);
        end
    end

    // ====================================================================
    // Rising-edge detect on load_done → 1-cycle cores_start pulse
    // ====================================================================
    logic load_done_r, cores_start;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) load_done_r <= 1'b0;
        else        load_done_r <= load_done;
    end
    assign cores_start = load_done & ~load_done_r;

    // ====================================================================
    // 4 × flash_attn_core instances
    // ====================================================================
    logic [NUM_HEADS-1:0] core_done;
    logic [11:0]          core_out_raddr [NUM_HEADS-1:0];
    logic signed [31:0]   core_out_rdata [NUM_HEADS-1:0];

    genvar h;
    generate
        for (h = 0; h < NUM_HEADS; h++) begin : g_cores
            /* verilator lint_off PINCONNECTEMPTY */
            flash_attn_core #(
                .SEQ_LEN(SEQ_LEN),
                .HEAD_DIM(PER_HEAD_DIM),    // each core processes one 16-dim head
                .TILE_SIZE(TILE_SIZE),
                .SRAM_DEPTH(SRAM_DEPTH)
            ) u_core (
                .clk(clk), .rst_n(rst_n),
                // Q/K/V write (from AXI slave, gated by head_sel)
                .q_we(core_q_we[h]),
                .q_waddr(slv_waddr),
                .q_wdata(slv_wdata),
                .k_we(core_k_we[h]),
                .k_waddr(slv_waddr),
                .k_wdata(slv_wdata),
                .v_we(core_v_we[h]),
                .v_waddr(slv_waddr),
                .v_wdata(slv_wdata),
                // Scales (shared across all heads)
                .scale_q(scale_q),
                .scale_k(scale_k),
                .scale_v(scale_v),
                // Control — prefill-only mode (mode=0, kv_len unused, causal=0)
                .start(cores_start),
                .done(core_done[h]),
                .mode(1'b0),
                .kv_len(16'b0),
                .causal(1'b0),
                // KV cache ports — not used in AXI top (tied off)
                .kc_write_en(1'b0),
                .kc_write_ptr(8'b0),
                .kc_k_flat('0),
                .kc_v_flat('0),
                .kc_read_addr(8'b0),
                .kc_k_out(),
                .kc_v_out(),
                .kc_cache_len(),
                // Output read (driven by AXI master)
                .out_raddr(core_out_raddr[h]),
                .out_rdata(core_out_rdata[h])
            );
            /* verilator lint_on PINCONNECTEMPTY */
        end
    endgenerate

    assign done = &core_done;  // all 4 cores done simultaneously

    // ====================================================================
    // Connect core output read ports to AXI master (flat ↔ unpacked)
    // ====================================================================
    logic [NUM_HEADS*12-1:0] mst_raddr_flat;
    logic [NUM_HEADS*32-1:0] mst_rdata_flat;

    generate
        for (h = 0; h < NUM_HEADS; h++) begin : g_mst_conn
            // Master drives addresses → cores read from SRAM
            assign core_out_raddr[h]           = mst_raddr_flat[h*12 +: 12];
            // Cores provide data → master streams it out
            assign mst_rdata_flat[h*32 +: 32]  = 32'(core_out_rdata[h]);
        end
    endgenerate

    // ====================================================================
    // Rising-edge detect on all_done → 1-cycle master_start pulse
    // ====================================================================
    logic all_done_r, master_start;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) all_done_r <= 1'b0;
        else        all_done_r <= done;
    end
    assign master_start = done & ~all_done_r;

    // ====================================================================
    // AXI4-Stream master — streams output after all cores complete
    // ====================================================================
    axi4_stream_master #(
        .SEQ_LEN(SEQ_LEN),
        .NUM_HEADS(NUM_HEADS),
        .PER_HEAD_DIM(PER_HEAD_DIM)
    ) u_master (
        .clk(clk), .rst_n(rst_n),
        .start(master_start),
        .out_raddr_flat(mst_raddr_flat),
        .out_rdata_flat(mst_rdata_flat),
        .m_tdata(m_axis_tdata),
        .m_tvalid(m_axis_tvalid),
        .m_tready(m_axis_tready),
        .m_tlast(m_axis_tlast)
    );

endmodule
