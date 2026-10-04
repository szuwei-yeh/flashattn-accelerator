`timescale 1ns/1ps
// Four-state main-path regression. Drive on falling edges, sample after NBA.
module tb_attention;
    parameter int D = 16, LANES = 256;
    localparam int N = 64, MAT = N*D;
    logic clk=0;
    always #5 clk=~clk;
    logic rst_n=0, start=0, causal=0, init_we=0;
    logic signed [15:0] scale_q=0, scale_k=0, scale_v=0;
    logic [15:0] rd_latency=0, cfg_seq_len=N;
    logic [31:0] cfg_q_base=0, cfg_k_base=MAT, cfg_v_base=2*MAT;
    logic [1:0] inject_rresp=0;
    logic inject_bad_rlast=0;
    logic [31:0] init_addr=0;
    logic [7:0] init_data=0;
    logic [11:0] out_raddr=0;
    wire done, cfg_error, dma_error, dbg_output_we;
    wire signed [31:0] out_rdata;
    wire [31:0] perf_total_cycles, perf_dma_bytes;
    wire [15:0] perf_kv_tiles_loaded, dbg_kv_tiles_ready;
    logic [7:0] q[0:MAT-1], k[0:MAT-1], v[0:MAT-1];
    logic [31:0] expected[0:MAT-1];
    string data_dir, activity_file;
    int sq,sk,sv,mode_arg,latency_arg,contract_arg=0,restart_arg=1;
    int event_id=0, cycles=0, output_writes=0;
    bit capture_activity=0;
    tb_dma_banked_prefetch_harness #(.SEQ_LEN(N),.HEAD_DIM(D),.DEQUANT_LANES(LANES)) dut (
        .clk(clk),.rst_n(rst_n),.start(start),.done(done),.causal(causal),
        .scale_q(scale_q),.scale_k(scale_k),.scale_v(scale_v),.rd_latency(rd_latency),
        .inject_rresp(inject_rresp),.inject_bad_rlast(inject_bad_rlast),
        .init_we(init_we),.init_addr(init_addr),.init_data(init_data),
        .out_raddr(out_raddr),.out_rdata(out_rdata),
        .cfg_seq_len(cfg_seq_len),.cfg_q_base(cfg_q_base),.cfg_k_base(cfg_k_base),.cfg_v_base(cfg_v_base),
        .cfg_error(cfg_error),.dma_error(dma_error),.perf_total_cycles(perf_total_cycles),
        .perf_dma_busy_cycles(),.perf_core_busy_cycles(),.perf_dma_bytes(perf_dma_bytes),
        .perf_kv_tiles_loaded(perf_kv_tiles_loaded),.perf_first_tile_wait_cycles(),
        .dbg_kv_tiles_ready(dbg_kv_tiles_ready),.dbg_output_we(dbg_output_we)
    );
    // Successful completions, not just attempted stimuli, feed these bins.
    covergroup completed_cg with function sample(int lat,bit mask);
        option.per_instance=1;
        latency: coverpoint lat { bins supported[] = {0,20,100}; }
        causal_mode: coverpoint mask { bins modes[] = {0,1}; }
        latency_x_causal: cross latency,causal_mode;
    endgroup
    covergroup contracts_cg with function sample(int event_kind);
        option.per_instance=1;
        event_cp: coverpoint event_kind {
            bins rejected_config={1}; bins locked_busy_inputs={2};
            bins bad_rresp={3}; bins bad_rlast={4}; bins resident_error={5};
            bins common_reset_recovery={6}; bins mid_output_abort={7};
            bins changed_v_restart={8}; bins single_shot={9};
        }
    endgroup
    completed_cg completion = new;
    contracts_cg contracts = new;
    // Explicit observation bins describe controller reachability; not a proof.
    covergroup controller_cg @(negedge clk);
        option.per_instance=1;
        state_cp: coverpoint dut.u_dut.u_core._dbg_state iff(rst_n) {
            bins states[] = {0,1,2,3,4,6,7,8,9,10,11,12};
        }
        promotion: coverpoint dut.u_dut.u_core.kv_swap_banks iff(rst_n) {bins used={1};}
    endgroup
    controller_cg controller = new;
    task automatic valid_config;
        cfg_seq_len=N;cfg_q_base=0;cfg_k_base=MAT;cfg_v_base=2*MAT;
        scale_q=16'(sq);scale_k=16'(sk);scale_v=16'(sv);causal=mode_arg!=0;
    endtask
    task automatic reset_common;
        rst_n=0;start=0;init_we=0;inject_rresp=0;inject_bad_rlast=0;
        valid_config();repeat(3) @(negedge clk);rst_n=1;@(negedge clk);
    endtask
    task automatic load_dram(input bit zero_v);
        for(int i=0;i<3*MAT;i++) begin
            init_we=1;init_addr=i;
            if(i<MAT) init_data=q[i];
            else if(i<2*MAT) init_data=k[i-MAT];
            else init_data=zero_v?8'b0:v[i-2*MAT];
            @(negedge clk);
        end
        init_we=0;@(negedge clk);
    endtask
    task automatic launch;
        if(cfg_error !== 0) $fatal(1,"Unexpected configuration error");
        start=1;@(negedge clk);start=0;
    endtask
    task automatic compare_output(input bit zero_v);
        for(int i=0;i<MAT;i++) begin
            out_raddr=12'(i);repeat(2) @(negedge clk);
            if(out_rdata !== (zero_v?32'b0:expected[i]))
                $fatal(1,"MISMATCH index=%0d got=%h expected=%h",i,out_rdata,zero_v?32'b0:expected[i]);
        end
        if(perf_dma_bytes !== 32'(3*MAT) || perf_kv_tiles_loaded !== 16'd4)
            $fatal(1,"DMA accounting mismatch");
    endtask
    task automatic successful_job(input bit zero_v,input bit hostile);
        bit accepted_mode;
        time active_begin,active_end;
        reset_common();load_dram(zero_v);accepted_mode=causal;
        if(capture_activity) begin
            $dumpfile(activity_file);$dumpvars(0,dut.u_dut);
            $dumpon;
        end
        active_begin=$time;
        launch();cycles=0;
        if(hostile) begin
            cfg_seq_len=48;cfg_q_base='h1010;cfg_k_base='h2020;cfg_v_base='h3030;
            scale_q=16'(sq^'h55);scale_k=16'(sk^'h33);scale_v=16'(sv^'h77);causal=!accepted_mode;
        end
        while(done !== 1'b1 && cycles<200000) begin
            if(hostile) start=(cycles==20 || cycles==500 || cycles==2000);
            @(negedge clk);cycles++;start=0;
            if(dma_error !== 0) $fatal(1,"DMA error/X in successful job");
        end
        if(done !== 1'b1) $fatal(1,"TIMEOUT");
        active_end=$time;
        if(capture_activity) begin
            $dumpoff;
            $display("ACTIVITY_WINDOW begin_ns=%0t end_ns=%0t cycles_including_start=%0d",active_begin,active_end,cycles+1);
        end
        // Logical accounting includes the accepted-start edge; TB cycles omit it.
        compare_output(zero_v);
        if(perf_total_cycles !== 32'(cycles+1)) $fatal(1,"Cycle counter mismatch");
        completion.sample(latency_arg,accepted_mode);
        if(hostile) contracts.sample(2);
        if(zero_v) contracts.sample(8);
        $display("EXACT_PASS d=%0d lanes=%0d causal=%0d latency=%0d zero_v=%0d tb_cycles=%0d words=%0d",D,LANES,accepted_mode,latency_arg,zero_v,cycles,MAT);
    endtask
    task automatic reject_config;
        start=1;@(negedge clk);start=0;repeat(8) @(negedge clk);
        if(cfg_error !== 1 || perf_total_cycles !== 0 || dbg_kv_tiles_ready !== 0 || done !== 0)
            $fatal(1,"Invalid configuration launched work");
        contracts.sample(1);
    endtask
    task automatic check_contracts;
        int wait_cycles,ready_before;
        reset_common();load_dram(0);
        cfg_seq_len=0;reject_config();cfg_seq_len=48;reject_config();cfg_seq_len=63;reject_config();
        valid_config();cfg_q_base=8;reject_config();
        for(int operand=0;operand<3;operand++) begin
            valid_config();
            if(operand==0) cfg_q_base=32'(0-MAT);else if(operand==1) cfg_k_base=32'(0-MAT);else cfg_v_base=32'(0-MAT);
            #1;if(cfg_error !== 0) $fatal(1,"Final legal address rejected");
            if(operand==0) cfg_q_base+=16;else if(operand==1) cfg_k_base+=16;else cfg_v_base+=16;
            reject_config();
        end
        successful_job(0,1);
        valid_config();start=1;@(negedge clk);start=0;repeat(20) @(negedge clk);
        if(perf_total_cycles !== 32'(cycles+1) || perf_dma_bytes !== 32'(3*MAT)) $fatal(1,"Single-shot contract violated");
        compare_output(0);
        contracts.sample(9);
        for(int kind=0;kind<3;kind++) begin
            reset_common();load_dram(0);launch();
            if(kind==2) begin
                wait_cycles=0;
                while(dbg_kv_tiles_ready==0 && wait_cycles<10000) begin @(negedge clk);wait_cycles++;end
                if(dbg_kv_tiles_ready !== 16'd1) $fatal(1,"No first resident tile");
            end
            ready_before=dbg_kv_tiles_ready;
            if(kind==1) inject_bad_rlast=1;else inject_rresp=2;
            wait_cycles=0;
            while(dma_error !== 1 && wait_cycles<10000) begin @(negedge clk);wait_cycles++;end
            if(dma_error !== 1) $fatal(1,"DMA fault missing");
            repeat(100) begin
                @(negedge clk);
                if(dma_error !== 1 || done !== 0 || dbg_kv_tiles_ready !== ready_before)
                    $fatal(1,"Fault cleared, completed, or promoted a failed tile");
            end
            contracts.sample(kind+3);
        end
        successful_job(0,0);contracts.sample(6);
        reset_common();load_dram(0);launch();output_writes=0;wait_cycles=0;
        while(output_writes<8 && wait_cycles<200000) begin
            if(dbg_output_we === 1) output_writes++;
            @(negedge clk);wait_cycles++;
        end
        if(output_writes!=8 || done !== 0) $fatal(1,"Failed to reach partial output abort");
        contracts.sample(7);successful_job(1,0);
        $display("CONTRACT_PASS");
    endtask
    initial begin
        $timeformat(-9,0,"",0);
        if(!$value$plusargs("DATA=%s",data_dir)) $fatal(1,"Missing DATA");
        if(!$value$plusargs("SQ=%d",sq) || !$value$plusargs("SK=%d",sk) || !$value$plusargs("SV=%d",sv)) $fatal(1,"Missing scales");
        if(!$value$plusargs("CAUSAL=%d",mode_arg) || !$value$plusargs("LATENCY=%d",latency_arg)) $fatal(1,"Missing mode/latency");
        if($value$plusargs("CONTRACT=%d",contract_arg)) begin end
        if($value$plusargs("RESTART=%d",restart_arg)) begin end
        capture_activity=$value$plusargs("ACTIVITY=%s",activity_file);
        rd_latency=16'(latency_arg);
        $readmemh({data_dir,"/q_input.hex"},q);$readmemh({data_dir,"/k_input.hex"},k);
        $readmemh({data_dir,"/v_input.hex"},v);$readmemh({data_dir,"/expected.hex"},expected);
        for(int i=0;i<MAT;i++)
            if($isunknown({q[i],k[i],v[i],expected[i]})) $fatal(1,"Incomplete/unknown fixture at %0d",i);
        #1;if(!$isunknown(dut.u_dut.u_core.u_out_buf.u_sram_o.mem[0])) $fatal(1,"Requires unknown initial SRAM");
        if(contract_arg!=0) check_contracts();
        else begin
            successful_job(0,0);
            capture_activity=0;
            if(restart_arg!=0) successful_job(1,0);
        end
        $display("FUNCTIONAL_COVERAGE completion=%0.2f contracts=%0.2f controller=%0.2f",completion.get_inst_coverage(),contracts.get_inst_coverage(),controller.get_inst_coverage());
        $display("RESULT: PASS VCS attention four-state");$finish;
    end
endmodule
