`timescale 1ns/1ps
module gqa_output_packer_m5 (
    input logic clk,rst_n,start,
    output logic ready,busy,done,start_rejected,error,
    input logic [15:0] words_per_head,result_words_per_head,
    input logic result_valid,
    output logic [1:0] out_head,
    output logic [7:0] out_word,
    input logic signed [31:0] out_data,
    output logic src_valid,
    input logic src_ready,
    output logic [63:0] src_data,
    output logic [31:0] retired_words
);
    typedef enum logic [2:0] {IDLE,LOW,HIGH,STREAM,FATAL} state_t;
    state_t state;
    logic [15:0] count;
    logic [31:0] low_word;
    logic valid_start,owner_bad;
    assign ready=state==IDLE;
    assign busy=state==LOW || state==HIGH || state==STREAM;
    assign error=state==FATAL;
    assign valid_start=(words_per_head==16 || words_per_head==256) && result_valid && result_words_per_head==words_per_head;
    assign owner_bad=busy && (!result_valid || result_words_per_head!=count);
    assign src_valid=state==STREAM && !owner_bad;
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=IDLE;count<=0;low_word<=0;src_data<=0;out_head<=0;out_word<=0;
            retired_words<=0;done<=0;start_rejected<=0;
        end else begin
            done<=0;start_rejected<=start && (!ready || !valid_start);
            if(owner_bad)state<=FATAL;
            else if(start && ready && valid_start)begin
                count<=words_per_head;out_head<=0;out_word<=0;retired_words<=0;state<=LOW;
            end else case(state)
                LOW:begin low_word<=out_data;out_word<=out_word+8'd1;state<=HIGH;end
                HIGH:begin src_data<={out_data,low_word};state<=STREAM;end
                STREAM:if(src_valid && src_ready)begin
                    retired_words<=retired_words+32'd2;
                    if({8'd0,out_word}==count-16'd1)begin
                        if(out_head==3)begin state<=IDLE;done<=1;end
                        else begin out_head<=out_head+2'd1;out_word<=0;state<=LOW;end
                    end else begin out_word<=out_word+8'd1;state<=LOW;end
                end
                default:begin end
            endcase
        end
    end
endmodule
