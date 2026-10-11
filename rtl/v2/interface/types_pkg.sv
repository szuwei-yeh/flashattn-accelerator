`timescale 1ns/1ps
package m5_types_pkg;
    // Fixed M5A geometry; packed scale lanes preserve signed INT16 bit patterns.
    // Cast individual scale lanes with $signed at the future M4 connection.
    typedef struct packed {
        logic [1:0] op;
        logic causal,replace_cache;
        logic [6:0] query_count,query_position,kv_length;
        logic [31:0] output_base;
        logic [3:0][31:0] q_base;
        logic [1:0][31:0] k_base,v_base;
        logic [3:0][15:0] scale_q;
        logic [1:0][15:0] scale_k,scale_v;
    } config_t;
    localparam logic [2:0] IDLE=0,DISPATCH=1,RUN=2,WRITEBACK=3,DONE=4,RELEASE=5,ERROR=6;
endpackage
