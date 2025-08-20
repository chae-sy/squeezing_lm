`timescale 1ns / 1ps

module top_block_ctrl(
    input  clk,
    input  reset,

    input  proj_done,
    input  qk_matmul_done,
    input  attn_reader_done,
    input  linear1_done, 
    input  linear2_done,
    input  ln_done,

    output reg proj_start,
    output reg qk_matmul_start,
    output reg attn_reader_start,
    output reg linear1_start,
    output reg linear2_start,
    output reg ln_start,
    
    output reg state
);

    // ---- State encoding ----
    localparam [3:0]
        IDLE    = 4'd0,
        LN1     = 4'd1,
        PRJ_Q   = 4'd2,
        PRJ_K   = 4'd3,
        PRJ_V   = 4'd4,
        QK_MM   = 4'd5,
        ATTN_R  = 4'd6,
        OUT_PRJ = 4'd7,
        LN2     = 4'd8,
        FFN1    = 4'd9,
        FFN2    = 4'd10;

    reg [3:0] next_state;

    // ---- State register ----
    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= IDLE;
        else
            state <= next_state;
    end

    // ---- Next-state logic ----
    always @* begin
        next_state = state;
        case (state)
            IDLE:     next_state = LN1;

            // LayerNorm 1
            LN1:      next_state = (ln_done)           ? PRJ_Q   : LN1;

            // Projections Q/K/V (share proj_done)
            PRJ_Q:    next_state = (proj_done)         ? PRJ_K   : PRJ_Q;
            PRJ_K:    next_state = (proj_done)         ? PRJ_V   : PRJ_K;
            PRJ_V:    next_state = (proj_done)         ? QK_MM   : PRJ_V;

            // QK matmul
            QK_MM:    next_state = (qk_matmul_done)    ? ATTN_R  : QK_MM;

            // Attention reader
            ATTN_R:   next_state = (attn_reader_done)  ? OUT_PRJ : ATTN_R;

            // Output projection (linear)
            OUT_PRJ:  next_state = (proj_done)         ? LN2     : OUT_PRJ;

            // LayerNorm 2
            LN2:      next_state = (ln_done)           ? FFN1    : LN2;

            // FFN1 (linear)
            FFN1:     next_state = (linear1_done)       ? FFN2    : FFN1;

            // FFN2 (linear) then finish
            FFN2:     next_state = (linear2_done)       ? IDLE    : FFN2;

            default:  next_state = IDLE;
        endcase
    end

    // ---- Output logic (level-high while the stage is active) ----
    always @* begin
        proj_start        = 1'b0;
        qk_matmul_start   = 1'b0;
        attn_reader_start = 1'b0;
        linear1_start     = 1'b0;
        linear2_start     = 1'b0;
        ln_start          = 1'b0;

        case (state)
            LN1, LN2:                           ln_start          = 1'b1;
            PRJ_Q, PRJ_K, PRJ_V, OUT_PRJ:       proj_start        = 1'b1;
            QK_MM:                              qk_matmul_start   = 1'b1;
            ATTN_R:                             attn_reader_start = 1'b1;
            FFN1:                               linear1_start     = 1'b1;
            FFN2:                               linear2_start     = 1'b1;
            default: ;
        endcase
    end

endmodule
