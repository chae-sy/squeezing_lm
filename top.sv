//======================================================================
// top_transformer_block.v
//  Pre-LN Transformer Block: LN → Attn+Res → LN → FFN+Res
//======================================================================
`timescale 1ns/1ps
module top_transformer_block #(
  parameter D       = 2,   // hidden size
  parameter HEADS   = 12,
  parameter SEQ_LEN = 2,
  parameter D_FF    = 4,   // FFN hidden
  parameter DW      = 8,
  parameter FRAC_W  = 4,
  // LayerNorm widths
  parameter ACC1_W  = DW + $clog2(D),
  parameter ACC2_W  = 2*DW + $clog2(D),
  parameter OUT_W   = DW + FRAC_W + 1,
  // FFN widths
  parameter ACC_FF_W= DW + $clog2(D_FF)
)(
  input  wire                        clk,
  input  wire                        rst,
  input  wire                        start,

  // --- Block inputs ---
  input  wire signed [DW-1:0]        x_in  [0:D-1],
  // LN1 params
  input  wire signed [DW-1:0]        ln1_gamma[0:D-1],
  input  wire signed [DW-1:0]        ln1_beta [0:D-1],
  // Attention params
  input  wire signed [DW-1:0]        wq       [0:HEADS*D-1],
  input  wire signed [DW-1:0]        wk       [0:HEADS*D-1],
  input  wire signed [DW-1:0]        wv       [0:HEADS*D-1],
  // LN2 params
  input  wire signed [DW-1:0]        ln2_gamma[0:D-1],
  input  wire signed [DW-1:0]        ln2_beta [0:D-1],
  // FFN params
  input  wire signed [DW-1:0]        w1       [0:D_FF*D-1],
  input  wire signed [ACC_FF_W-1:0]  b1       [0:D_FF-1],
  input  wire signed [DW-1:0]        w2       [0:D*D_FF-1],
  input  wire signed [ACC_FF_W-1:0]  b2       [0:D-1],

  // --- Block output ---
  output reg                         out_valid,
  output reg signed [DW-1:0]         y_out    [0:D-1]
);

  // FSM states
  typedef enum logic [2:0] {
    S_IDLE, S_LN1, S_ATT, S_LN2, S_FFN, S_DONE
  } state_t;
  state_t state;

  // Inter-stage signals
  wire signed [DW-1:0]    ln1_out  [0:D-1];
  wire                    ln1_v;
  wire signed [DW+FRAC_W+$clog2(SEQ_LEN)-1:0] att_out [0:HEADS-1];
  wire                    att_v;
  wire signed [DW-1:0]    ln2_out  [0:D-1];
  wire                    ln2_v;
  wire signed [DW-1:0]    ffn_out  [0:D-1];
  wire                    ffn_v;

 genvar i;
 integer j;

  // 1) Pre-LN
  top_layernorm #(
    .D(D), .DW(DW),
    .ACC1_W(ACC1_W), .ACC2_W(ACC2_W),
    .FRAC_W(FRAC_W), .OUT_W(DW)
  ) U_ln1 (
    .clk(clk), .rst(rst), .start(state==S_LN1),
    .x(x_in), .gamma(ln1_gamma), .beta(ln1_beta),
    .out_valid(ln1_v), .y(ln1_out)
  );

  // 2) Attention + Res Add
  top_attention #(
    .D(D), .HEADS(HEADS), .SEQ_LEN(SEQ_LEN),
    .DW(DW), .FRAC_W(FRAC_W)
  ) U_att (
    .clk(clk), .rst(rst), .start(ln1_v),
    .in_vec(ln1_out), .wq(wq), .wk(wk), .wv(wv),
    .out_valid(att_v), .out_vec(att_out)
  );
  wire signed [DW-1:0] att_res [0:D-1];
  // 각 head 출력을 더하고 residual 합산 (여기선 간단히 head0만 사용; D==HEADS)
  generate
    for (i = 0; i < D; i = i + 1) begin
      // 만약 D>HEADS, 적절히 합산 로직 수정 필요
      assign att_res[i] = att_out[i][DW+FRAC_W-1 -: DW] + ln1_out[i];
    end
  endgenerate

  // 3) Post-LN
  top_layernorm #(
    .D(D), .DW(DW),
    .ACC1_W(ACC1_W), .ACC2_W(ACC2_W),
    .FRAC_W(FRAC_W), .OUT_W(DW)
  ) U_ln2 (
    .clk(clk), .rst(rst), .start(att_v),
    .x(att_res), .gamma(ln2_gamma), .beta(ln2_beta),
    .out_valid(ln2_v), .y(ln2_out)
  );

  // 4) FFN + Res Add
  top_ffn #(
    .D_IN(D), .D_FF(D_FF),
    .DW(DW), .ACC1_W(ACC_FF_W),
    .ADDR_W(DW), .DATA_W(DW)
  ) U_ffn (
    .clk(clk), .rst(rst), .start(ln2_v),
    .in_vec(ln2_out),
    .w1_mat(w1), .b1_vec(b1),
    .w2_mat(w2), .b2_vec(b2),
    .out_valid(ffn_v), .out_vec(ffn_out)
  );
  wire signed [DW-1:0] ffn_res [0:D-1];
  generate
    for (i = 0; i < D; i = i + 1) begin
      assign ffn_res[i] = ffn_out[i] + ln2_out[i];
    end
  endgenerate

  // FSM
  always @(posedge clk) begin
    if (rst) begin
      state      <= S_IDLE;
      out_valid  <= 0;
    end else begin
      out_valid <= 0;
      case (state)
        S_IDLE: if (start) state <= S_LN1;
        S_LN1:  if (ln1_v)   state <= S_ATT;
        S_ATT:  if (att_v)   state <= S_LN2;
        S_LN2:  if (ln2_v)   state <= S_FFN;
        S_FFN:  if (ffn_v)   state <= S_DONE;
        S_DONE: begin
                   for (j = 0; j < D; j = j + 1)
                     y_out[j] <= ffn_res[j];
                   out_valid <= 1;
                   state     <= S_IDLE;
                 end
      endcase
    end
  end

endmodule
