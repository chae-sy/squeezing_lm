//======================================================================
// attn_reader.v
//  Attention Reader: y_h = ¥Ò_i score[i] * v_mat[i][h]
//======================================================================
`timescale 1ns/1ps

module attn_reader #(
  parameter SEQ_LEN   = 3,                     // sequence length
  parameter HEAD_DIM  = 2,                     // number of heads
  parameter DW        = 4,                     // bitwidth of v_mat elements (signed)
  parameter FRAC_W    = 4                      // fractional bits in score (unsigned)
)(
  input  wire                             clk,
  input  wire                             rst,      // sync reset
  input  wire                             start,    // pulse to begin

  // inputs
  input  wire [FRAC_W-1:0]                score   [0:SEQ_LEN-1],               // Q0.FRAC_W
  input  wire signed [DW-1:0]             v_mat   [0:SEQ_LEN-1][0:HEAD_DIM-1], // signed

  // outputs
  output reg                              out_valid,
  output reg signed [DW+FRAC_W+$clog2(SEQ_LEN)-1:0]
                                          out_vec [0:HEAD_DIM-1]
);

  // internal index and accumulators
  reg [$clog2(SEQ_LEN):0] idx;
  reg running;
  integer h;

  // reset & start logic
  always @(posedge clk) begin
    if (rst) begin
      running   <= 1'b0;
      idx       <= 0;
      out_valid <= 1'b0;
      for (h = 0; h < HEAD_DIM; h = h + 1)
        out_vec[h] <= 0;
    end else begin
      out_valid <= 1'b0;
      if (start) begin
        running <= 1'b1;
        idx     <= 0;
        // clear accumulators
        for (h = 0; h < HEAD_DIM; h = h + 1)
          out_vec[h] <= 0;
      end else if (running) begin
        // multiply-accumulate for each head in parallel
        for (h = 0; h < HEAD_DIM; h = h + 1) begin
          out_vec[h] <= out_vec[h]
                        + $signed({{FRAC_W{1'b0}}, score[idx]})  // extend score to signed DW+FRAC_W bits
                          * $signed({{(FRAC_W+$clog2(SEQ_LEN)){v_mat[idx][h][DW-1]}}, v_mat[idx][h]});
          // Note: v_mat is sign-extended to match accumulator width
        end

        if (idx + 1 < SEQ_LEN) begin
          idx <= idx + 1;
        end else begin
          // last element processed
          running   <= 1'b0;
          out_valid <= 1'b1;
        end
      end
    end
  end

endmodule
