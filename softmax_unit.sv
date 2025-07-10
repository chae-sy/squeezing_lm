//======================================================================
// softmax_unit.v
//  Parameterizable fixed-point softmax
//======================================================================
`timescale 1ns/1ps

module softmax_unit #(
  parameter SEQ_LEN    = 3,    // number of scores
  parameter IN_W       = 8,    // input score bit-width (signed)
  parameter FRAC_W     = 8,    // fractional bits for fixed-point
  parameter EXP_LUT_AW = 6,    // LUT address width for exp (input range)
  parameter EXP_LUT_DW = IN_W+FRAC_W, // LUT data width
  parameter RECIP_LUT_AW = 10, // sum range address width
  parameter RECIP_LUT_DW = FRAC_W // reciprocal fractional width
)(
  input  wire                         clk,
  input  wire                         rst,
  input  wire                         start,       // pulse to begin
  input  wire signed [IN_W-1:0]      scores  [0:SEQ_LEN-1],
  output reg                          done,        // high one cycle when done
  output reg  [FRAC_W-1:0]            out     [0:SEQ_LEN-1]  // Q0.F bits
);

  //====================================================================
  // 1) EXP LUT stage
  //====================================================================
  // simple ROM: exp(x) ? LUT(x_addr)
  wire [EXP_LUT_DW-1:0] exp_val [0:SEQ_LEN-1];
  genvar i;
  generate
    for (i = 0; i < SEQ_LEN; i = i + 1) begin : EXP
      exp_lut #(
        .ADDR_W (EXP_LUT_AW),
        .DATA_W (EXP_LUT_DW)
      ) lut_inst (
        .addr    (scores[i][IN_W-1 -: EXP_LUT_AW]), // use MSBs as address
        .data_out(exp_val[i])
      );
    end
  endgenerate

  //====================================================================
  // 2) Accumulate Σ exp(x)
  //====================================================================
  reg [RECIP_LUT_AW-1:0] sum_addr;
  reg [EXP_LUT_DW+ $clog2(SEQ_LEN)-1:0] sum_val;
  integer idx;
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      sum_val <= 0;
    end else if (start) begin
      sum_val <= 0;
      for (idx = 0; idx < SEQ_LEN; idx = idx + 1)
        sum_val <= sum_val + exp_val[idx];
    end
  end

  // quantize sum to address for reciprocal LUT
  always @* begin
    // take upper RECIP_LUT_AW bits of sum_val
    sum_addr = sum_val[EXP_LUT_DW+ $clog2(SEQ_LEN)-1 -: RECIP_LUT_AW];
  end

  //====================================================================
  // 3) Reciprocal LUT: 1 / sum
  //====================================================================
  wire [RECIP_LUT_DW-1:0] recip_val;
  recip_lut #(
    .ADDR_W(RECIP_LUT_AW),
    .DATA_W(RECIP_LUT_DW)
  ) recip_inst (
    .addr    (sum_addr),
    .data_out(recip_val)
  );

  //====================================================================
  // 4) Multiply exp × reciprocal → softmax outputs
  //====================================================================
  // simple combinational multiplies (truncated to FRAC_W)
  wire [EXP_LUT_DW+RECIP_LUT_DW-1:0] mult [0:SEQ_LEN-1];
  generate
    for (i = 0; i < SEQ_LEN; i = i + 1) begin : MUL
      assign mult[i] = exp_val[i] * recip_val;
      // take middle FRAC_W bits as Q0.FRAC_W result
      always @(posedge clk or posedge rst) begin
        if (rst) out[i] <= 0;
        else if (start) out[i] <= mult[i][FRAC_W +: FRAC_W];
      end
    end
  endgenerate

  //====================================================================
  // 5) Done flag
  //====================================================================
  reg running;
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      running <= 1'b0;
      done    <= 1'b0;
    end else begin
      done <= 1'b0;
      if (start) begin
        running <= 1'b1;
      end else if (running) begin
        // one cycle after start (pipeline simplified)
        done    <= 1'b1;
        running <= 1'b0;
      end
    end
  end

endmodule

//======================================================================
// exp_lut.v (예시 ROM)
//======================================================================
module exp_lut #(
  parameter ADDR_W = 6,
            DATA_W = 16
)(
  input  wire [ADDR_W-1:0] addr,
  output reg  [DATA_W-1:0] data_out
);
  // 간단히 addr→addr^2 예시로 대체 (실제 exp 테이블로 교체하세요)
  always @* data_out = {{(DATA_W-ADDR_W){1'b0}}, addr * addr};
endmodule

//======================================================================
// recip_lut.v (예시 ROM)
//======================================================================
module recip_lut #(
  parameter ADDR_W = 10,
            DATA_W = 8
)(
  input  wire [ADDR_W-1:0] addr,
  output reg  [DATA_W-1:0] data_out
);
  // 간단히 역수 대신 상수 1 예시 (실제 reciprocal 테이블로 교체하세요)
  always @* data_out = {1'b1, {(DATA_W-1){1'b0}}};
endmodule
