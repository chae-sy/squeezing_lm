//======================================================================
// layernorm_unit.v
//  Full LayerNorm: y_i = gamma_i * ((x_i - mean) * rsqrt) + beta_i
//======================================================================
`timescale 1ns/1ps

module layernorm_unit #(
  parameter D        = 4,                           // vector length
  parameter DW       = 8,                           // input bit-width
  parameter ACC1_W   = DW + $clog2(D),              // mean accumulator
  parameter ACC2_W   = 2*DW + $clog2(D),            // var accumulator
  parameter FRAC_W   = 8,                           // fractional bits for rsqrt Q0.FRAC_W
  parameter OUT_W    = DW + FRAC_W + 1              // output width to hold scale result
)(
  input  wire                     clk,
  input  wire                     rst,
  input  wire                     start,            // one-cycle pulse to begin
  input  wire signed [DW-1:0]     x      [0:D-1],
  input  wire signed [DW-1:0]     gamma  [0:D-1],
  input  wire signed [DW-1:0]     beta   [0:D-1],

  output reg                      out_valid,
  output reg signed [OUT_W-1:0]   y      [0:D-1]
);

  // 내부 신호
  wire signed [ACC1_W-1:0] mean;
  wire                   mean_v;
  wire signed [ACC2_W-1:0] vari;
  wire                   var_v;
  wire [FRAC_W-1:0]      rsqrt;
  wire                   rs_v;

  // 1) Mean
  mean_unit #(.D(D), .DW(DW)) U_mean (
    .clk(clk), .rst(rst), .start(start),
    .x(x), .mean(mean), .valid(mean_v)
  );

  // 2) Variance (start when mean_v)
  var_unit #(.D(D), .DW(DW)) U_var (
    .clk(clk), .rst(rst), .start(mean_v),
    .x(x), .mean(mean),
    .vari(vari), .valid(var_v)
  );

  // 3) Rsqrt(variance) (start when var_v)
  // addr: 상위 FRAC_W 비트 사용 (divide range)
  wire [FRAC_W-1:0] var_addr = vari[ACC2_W-1 -: FRAC_W];
  rsqrt_unit #(.ADDR_W(FRAC_W), .DATA_W(FRAC_W)) U_rsqrt (
    .clk(clk), .rst(rst), .in_valid(var_v),
    .addr(var_addr), .rsqrt(rsqrt), .out_valid(rs_v)
  );
  
  reg diff_scaled; //declare diff_scaled
  // 4) Normalize & Scale/Shift
  // start when rs_v
  integer i;
  reg processing;
  wire signed [DW+FRAC_W:0] diff_scaled;
  always @(posedge clk) begin
    if (rst) begin
      out_valid  <= 0;
      processing <= 0;
      for (i = 0; i < D; i = i + 1) y[i] <= 0;
    end else begin
      out_valid <= 0;
      if (rs_v) begin
        processing <= 1;
      end else if (processing) begin
        // 한 사이클에 D개 병렬로 처리
        for (i = 0; i < D; i = i + 1) begin
          // (x - mean) * rsqrt  => Q(DW + FRAC_W)  
          diff_scaled = (x[i] - mean) * $signed({1'b0, rsqrt});
          // multiply by gamma (DW x Q format) >> FRAC_W, then add beta
          y[i] <= (diff_scaled * gamma[i]) >>> FRAC_W + beta[i];
        end
        out_valid  <= 1;
        processing <= 0;
      end
    end
  end

endmodule
