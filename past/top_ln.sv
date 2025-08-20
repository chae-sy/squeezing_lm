//======================================================================
// top_layernorm.v
//  FSM-based top for LayerNorm
//======================================================================
`timescale 1ns/1ps
module top_layernorm #(
  parameter D       = 4,
  parameter DW      = 8,
  parameter ACC1_W  = DW + $clog2(D),
  parameter ACC2_W  = 2*DW + $clog2(D),
  parameter FRAC_W  = 8,
  parameter OUT_W   = DW + FRAC_W + 1
)(
  input  wire                     clk,
  input  wire                     rst,
  input  wire                     start,      // 연산 시작 펄스
  input  wire signed [DW-1:0]     x    [0:D-1],
  input  wire signed [DW-1:0]     gamma[0:D-1],
  input  wire signed [DW-1:0]     beta [0:D-1],

  output reg                      out_valid,  // 완료 펄스
  output reg signed [OUT_W-1:0]   y    [0:D-1]
);

  // FSM 상태 정의
  typedef enum logic [2:0] {
    S_IDLE, S_MEAN, S_VAR, S_RSQRT, S_NORM, S_DONE
  } state_t;
  state_t state;

  // 내부 신호
  wire signed [ACC1_W-1:0] mean;
  wire                     mean_v;
  wire signed [ACC2_W-1:0] vari;
  wire                     var_v;
  wire [FRAC_W-1:0]        rsqrt;
  wire                     rs_v;

  integer i;

  // mean
  mean_unit #(.D(D), .DW(DW)) Umean (
    .clk(clk), .rst(rst), .start(state==S_MEAN),
    .x(x), .mean(mean), .valid(mean_v)
  );

  // var (start when mean_v)
  var_unit #(.D(D), .DW(DW)) Uvar (
    .clk(clk), .rst(rst), .start(mean_v),
    .x(x), .mean(mean), .vari(vari), .valid(var_v)
  );

  // rsqrt (start when var_v)
  // LUT 크기에 맞춰 ADDRESS_W = FRAC_W (var 상위 비트 사용)
  wire [FRAC_W-1:0] var_addr = vari[ACC2_W-1 -: FRAC_W];
  rsqrt_unit #(.ADDR_W(FRAC_W), .DATA_W(FRAC_W)) Ursqrt (
    .clk(clk), .rst(rst),
    .in_valid(var_v),
    .addr(var_addr),
    .rsqrt(rsqrt),
    .out_valid(rs_v)
  );

  // 최종 정규화 및 scale/shift
  reg processing;
  reg signed [DW+FRAC_W:0] diff_scaled;
  always @(posedge clk) begin
    if (rst) begin
      state     <= S_IDLE;
      out_valid <= 0;
      processing<= 0;
      for (i=0; i<D; i=i+1) y[i] <= 0;
    end else begin
      // 기본 신호 클리어
      out_valid <= 0;

      case (state)
        S_IDLE: if (start) state <= S_MEAN;

        S_MEAN: if (mean_v) state <= S_VAR;

        S_VAR:  if (var_v)  state <= S_RSQRT;

        S_RSQRT: if (rs_v) begin
          state      <= S_NORM;
          processing <= 1;
        end

        S_NORM: if (processing) begin
          // 병렬로 D개 처리
          for (i=0; i<D; i=i+1) begin
            // (x-mean)*rsqrt
            diff_scaled = (x[i] - mean) * $signed({1'b0, rsqrt});
            // *gamma >> FRAC_W + beta
            y[i] <= (diff_scaled * gamma[i]) >>> FRAC_W + beta[i];
          end
          out_valid  <= 1;
          processing <= 0;
          state      <= S_DONE;
        end

        S_DONE: state <= S_IDLE;
      endcase
    end
  end

endmodule
