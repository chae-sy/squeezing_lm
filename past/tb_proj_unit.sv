`timescale 1ns/1ps

module tb_proj_unit_weight_cap;

  //-------------------------------------------------------------------------
  // 파라미터
  //-------------------------------------------------------------------------
  localparam N       = 16;
  localparam DW      = 4;      // 4-bit signed: -8…+7
  localparam PE_NUM  = 12;

  //-------------------------------------------------------------------------
  // 신호 선언
  //-------------------------------------------------------------------------
  reg                    clk, rst;
  reg                    start, in_valid;
  reg  signed [DW-1:0]   in_vec   [0:N-1];
  reg  signed [DW-1:0]   w_q      [0:PE_NUM*N-1];
  reg  signed [DW-1:0]   w_k      [0:PE_NUM*N-1];
  reg  signed [DW-1:0]   w_v      [0:PE_NUM*N-1];

  wire                   out_valid;
  wire signed [2*DW+$clog2(N)-1:0] out_q   [0:PE_NUM-1];
  wire signed [2*DW+$clog2(N)-1:0] out_k   [0:PE_NUM-1];
  wire signed [2*DW+$clog2(N)-1:0] out_v   [0:PE_NUM-1];

  integer i, h;

  //-------------------------------------------------------------------------
  // DUT 인스턴스
  //-------------------------------------------------------------------------
  proj_unit #(
    .N      (N),
    .DW     (DW),
    .PE_NUM (PE_NUM)
  ) dut (
    .clk       (clk),
    .rst       (rst),
    .start     (start),
    .in_valid  (in_valid),
    .in_vec    (in_vec),
    .w_q       (w_q),
    .w_k       (w_k),
    .w_v       (w_v),
    .out_valid (out_valid),
    .out_q     (out_q),
    .out_k     (out_k),
    .out_v     (out_v)
  );

  //-------------------------------------------------------------------------
  // Clock
  //-------------------------------------------------------------------------
  initial clk = 0;
  always #5 clk = ~clk;  // 100 MHz

  //-------------------------------------------------------------------------
  // Test sequence
  //-------------------------------------------------------------------------
  initial begin
    // 1) Reset & init
    rst       = 1;
    start     = 0;
    in_valid  = 0;
    for (i = 0; i < N;    i = i + 1) in_vec[i] = 0;
    for (i = 0; i < PE_NUM*N; i = i + 1) begin
      w_q[i] = 0;
      w_k[i] = 0;
      w_v[i] = 0;
    end
    #20; rst = 0;

    // 2) Stimulus: 앞 8원소만 1
    for (i = 0; i < 8; i = i + 1)
      in_vec[i] = 4'd1;

    // 3) Weight 설정: head h → w = min(h+1, 7)
    for (h = 0; h < PE_NUM; h = h + 1) begin
      // compute capped weight
      // for Q: multiply by 1
      // for K: multiply by 2
      // for V: multiply by 3
      integer cap = (h+1 <= 7 ? h+1 : 7);
      integer cap2 = (cap * 2 <= 7 ? cap*2 : 7);
      integer cap3 = (cap * 3 <= 7 ? cap*3 : 7);
      for (i = 0; i < N; i = i + 1) begin
        w_q[h*N + i] = cap;
        w_k[h*N + i] = cap2;
        w_v[h*N + i] = cap3;
      end
    end

    // 4) Start one projection
    #10;
    start    = 1;
    in_valid = 1;
    #10;
    start    = 0;
    in_valid = 0;

    // 5) wait for completion
    wait(out_valid);

    // 6) 결과 출력 & 체크
    $display("=== proj_unit (weight capped) 결과 ===");
    for (h = 0; h < PE_NUM; h = h + 1) begin
      integer exp_q = 8 * (h+1 <= 7 ? h+1 : 7);
      integer exp_k = 8 * (h+1 <= 7 ? (h+1)*2 <= 7 ? (h+1)*2 : 7 : 7);
      integer exp_v = 8 * (h+1 <= 7 ? (h+1)*3 <= 7 ? (h+1)*3 : 7 : 7);
      $display("head%0d: out_q=%0d (exp %0d), out_k=%0d (exp %0d), out_v=%0d (exp %0d)",
        h,
        out_q[h], exp_q,
        out_k[h], exp_k,
        out_v[h], exp_v
      );
    end

    #20;
    $finish;
  end

endmodule
