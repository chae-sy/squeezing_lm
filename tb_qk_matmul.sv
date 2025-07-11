`timescale 1ns/1ps

module tb_qk_matmul;
  //-------------------------------------------------------------------------
  // 1) Parameters
  //-------------------------------------------------------------------------
  localparam HEAD_DIM = 4;
  localparam SEQ_LEN  = 3;
  localparam DW       = 4;

  //-------------------------------------------------------------------------
  // 2) Signal declarations
  //-------------------------------------------------------------------------
  reg                        clk, rst;
  reg                        start;
  reg   signed [DW-1:0]      q_vec   [0:HEAD_DIM-1];
  reg   signed [DW-1:0]      k_mat   [0:SEQ_LEN-1][0:HEAD_DIM-1];

  wire                       done;
  wire signed [2*DW-1:0]     score   [0:SEQ_LEN-1];

  integer i;

  //-------------------------------------------------------------------------
  // 3) DUT instantiation
  //-------------------------------------------------------------------------
  qk_matmul #(
    .HEAD_DIM(HEAD_DIM),
    .SEQ_LEN (SEQ_LEN),
    .DW      (DW)
  ) dut (
    .clk    (clk),
    .rst    (rst),
    .start  (start),
    .q_vec  (q_vec),
    .k_mat  (k_mat),
    .done   (done),
    .score  (score)
  );

  //-------------------------------------------------------------------------
  // 4) Clock generation
  //-------------------------------------------------------------------------
  initial clk = 0;
  always #5 clk = ~clk;  // 100 MHz

  //-------------------------------------------------------------------------
  // 5) Test sequence
  //-------------------------------------------------------------------------
  initial begin
    // 5.1 Reset
    rst   = 1;
    start = 0;
    // clear vectors
    for (i = 0; i < HEAD_DIM; i = i + 1) q_vec[i] = 0;
    for (i = 0; i < SEQ_LEN;  i = i + 1)
      for (integer j = 0; j < HEAD_DIM; j = j + 1)
        k_mat[i][j] = 0;
    #20; rst = 0;

    // 5.2 Apply stimulus
    // q_vec = [1,2,3,4]
    q_vec[0] = 4'd1;
    q_vec[1] = 4'd2;
    q_vec[2] = 4'd3;
    q_vec[3] = 4'd4;
    // k_mat rows:
    // row0 = [1,0,1,0] → dot = 1*1 + 2*0 + 3*1 + 4*0 = 4
    // row1 = [0,1,0,1] → dot = 1*0 + 2*1 + 3*0 + 4*1 = 6
    // row2 = [1,1,1,1] → dot = 1+2+3+4 = 10
    k_mat[0] = '{4'd1,4'd0,4'd1,4'd0};
    k_mat[1] = '{4'd0,4'd1,4'd0,4'd1};
    k_mat[2] = '{4'd1,4'd1,4'd1,4'd1};

    // 5.3 Pulse start
    #10;
    start = 1;
    #10;
    start = 0;

    // 5.4 Wait for done
    wait (done);

    // 5.5 Display results
    $display("\n=== qk_matmul Results ===");
    $display("score[0] = %0d (expected  4)", score[0]);
    $display("score[1] = %0d (expected  6)", score[1]);
    $display("score[2] = %0d (expected 10)", score[2]);

    #20;
    $finish;
  end

endmodule
