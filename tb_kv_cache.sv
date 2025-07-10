`timescale 1ns/1ps

module tb_kv_cache;

  //------------------------------------------------------------------------
  // 1) パラメータ
  //------------------------------------------------------------------------
  localparam int MAX_SEQ_LEN = 8;
  localparam int HEAD_DIM    = 12;
  localparam int DW          = 4;

  //------------------------------------------------------------------------
  // 2) 信号宣言
  //------------------------------------------------------------------------
  logic                         clk, rst;
  logic                         write_en;
  logic  [$clog2(MAX_SEQ_LEN)-1:0] write_addr, read_addr;
  logic  [HEAD_DIM*DW-1:0]      data_in;
  logic  [HEAD_DIM*DW-1:0]      data_out;
  logic                         valid_out;

  // ヘッド単位の期待データ格納用
  logic [DW-1:0]                exp_vec [0:HEAD_DIM-1];

  integer addr, h=0;

  //------------------------------------------------------------------------
  // 3) DUT インスタンス
  //------------------------------------------------------------------------
  kv_cache #(
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .HEAD_DIM   (HEAD_DIM),
    .DW         (DW)
  ) dut (
    .clk        (clk),
    .rst        (rst),
    .write_en   (write_en),
    .write_addr (write_addr),
    .data_in    (data_in),
    .read_addr  (read_addr),
    .data_out   (data_out),
    .valid_out  (valid_out)
  );

  //------------------------------------------------------------------------
  // 4) Clock 生成
  //------------------------------------------------------------------------
  initial clk = 0;
  always #5 clk = ~clk;  // 100 MHz

  //------------------------------------------------------------------------
  // 5) テストシーケンス
  //------------------------------------------------------------------------
  integer k;

  
  initial begin
    // reset
    rst       = 1;
    write_en  = 0;
    write_addr= 0;
    read_addr = 0;
    data_in   = '0;
    // exp_vec 초기화
  for (k = 0; k < HEAD_DIM; k = k + 1) begin
    exp_vec[k] = '0;
    end
    #20; rst  = 0;


    $display("\n==== Read before any write ====");
    // 未書き込み addr を順に read
    for (addr = 0; addr < MAX_SEQ_LEN; addr++) begin
      read_addr = addr;
      #10;
      $display("read_addr=%0d | valid_out=%b | data_out=%p",
               addr, valid_out, data_out);
    end

    $display("\n==== Perform writes ====");
    // 各 addr に異なるベクトルを書き込み
    for (addr = 0; addr < MAX_SEQ_LEN; addr++) begin
      // ヘッド h ごとに data = addr*10 + h
      for (h = 0; h < HEAD_DIM; h++) begin
        exp_vec[h] = addr*10 + h;
      end
      // flatten data_in
      data_in = {exp_vec[0], exp_vec[1], exp_vec[2], exp_vec[3], exp_vec[4], exp_vec[5], exp_vec[6], exp_vec[7], exp_vec[8], exp_vec[9], exp_vec[10], exp_vec[11]};
      write_addr = addr;
      #10; write_en = 1;
      #10; write_en = 0;
      $display("Wrote addr=%0d data=%p", addr, data_in);
    end

    $display("\n==== Read back writes ====");
    // 書き込み後を read
    for (addr = 0; addr < MAX_SEQ_LEN; addr++) begin
      // set expected vector again
      for (h = 0; h < HEAD_DIM; h++)
        exp_vec[h] = addr*10 + h;
      data_in = {exp_vec[0], exp_vec[1], exp_vec[2], exp_vec[3], exp_vec[4], exp_vec[5], exp_vec[6], exp_vec[7], exp_vec[8], exp_vec[9], exp_vec[10], exp_vec[11]};
      read_addr = addr;
      #10;
      $display("read_addr=%0d | valid_out=%b | data_out=%p | exp=%p",
               addr, valid_out, data_out, data_in);
    end

    $display("\n==== Overwrite addr=3 and read again ====");
    // addr=3 を別ベクトルで上書き
    for (h = 0; h < HEAD_DIM; h++)
      exp_vec[h] = 100 + h;
    data_in = {exp_vec[0], exp_vec[1], exp_vec[2], exp_vec[3], exp_vec[4], exp_vec[5], exp_vec[6], exp_vec[7], exp_vec[8], exp_vec[9], exp_vec[10], exp_vec[11]};
    write_addr = 3;
    #10; write_en = 1;
    #10; write_en = 0;
    $display("Wrote addr=3 data=%p", data_in);

    read_addr = 3;
    #10;
    $display("read_addr=3 | valid_out=%b | data_out=%p | exp=%p",
             valid_out, data_out, data_in);

    #20;
    $display("\n==== TEST COMPLETE ====");
    $finish;
  end

endmodule
