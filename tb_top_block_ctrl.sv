`timescale 1ns/1ps

module top_block_ctrl_tb;

  // Clock & reset
  reg clk = 1'b0;
  reg reset;

  // DUT I/O
  reg  proj_done, qk_matmul_done, attn_reader_done, linear1_done, linear2_done, ln_done;
  wire proj_start, qk_matmul_start, attn_reader_start, linear1_start, linear2_start, ln_start;

  // Clock: 100 MHz
  always #5 clk = ~clk;

  // DUT
  top_block_ctrl dut (
    .clk(clk),
    .reset(reset),
    .proj_done(proj_done),
    .qk_matmul_done(qk_matmul_done),
    .attn_reader_done(attn_reader_done),
    .linear1_done(linear1_done),
    .linear2_done(linear2_done),
    .ln_done(ln_done),
    .proj_start(proj_start),
    .qk_matmul_start(qk_matmul_start),
    .attn_reader_start(attn_reader_start),
    .linear1_start(linear1_start),
    .linear2_start(linear2_start),
    .ln_start(ln_start)
  );

  // Init drives
  initial begin
    proj_done      = 1'b0;
    proj_done      = 1'b0;
    proj_done      = 1'b0;
    qk_matmul_done   = 1'b0;
    attn_reader_done = 1'b0;
    linear1_done      = 1'b0;
    linear2_done      = 1'b0;
    ln_done          = 1'b0;
  end

  // Reset
  initial begin
    reset = 1'b1;
    repeat (3) @(posedge clk);
    reset = 1'b0;
  end


  // Stimulus: full pass through the pipeline
  initial begin
    
    @(posedge clk);
    repeat (4) @(posedge clk);
    ln_done = 1'b1;
    @(posedge clk);
    ln_done = 1'b0;

    @(posedge clk);
    repeat (4) @(posedge clk);
    proj_done = 1'b1;
    @(posedge clk);
    proj_done = 1'b0;

    @(posedge clk);
    repeat (4) @(posedge clk);
    proj_done = 1'b1;
    @(posedge clk);
    proj_done = 1'b0;

    @(posedge clk);
    repeat (4) @(posedge clk);
    proj_done = 1'b1;
    @(posedge clk);
    proj_done = 1'b0;

    // QK_MM
    @(posedge clk);
    repeat (4) @(posedge clk);
    qk_matmul_done = 1'b1;
    @(posedge clk);
    qk_matmul_done = 1'b0;

    // ATTN_R
    @(posedge clk);
    repeat (4) @(posedge clk);
    attn_reader_done = 1'b1;
    @(posedge clk);
    attn_reader_done = 1'b0;

    // OUT_PRJ (linear)
    @(posedge clk);
    repeat (4) @(posedge clk);
    proj_done = 1'b1;
    @(posedge clk);
    proj_done = 1'b0;


    // LN2
    @(posedge clk);
    repeat (4) @(posedge clk);
    ln_done = 1'b1;
    @(posedge clk);
    ln_done = 1'b0;

    // FFN1 (linear)
    @(posedge clk);
    repeat (4) @(posedge clk);
    linear1_done = 1'b1;
    @(posedge clk);
    linear1_done = 1'b0;;

    // FFN2 (linear)
    @(posedge clk);
    repeat (4) @(posedge clk);
    linear2_done = 1'b1;
    @(posedge clk);
    linear2_done = 1'b0;

    // Allow FSM to settle back to IDLE (all starts low)
    repeat (3) @(posedge clk);

    $finish;
  end

  
endmodule
