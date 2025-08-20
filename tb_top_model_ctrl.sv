`timescale 1ns/1ps

module tb_top_model_ctrl;

  // ----------------------------------------------------------------
  // Parameters
  // ----------------------------------------------------------------
  localparam integer NUM_LAYER = 12;
  localparam CLK_PERIOD = 10; // 100 MHz

  // ----------------------------------------------------------------
  // DUT I/O
  // ----------------------------------------------------------------
  reg  clk;
  reg  reset;
  reg  start;
  reg  linear2_done;
  reg  out_done;
  wire done;
  wire ln_start;
  wire out_start;

  // ----------------------------------------------------------------
  // Instantiate DUT
  // ----------------------------------------------------------------
  top_model_ctrl #(
    .NUM_LAYER(NUM_LAYER)
  ) dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .linear2_done(linear2_done),
    .out_done(out_done),
    .done(done),
    .ln_start(ln_start),
    .out_start(out_start)
  );

  // ----------------------------------------------------------------
  // Clock
  // ----------------------------------------------------------------
  initial clk = 1'b0;
  always #(CLK_PERIOD/2) clk = ~clk;



  // ----------------------------------------------------------------
  // Stimulus
  // ----------------------------------------------------------------
  integer i;

  initial begin
    // Init
    reset            = 1'b1;
    start            = 1'b0;
    linear2_done     = 1'b0;
    out_done         = 1'b0;
    

    // Hold reset a few cycles
    repeat(1) @ (posedge clk);
    reset = 1'b0;
    repeat(5) @ (posedge clk);

    // Kick off: single-cycle start
    start = 1'b1;
     @ (posedge clk);
    start = 1'b0;

    // Now complete NUM_LAYER layers by pulsing linear2_done
    // Space them out a few cycles to emulate work
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    repeat(3) @ (posedge clk);
    linear2_done = 1'b1;
    @ (posedge clk);
    linear2_done = 1'b0;
    // After final layer, DUT should issue out_start for 1 cycle.
    // Give a few cycles, then respond with out_done pulse
    repeat(3) @ (posedge clk);
    out_done = 1'b1;
    @ (posedge clk);
    out_done = 1'b0;
    // Give the DUT time to assert done and return to IDLE
    repeat(5) @ (posedge clk);

  
    $finish;
  end

endmodule
