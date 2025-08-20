`timescale 1ns/1ps

module top #(
    parameter HIDDEN = 768,
    parameter ACT_BITWIDTH = 4,
    parameter WEIGHT_BITWIDTH = 4,
    parameter NUM_LAYER = 12

)(
    input clk,
    input reset,
    input start,
    output done,
    input [HIDDEN*ACT_BITWIDTH-1:0] input,
    output reg [HIDDEN*ACT_BITWIDTH-1:0] output

);

wire ln_start, ln_done;
wire proj_start, proj_done;
wire qk_matmul_start, qk_matmul_done;
wire attn_reader_start, attn_reader_done;
wire linear1_start, linear1_done;
wire linear2_start, linear2_done;
wire out_start, out_done;

top_block_ctrl u_top_block_ctrl(
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


endmodule