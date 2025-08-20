//======================================================================
// mean_unit.v
//  Compute mean = (¥Ò x[i]) / D
//======================================================================
`timescale 1ns/1ps
module mean_unit #(
  parameter D       = 4,                     // vector length
  parameter DW      = 8,                     // input bit-width
  parameter ACC_W   = DW + $clog2(D)         // accumulator width
)(
  input  wire                     clk,
  input  wire                     rst,       // sync reset
  input  wire                     start,     // one-cycle pulse
  input  wire signed [DW-1:0]     x        [0:D-1],
  output reg  signed [ACC_W-1:0]  mean,
  output reg                      valid      // one-cycle pulse
);
  integer i;
  reg [ACC_W-1:0] acc;
  reg running;
  always @(posedge clk) begin
    if (rst) begin
      acc   <= 0;
      mean  <= 0;
      valid <= 0;
      running <= 0;
    end else begin
      valid <= 0;
      if (start) begin
        acc    <= 0;
        running<= 1;
        i      <= 0;
      end else if (running) begin
        acc <= acc + x[i];
        i   <= i + 1;
        if (i == D-1) begin
          mean  <= acc / D; 
          valid <= 1;
          running <= 0;
        end
      end
    end
  end
endmodule

//======================================================================
// var_unit.v
//  Compute variance = (¥Ò (x[i]-¥ì)^2) / D
//======================================================================
`timescale 1ns/1ps
module var_unit #(
  parameter D       = 4,
  parameter DW      = 8,
  parameter ACC1_W  = DW + $clog2(D),                     // sum width
  parameter ACC2_W  = 2*DW + $clog2(D)                     // variance width
)(
  input  wire                     clk,
  input  wire                     rst,
  input  wire                     start,     // pulse
  input  wire signed [DW-1:0]     x        [0:D-1],
  input  wire signed [ACC1_W-1:0] mean,      // from mean_unit
  output reg  signed [ACC2_W-1:0] vari,       // Q-format
  output reg                      valid
);
  integer i;
  reg signed [ACC2_W-1:0] acc2;
  reg running;
  always @(posedge clk) begin
    if (rst) begin
      acc2   <= 0;
      vari    <= 0;
      valid  <= 0;
      running<= 0;
    end else begin
      valid <= 0;
      if (start) begin
        acc2    <= 0;
        running <= 1;
        i       <= 0;
      end else if (running) begin
        acc2 <= acc2 + (x[i] - mean)*(x[i] - mean);
        i    <= i + 1;
        if (i == D-1) begin
          vari    <= acc2 / D;
          valid  <= 1;
          running<= 0;
        end
      end
    end
  end
endmodule

//======================================================================
// rsqrt_unit.v
//  Reciprocal sqrt via 16-entry LUT (ADDR_W=4), Q0.8 format
//======================================================================
`timescale 1ns/1ps
module rsqrt_unit #(
  parameter ADDR_W = 4,
  parameter DATA_W = 8          // fractional width Q0.8
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    in_valid,
  input  wire [ADDR_W-1:0]       addr,      // upper bits of var
  output reg  [DATA_W-1:0]       rsqrt,     // Q0.8
  output reg                     out_valid
);
  reg [DATA_W-1:0] lut [0:(1<<ADDR_W)-1];
  initial $readmemh("rsqrt_lut.coe", lut);

  always @(posedge clk) begin
    if (rst) begin
      rsqrt     <= 0;
      out_valid <= 0;
    end else begin
      out_valid <= in_valid;
      if (in_valid) rsqrt <= lut[addr];
    end
  end
endmodule
