//======================================================================
// score_fifo.v
//  Parameterizable synchronous FIFO for buffering scores
//======================================================================
`timescale 1ns/1ps

module score_fifo #(
  parameter WIDTH  = 8,       // data bitwidth
  parameter DEPTH  = 4,       // FIFO depth
  parameter ADDR_W = 2        // addr width: ceil(log2(DEPTH))
)(
  input  wire             clk,
  input  wire             rst,       // sync reset
  input  wire             write_en,  // push when not full
  input  wire             read_en,   // pop when not empty
  input  wire [WIDTH-1:0] data_in,
  output reg  [WIDTH-1:0] data_out,
  output wire             full,
  output wire             empty
);

  // pointers and count
  reg [ADDR_W-1:0] wr_ptr, rd_ptr;
  reg [ADDR_W:0]   count;
  reg [WIDTH-1:0]  mem [0:DEPTH-1];

  // write/read logic
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr   <= {ADDR_W{1'b0}};
      rd_ptr   <= {ADDR_W{1'b0}};
      count    <= 0;
      data_out <= {WIDTH{1'b0}};
    end else begin
      // write
      if (write_en && !full) begin
        mem[wr_ptr] <= data_in;
        wr_ptr      <= wr_ptr + 1;
      end
      // read
      if (read_en && !empty) begin
        data_out <= mem[rd_ptr];
        rd_ptr   <= rd_ptr + 1;
      end
      // count update
      case ({write_en && !full, read_en && !empty})
        2'b10: count <= count + 1;
        2'b01: count <= count - 1;
        default:   count <= count;
      endcase
    end
  end

  assign full  = (count == DEPTH);
  assign empty = (count == 0);

endmodule
