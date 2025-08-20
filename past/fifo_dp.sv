//======================================================================
// fifo_dp.v
//  2-deep, synchronous, single-clock FIFO
//======================================================================
`timescale 1ns/1ps

module fifo_dp #(
  parameter WIDTH = 8
)(
  input  wire             clk,
  input  wire             rst,       // sync reset
  input  wire             write_en,
  input  wire             read_en,
  input  wire [WIDTH-1:0] data_in,
  output reg  [WIDTH-1:0] data_out,
  output wire             full,
  output wire             empty
);

  // write/read 포인터 (0,1)
  reg wr_ptr, rd_ptr;
  // 저장소: 두 워드
  reg [WIDTH-1:0] mem [1:0];
  // current count
  reg [1:0] count;

  // 포인터 및 카운터 업데이트
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr   <= 1'b0;
      rd_ptr   <= 1'b0;
      count    <= 2'd0;
      data_out <= {WIDTH{1'b0}};
    end else begin
      // write
      if (write_en && !full) begin
        mem[wr_ptr] <= data_in;
        wr_ptr      <= ~wr_ptr;
      end
      // read
      if (read_en && !empty) begin
        data_out <= mem[rd_ptr];
        rd_ptr   <= ~rd_ptr;
      end
      // count update
      case ({write_en && !full, read_en && !empty})
        2'b10: count <= count + 1;  // write only
        2'b01: count <= count - 1;  // read only
        default: count <= count;    // no op or simultaneous
      endcase
    end
  end

  assign full  = (count == 2);
  assign empty = (count == 0);

endmodule
