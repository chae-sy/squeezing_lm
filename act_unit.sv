//======================================================================
// activation_unit.v
//  Fixed-point GELU lookup via LUT
//======================================================================
`timescale 1ns/1ps

module activation_unit #(
  parameter ADDR_W  = 8,       // LUT address width
  parameter DATA_W  = 16       // LUT data width (Q8.8)
)(
  input  wire                      clk,
  input  wire                      rst,       // sync reset
  input  wire                      in_valid,  // input 고정된 주소 유효
  input  wire [ADDR_W-1:0]         addr,      // LUT 주소 (signed input mapped)
  output reg  [DATA_W-1:0]         data_out,  // GELU(addr) Q8.8
  output reg                       out_valid  // 1 사이클 펄스
);

  // 2^ADDR_W 깊이의 LUT
  reg [DATA_W-1:0] lut [0:(1<<ADDR_W)-1];

  // 초기화: .coe 파일 로드
  initial begin
    $readmemh("gelu_lut.coe", lut);
  end

  // pipeline: in_valid 한 사이클 후 출력
  always @(posedge clk) begin
    if (rst) begin
      data_out  <= {DATA_W{1'b0}};
      out_valid <= 1'b0;
    end else begin
      out_valid <= in_valid;
      if (in_valid) begin
        data_out <= lut[addr];
      end
    end
  end

endmodule
