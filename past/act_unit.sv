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
  input  wire                      in_valid,  // input ������ �ּ� ��ȿ
  input  wire [ADDR_W-1:0]         addr,      // LUT �ּ� (signed input mapped)
  output reg  [DATA_W-1:0]         data_out,  // GELU(addr) Q8.8
  output reg                       out_valid  // 1 ����Ŭ �޽�
);

  // 2^ADDR_W ������ LUT
  reg [DATA_W-1:0] lut [0:(1<<ADDR_W)-1];

  // �ʱ�ȭ: .coe ���� �ε�
  initial begin
    $readmemh("gelu_lut.coe", lut);
  end

  // pipeline: in_valid �� ����Ŭ �� ���
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
