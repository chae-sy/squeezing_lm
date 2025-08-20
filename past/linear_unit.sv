//======================================================================
// linear_unit.v
//  y = W��x + b �� D_OUT ��?֪��ͪߩ����⫸��?��
//======================================================================
`timescale 1ns/1ps

module linear_unit #(
    parameter D_IN   = 4,   // �Է� ���� ����
    parameter D_OUT  = 2,   // ��� ���� ����
    parameter DW     = 4,   // �Է¡�����ġ ��Ʈ��
    // accumulator ��Ʈ��: 2*DW + ceil(log2(D_IN))
    parameter ACC_W  = 2*DW + $clog2(D_IN)
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,      // ���� ���� �޽�
    input  wire                     in_valid,   // in_vec, w_mat, b_vec ��ȿ ��ȣ
    input  wire signed [DW-1:0]     in_vec   [0:D_IN-1],
    input  wire signed [DW-1:0]     w_mat    [0:D_OUT*D_IN-1], // flattened row-major
    input  wire signed [ACC_W-1:0]  b_vec    [0:D_OUT-1],

    output reg                      out_valid,  // ��� ��ȿ �޽�
    output reg signed [ACC_W-1:0]   out_vec  [0:D_OUT-1]
);

    // 1) flattened w_mat �� 2D �迭�� reshape
    wire signed [DW-1:0] w2d [0:D_OUT-1][0:D_IN-1];
    genvar o,i;
    generate
      for(o=0; o<D_OUT; o=o+1) begin: RESHAPE
        for(i=0; i<D_IN; i=i+1) begin
          assign w2d[o][i] = w_mat[o*D_IN + i];
        end
      end
    endgenerate

    // 2) pe_array ���󫹫�����
    wire [0:D_OUT-1]                 done_pe;
    wire signed [2*DW-1:0]           pe_out [0:D_OUT-1];

    pe_array #(
      .N   (D_IN),
      .DW  (DW),
      .PE_NUM(D_OUT)
    ) u_pe (
      .clk     (clk),
      .rst     (rst),
      .valid   (in_valid),
      .A_block (w2d),
      .x       (in_vec),
      .y_out   (pe_out),
      .done_out(done_pe)
    );

    // 3) ��� ���� �� bias �߰�
    reg collecting;
    always @(posedge clk or posedge rst) begin
      if (rst) begin
        collecting <= 1'b0;
        out_valid  <= 1'b0;
        for(o=0; o<D_OUT; o=o+1) out_vec[o] <= 0;
      end else begin
        out_valid <= 1'b0;
        if (start) begin
          collecting <= 1'b1;
        end else if (collecting && &done_pe) begin
          // ��� PE �Ϸ�
          for(o=0; o<D_OUT; o=o+1) begin
            // dot-product ���(pe_out) + bias
            out_vec[o] <= $signed(pe_out[o]) + b_vec[o];
          end
          out_valid  <= 1'b1;
          collecting <= 1'b0;
        end
      end
    end

endmodule
