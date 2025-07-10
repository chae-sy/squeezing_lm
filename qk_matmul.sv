//======================================================================
// qk_matmul.v
//  Q��K? matrix-vector multiply: produces SEQ_LEN scores
//======================================================================
`timescale 1ns/1ps

module qk_matmul #(
  parameter HEAD_DIM = 4,    // dimension of Q,K vectors
  parameter SEQ_LEN  = 3,    // number of rows in K
  parameter DW       = 4     // bitwidth of elements
)(
  input  wire                        clk,
  input  wire                        rst,       // sync reset
  input  wire                        start,     // pulse to begin
  input  wire signed [DW-1:0]        q_vec   [0:HEAD_DIM-1],
  input  wire signed [DW-1:0]        k_mat   [0:SEQ_LEN-1][0:HEAD_DIM-1],

  output reg                         done,      // pulse when all done
  output reg  signed [2*DW-1:0]      score   [0:SEQ_LEN-1]
);

  // internal counters & control
  reg [$clog2(SEQ_LEN):0] row_idx;
  reg                     pe_valid;
  wire                    pe_done;
  wire signed [2*DW-1:0]  pe_out;

  // --------------------------------------------------------------------
  // 1) �ϳ��� �� dot-product�� pe ��� ����
  // --------------------------------------------------------------------
  pe #(
    .N   (HEAD_DIM),
    .DW  (DW)
  ) dot_unit (
    .clk      (clk),
    .rst      (rst),
    .valid    (pe_valid),
    .row_data (k_mat[row_idx]),
    .x        (q_vec),
    .y        (pe_out),
    .done     (pe_done)
  );

  // --------------------------------------------------------------------
  // 2) state machine: �� row���� pe_valid �޽� �� ��� ���� �� ���� row
  // --------------------------------------------------------------------
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      row_idx  <= 0;
      pe_valid <= 1'b0;
      done     <= 1'b0;
    end else begin
      done <= 1'b0;
      if (start) begin
        row_idx  <= 0;
        pe_valid <= 1'b1;       // ù ��° row ��� ����
      end else if (pe_valid && pe_done) begin
        score[row_idx] <= pe_out;
        pe_valid       <= 1'b0;
        if (row_idx + 1 < SEQ_LEN) begin
          row_idx  <= row_idx + 1;
          pe_valid <= 1'b1;     // ���� row ���
        end else begin
          done <= 1'b1;         // ������ row���� �Ϸ�
        end
      end
    end
  end

endmodule
