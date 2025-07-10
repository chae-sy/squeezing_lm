//======================================================================
// top_ffn.v
//  Top-level FSM for Transformer FFN block
//======================================================================
`timescale 1ns/1ps
module top_ffn #(
  parameter D_IN    = 2,
  parameter D_FF    = 4,
  parameter DW      = 8,
  parameter ACC1_W  = DW + $clog2(D_FF),
  parameter ADDR_W  = DW,          // assume mid_vec fits into DW for LUT addr
  parameter DATA_W  = DW           // GELU output bit-width
)(
  input  wire                        clk,
  input  wire                        rst,
  input  wire                        start,

  input  wire signed [DW-1:0]        in_vec   [0:D_IN-1],
  input  wire signed [DW-1:0]        w1_mat   [0:D_FF*D_IN-1],
  input  wire signed [ACC1_W-1:0]    b1_vec   [0:D_FF-1],
  input  wire signed [DW-1:0]        w2_mat   [0:D_IN*D_FF-1],
  input  wire signed [ACC1_W-1:0]    b2_vec   [0:D_IN-1],

  output reg                         out_valid,
  output reg signed [DW-1:0]         out_vec   [0:D_IN-1]
);

  // FSM states
  typedef enum logic [2:0] {
    IDLE, LIN1, ACT, LIN2, ADD, DONE
  } state_t;
  state_t state;

  // Signals between blocks
  // 1) Linear1
  reg                     lin1_start, lin1_in_valid;
  wire                    lin1_out_valid;
  wire signed [ACC1_W-1:0] mid_vec   [0:D_FF-1];

  // 2) Activation
  reg                     act_valid;
  reg  [ADDR_W-1:0]       act_addr  [0:D_FF-1];
  wire [DATA_W-1:0]       act_out   [0:D_FF-1];
  wire                    act_done;

  // 3) Linear2
  reg                     lin2_start, lin2_in_valid;
  wire                    lin2_out_valid;
  wire signed [ACC1_W-1:0] ffn_vec   [0:D_IN-1];

  integer i;

  // --------------------------------------------------------------------
  // (1) Linear1: D_IN → D_FF
  // --------------------------------------------------------------------
  linear_unit #(
    .D_IN (D_IN), .D_OUT(D_FF),
    .DW   (DW),   .ACC_W(ACC1_W)
  ) U_lin1 (
    .clk(clk), .rst(rst),
    .start(lin1_start), .in_valid(lin1_in_valid),
    .in_vec(in_vec), .w_mat(w1_mat), .b_vec(b1_vec),
    .out_valid(lin1_out_valid), .out_vec(mid_vec)
  );

  // --------------------------------------------------------------------
  // (2) Activation (GELU via LUT)
  // --------------------------------------------------------------------
  // 하나씩 순차적으로 LUT 조회
  reg [1:0] act_idx;
  always @(posedge clk) begin
    if (rst) begin
      act_idx    <= 0;
      act_valid  <= 0;
    end else if (state==ACT) begin
      act_valid <= 1;
      act_addr[act_idx] <= mid_vec[act_idx][DW-1 -: ADDR_W]; // 상위 비트 주소화
      if (act_idx == D_FF-1) act_valid <= 0;
    end else act_valid <= 0;
  end

  generate
    genvar j;
    for (j=0; j<D_FF; j=j+1) begin : ACT_INST
      activation_unit #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) U_act (
        .clk(clk), .rst(rst),
        .in_valid((state==ACT) && (act_idx==j)),
        .addr(act_addr[j]),
        .data_out(act_out[j]),
        .out_valid() // ignore per-entry
      );
    end
  endgenerate
  // ACT_DONE: 모든 엔트리가 한 번씩 LUT 조회된 후
  assign act_done = (act_idx==D_FF-1) && act_valid==1'b0 && (state==ACT);

  // --------------------------------------------------------------------
  // (3) Linear2: D_FF → D_IN
  // --------------------------------------------------------------------
  // Act 결과를 signed DW로 truncating
  wire signed [DW-1:0] act_vec_trim [0:D_FF-1];
  generate
    for (j=0; j<D_FF; j=j+1)
      assign act_vec_trim[j] = act_out[j][DATA_W-1 -: DW];
  endgenerate

  linear_unit #(
    .D_IN (D_FF), .D_OUT(D_IN),
    .DW   (DW),   .ACC_W(ACC1_W)
  ) U_lin2 (
    .clk(clk), .rst(rst),
    .start(lin2_start), .in_valid(lin2_in_valid),
    .in_vec(act_vec_trim), .w_mat(w2_mat), .b_vec(b2_vec),
    .out_valid(lin2_out_valid), .out_vec(ffn_vec)
  );

  // --------------------------------------------------------------------
  // (4) Residual Add
  // --------------------------------------------------------------------
  wire signed [DW-1:0] res_out [0:D_IN-1];
  generate
    for (j=0; j<D_IN; j=j+1)
      assign res_out[j] = ffn_vec[j][DW-1:0] + in_vec[j];
  endgenerate

  // --------------------------------------------------------------------
  // FSM
  // --------------------------------------------------------------------
  always @(posedge clk) begin
    if (rst) begin
      state          <= IDLE;
      lin1_start     <= 0; lin1_in_valid <= 0;
      act_idx        <= 0;
      lin2_start     <= 0; lin2_in_valid <= 0;
      out_valid      <= 0;
    end else begin
      // default deassert
      lin1_start     <= 0; lin1_in_valid <= 0;
      lin2_start     <= 0; lin2_in_valid <= 0;
      out_valid      <= 0;

      case (state)
        IDLE: if (start) begin
          state       <= LIN1;
          lin1_start  <= 1; lin1_in_valid <= 1;
        end

        LIN1: if (lin1_out_valid) begin
          state       <= ACT;
          act_idx     <= 0;
        end

        ACT: if (act_done) begin
          state       <= LIN2;
          lin2_start  <= 1; lin2_in_valid <= 1;
        end else begin
          act_idx <= act_idx + 1;
        end

        LIN2: if (lin2_out_valid) begin
          state      <= ADD;
        end

        ADD: begin
          // residual add done combinationally
          for (i=0; i<D_IN; i=i+1)
            out_vec[i] <= res_out[i];
          out_valid <= 1;
          state     <= DONE;
        end

        DONE: state <= IDLE;
      endcase
    end
  end

endmodule
