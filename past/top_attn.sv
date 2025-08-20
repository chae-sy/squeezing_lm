//======================================================================
// top_attention.v
//  Top-level FSM for one step of Transformer Attention
//======================================================================
`timescale 1ns/1ps
module top_attention #(
  parameter D       = 64,   // head_dim
  parameter HEADS   = 12,   // num_heads
  parameter SEQ_LEN = 2048,   // seq length
  parameter DW      = 4,   // data bitwidth
  parameter FRAC_W  = 4    // fractional bits for softmax / rsqrt
)(
  input  wire                         clk,
  input  wire                         rst,       // sync reset
  input  wire                         start,     // begin one attention op
  // Projection inputs
  input  wire signed [DW-1:0]         in_vec     [0:D-1],
  input  wire signed [DW-1:0]         wq        [0:HEADS*D-1],
  input  wire signed [DW-1:0]         wk        [0:HEADS*D-1],
  input  wire signed [DW-1:0]         wv        [0:HEADS*D-1],
  input  wire signed [DW2-1:0]       w_proj  [0:HS*HEADS-1],
  input  wire signed [ACC_PW-1:0]    b_proj  [0:HS-1],
  // final attention output
  output reg                          out_valid,
  output reg signed [DW+FRAC_W+ $clog2(SEQ_LEN)-1:0]
                                      out_vec  [0:HEADS*D-1]  // 최종 hidden_size
);
localparam DW2    = DW+FRAC_W+$clog2(SEQ_LEN);    // attention reader 폭
  localparam HS     = HEADS*D;                     // hidden_size
  localparam ACC_PW = DW2 + $clog2(HEADS);          // proj accumulator 폭

  // FSM states
    // FSM states (with ATTN_DONE between READ_V and CPROJ)
  typedef enum logic [3:0] {
    IDLE,        // 0
    PROJ,        // 1
    WRITE_KV,    // 2
    LOAD_KV,     // 3
    WAIT_LOAD,   // 4
    QK,          // 5
    SOFTMAX,     // 6
    READ_V,      // 7
    ATTN_DONE,   // 8  ← attention reader 완료
    CPROJ,       // 9  ← final projection 단계
    DONE         // 10 ← 전체 완료
  } state_t;

  state_t state;
  integer i;

  // --------------------------------------------------------------------
  // Signals between blocks
  // --------------------------------------------------------------------
  // proj_unit
  reg                    pu_start, pu_in_valid;
  wire                   pu_out_valid;
  wire signed [2*DW-1:0] q_vec      [0:HEADS-1],
                         k_vec      [0:HEADS-1],
                         v_vec      [0:HEADS-1];
  // proj counter
  reg [$clog2(D)-1:0] proj_cnt;
  // kv_cache (two instances: k_cache, v_cache)
  reg                              kv_we;
  reg  [$clog2(SEQ_LEN)-1:0]       kv_addr;
  reg  [HEADS*DW-1:0]              kv_din;
  wire [HEADS*DW-1:0]              k_cache_dout, v_cache_dout;
  wire                             k_cache_val, v_cache_val;

  // load buffers
  reg  [$clog2(SEQ_LEN)-1:0]       load_idx;
  reg  [HEADS*DW-1:0]              k_mat [0:SEQ_LEN-1],
                                   v_mat [0:SEQ_LEN-1];

  // qk_matmul
  reg                    qk_start;
  wire                   qk_done;
  wire signed [2*DW-1:0] score [0:SEQ_LEN-1];

  // softmax_unit
  reg                    sm_start;
  wire                   sm_done;
  wire [FRAC_W-1:0]      pmf   [0:SEQ_LEN-1];

  // attn_reader
  reg                    ar_start;
  wire                   ar_done;
  wire signed [DW+FRAC_W+$clog2(SEQ_LEN)-1:0]
                         attn_out [0:HEADS-1];

  integer i;

  // --------------------------------------------------------------------
  // 1) Projection: Q/K/V
  // --------------------------------------------------------------------
  proj_unit #(.N(D), .DW(DW), .PE_NUM(HEADS)) U_proj (
    .clk(clk), .rst(rst),
    .start(pu_start),
    .in_valid(pu_in_valid),
    .in_vec(in_vec),
    .w_q(wq), .w_k(wk), .w_v(wv),
    .out_valid(pu_out_valid),
    .out_q(q_vec), .out_k(k_vec), .out_v(v_vec)
  );

  // --------------------------------------------------------------------
  // 2) KV Cache
  // --------------------------------------------------------------------
  kv_cache #(.MAX_SEQ_LEN(SEQ_LEN), .HEAD_DIM(HEADS), .DW(DW)) k_cache (
    .clk(clk), .rst(rst),
    .write_en(kv_we), .write_addr(kv_addr),
    .data_in(kv_din),
    .read_addr(load_idx), .data_out(k_cache_dout),
    .valid_out(k_cache_val)
  );
  kv_cache #(.MAX_SEQ_LEN(SEQ_LEN), .HEAD_DIM(HEADS), .DW(DW)) v_cache (
    .clk(clk), .rst(rst),
    .write_en(kv_we), .write_addr(kv_addr),
    .data_in(kv_din),  // same din but from v_vec
    .read_addr(load_idx), .data_out(v_cache_dout),
    .valid_out(v_cache_val)
  );

  // --------------------------------------------------------------------
  // 3) Q·K?
  // --------------------------------------------------------------------
  // reshape k_mat to [SEQ_LEN][D]
  qk_matmul #(.HEAD_DIM(D), .SEQ_LEN(SEQ_LEN), .DW(DW)) U_qk (
    .clk(clk), .rst(rst),
    .start(qk_start),
    .q_vec(q_vec),
    .k_mat(k_mat),
    .done(qk_done),
    .score(score)
  );

  // --------------------------------------------------------------------
  // 4) Softmax
  // --------------------------------------------------------------------
  softmax_unit #(.SEQ_LEN(SEQ_LEN), .IN_W(2*DW), .FRAC_W(FRAC_W),
                 .EXP_LUT_AW(4), .EXP_LUT_DW(2*DW+FRAC_W),
                 .RECIP_LUT_AW(4), .RECIP_LUT_DW(FRAC_W)) U_sm (
    .clk(clk), .rst(rst),
    .start(sm_start),
    .scores(score),
    .done(sm_done),
    .out(pmf)
  );

  // --------------------------------------------------------------------
  // 5) Attention Reader
  // --------------------------------------------------------------------
 wire signed [DW2-1:0] attn_out [0:HEADS-1];
  wire                  ar_done;

  attn_reader #(
    .SEQ_LEN (SEQ_LEN),
    .HEAD_DIM(HEADS),
    .DW      (DW),
    .FRAC_W  (FRAC_W)
  ) U_ar (
    .clk      (clk), .rst(rst),
    .start    (ar_start),
    .score    (pmf),
    .v_mat    (v_mat),
    .out_valid(ar_done),
    .out_vec  (attn_out)
  );
 // --------------------------------------------------------------------
  // 6) Final projection (c_proj) : HEADS → HS
  // --------------------------------------------------------------------
  // w_proj, b_proj 는 HS×HEADS 크기
  

  reg                                  proj2_start, proj2_in_valid;
  wire                                 proj2_out_valid;
  wire signed [ACC_PW-1:0]             proj2_out   [0:HS-1];

  linear_unit #(
    .D_IN  (HEADS),
    .D_OUT (HS),
    .DW    (DW2),
    .ACC_W (ACC_PW)
  ) U_cproj (
    .clk       (clk), 
    .rst       (rst),
    .start     (proj2_start),
    .in_valid  (proj2_in_valid),
    .in_vec    (attn_out),
    .w_mat     (w_proj),
    .b_vec     (b_proj),
    .out_valid (proj2_out_valid),
    .out_vec   (proj2_out)
  );

  // --------------------------------------------------------------------
  // FSM
  // --------------------------------------------------------------------
  reg [$clog2(SEQ_LEN)-1:0] kv_addr;  // token index
reg [$clog2(D)-1:0]       proj_cnt;
     

      //=========================================================
// inside top_attention, main FSM
//=========================================================
always @(posedge clk or posedge rst) begin
  if (rst) begin
    // reset all control signals and counters
    state        <= IDLE;
    pu_start     <= 1'b0;
    pu_in_valid  <= 1'b0;
    proj_cnt     <= 0;
    kv_we        <= 1'b0;
    kv_addr      <= 0;
    load_idx     <= 0;
    qk_start     <= 1'b0;
    sm_start     <= 1'b0;
    ar_start     <= 1'b0;
    out_valid    <= 1'b0;
  end else begin
    // default: deassert all pulsed signals
    pu_start     <= 1'b0;
    pu_in_valid  <= 1'b0;
    kv_we        <= 1'b0;
    qk_start     <= 1'b0;
    sm_start     <= 1'b0;
    ar_start     <= 1'b0;
    out_valid    <= 1'b0;

    case (state)
      //----------------------------------------
      IDLE: begin
        if (start) begin
          // start projection of first token
          state        <= PROJ;
          proj_cnt     <= 0;
          pu_start     <= 1'b1;
          pu_in_valid  <= 1'b1;
        end
      end

      //----------------------------------------
      PROJ: begin
        // feed one dimension per cycle
        pu_start     <= 1'b1;
        pu_in_valid  <= 1'b1;

        if (proj_cnt == D-1) begin
          // after D cycles, projection done → write KV
          state    <= WRITE_KV;
        end else begin
          proj_cnt <= proj_cnt + 1;
        end
      end

      //----------------------------------------
      WRITE_KV: begin
        // pack all HEADS k_vec into kv_din once
        kv_din   <= { k_vec[0], k_vec[1], k_vec[2],  k_vec[3],
                      k_vec[4], k_vec[5], k_vec[6],  k_vec[7],
                      k_vec[8], k_vec[9], k_vec[10], k_vec[11] };
        kv_we    <= 1'b1;
        kv_addr  <= kv_addr + 1;   // next token slot
        state    <= LOAD_KV;
      end

      //----------------------------------------
      LOAD_KV: begin
        // kick off cache-to-array loading
        state <= WAIT_LOAD;
      end

      //----------------------------------------
      WAIT_LOAD: begin
        // wait until both k_cache and v_cache produce valid data
        if (k_cache_val && v_cache_val) begin
          k_mat[load_idx] <= k_cache_dout;
          v_mat[load_idx] <= v_cache_dout;
          if (load_idx == SEQ_LEN-1) begin
            state    <= QK;
          end else begin
            load_idx <= load_idx + 1;
          end
        end
      end

      //----------------------------------------
      QK: begin
        // start Q·K? matmul
        qk_start <= 1'b1;
        state    <= SOFTMAX;
      end

      //----------------------------------------
      SOFTMAX: begin
        if (qk_done) begin
          // when QK done, start softmax
          sm_start <= 1'b1;
          state    <= READ_V;
        end
      end

      //----------------------------------------
      READ_V: begin
        if (sm_done) begin
          // when softmax done, start attention read
          ar_start <= 1'b1;
          state    <= ATTN_DONE;
        end
      end

      //----------------------------------------
      ATTN_DONE: begin
        if (ar_done) begin
          // finally latch multi-head outputs
          for (i = 0; i < HEADS; i = i + 1)
            out_vec[i] <= attn_out[i];
          out_valid <= 1'b1;
          state     <= IDLE;  // ready for next token
        end
      end

      //----------------------------------------
      default: state <= IDLE;
    endcase
  end
end

  

endmodule
