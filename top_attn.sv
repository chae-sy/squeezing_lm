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
  // final attention output
  output reg                          out_valid,
  output reg signed [DW+FRAC_W+ $clog2(SEQ_LEN)-1:0]
                                      out_vec   [0:HEADS-1]
);

  // FSM states
  typedef enum logic [3:0] {
    IDLE,         PROJ,       WRITE_KV,
    LOAD_KV,      WAIT_LOAD,  QK,
    SOFTMAX,      READ_V,     ATTN_DONE
  } state_t;
  state_t state;

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
  attn_reader #(.SEQ_LEN(SEQ_LEN), .HEAD_DIM(HEADS),
                .DW(DW), .FRAC_W(FRAC_W)) U_ar (
    .clk(clk), .rst(rst),
    .start(ar_start),
    .score(pmf),
    .v_mat(v_mat),
    .out_valid(ar_done),
    .out_vec(attn_out)
  );

  // --------------------------------------------------------------------
  // FSM
  // --------------------------------------------------------------------
  always @(posedge clk) begin
    if (rst) begin
      state      <= IDLE;
      pu_start   <= 0; pu_in_valid <= 0;
      kv_we      <= 0; kv_addr <= 0; kv_din <= 0;
      load_idx   <= 0;
      qk_start   <= 0;
      sm_start   <= 0;
      ar_start   <= 0;
      out_valid  <= 0;
      proj_cnt    <= 0;
    end else begin
      // default deassert
      pu_start   <= 0; pu_in_valid <= 0;
      kv_we      <= 0;
      qk_start   <= 0;
      sm_start   <= 0;
      ar_start   <= 0;
      out_valid  <= 0;
      

      case (state)
        IDLE: if (start) begin
          state      <= PROJ;
          pu_start   <= 1; pu_in_valid <= 1;
        end

        PROJ: if (pu_out_valid) begin
          // got q_vec, k_vec, v_vec
          pu_start    <= 1;
          pu_in_valid <= 1;

          if (proj_cnt == D-1) begin
            // D번째 사이클 끝나면 다음 단계로
            state    <= WRITE_KV;
          end else begin
            proj_cnt <= proj_cnt + 1;
          end
        end

        WRITE_KV: begin
          // write V next
          kv_din   <= v_vec[0] << DW | v_vec[1];
          kv_we    <= 1;
          state    <= LOAD_KV;
          load_idx <= 0;
        end

        LOAD_KV: begin
          // start loading caches into k_mat/v_mat arrays
          state <= WAIT_LOAD;
        end

        WAIT_LOAD: if (k_cache_val && v_cache_val) begin
          // latch both
          k_mat[load_idx] <= k_cache_dout;
          v_mat[load_idx] <= v_cache_dout;
          if (load_idx == SEQ_LEN-1) begin
            state    <= QK;
          end else begin
            load_idx <= load_idx + 1;
          end
        end

        QK: begin
          qk_start <= 1;
          state    <= SOFTMAX;
        end

        SOFTMAX: if (qk_done) begin
          sm_start <= 1;
          state    <= READ_V;
        end

        READ_V: if (sm_done) begin
          ar_start <= 1;
          state    <= ATTN_DONE;
        end

        ATTN_DONE: if (ar_done) begin
          // output attention result
          for (i = 0; i < HEADS; i = i + 1)
            out_vec[i] <= attn_out[i];
          out_valid <= 1;
          // prepare next token? here just go IDLE
          state <= IDLE;
        end
      endcase
    end
  end

endmodule
