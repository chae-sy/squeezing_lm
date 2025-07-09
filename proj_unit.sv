module proj_unit #(
    parameter N       = 768,  // hidden_size
    parameter DW      = 4,    // bitwidth
    parameter PE_NUM  = 12    // num_heads
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    input  wire                     in_valid,
    input  wire signed [DW-1:0]     in_vec   [0:N-1],

    // flattened weight vectors: [PE_NUM x N]
    input  wire signed [DW-1:0]     w_q      [0:PE_NUM*N-1],
    input  wire signed [DW-1:0]     w_k      [0:PE_NUM*N-1],
    input  wire signed [DW-1:0]     w_v      [0:PE_NUM*N-1],

    output reg                      out_valid,
    output wire signed [2*DW-1:0]  out_q    [0:PE_NUM-1],
    output wire signed [2*DW-1:0]  out_k    [0:PE_NUM-1],
    output wire signed [2*DW-1:0]  out_v    [0:PE_NUM-1]
);

    //========================================================================
    // 1) reshape flattened weights ¡æ 2D arrays [PE_NUM][N]
    //========================================================================
    wire signed [DW-1:0] wq_mat [0:PE_NUM-1][0:N-1],
                         wk_mat [0:PE_NUM-1][0:N-1],
                         wv_mat [0:PE_NUM-1][0:N-1];

    genvar h, i;
    generate
      for (h = 0; h < PE_NUM; h = h + 1) begin : WEIGHT_RESHAPE
        for (i = 0; i < N; i = i + 1) begin
          // each row: w_q[h][i] = w_q[h*N + i]
          assign wq_mat[h][i] = w_q[h*N + i];
          assign wk_mat[h][i] = w_k[h*N + i];
          assign wv_mat[h][i] = w_v[h*N + i];
        end
      end
    endgenerate

    //========================================================================
    // 2) instantiate three pe_array: one each for Q, K, V
    //========================================================================
    wire [0:PE_NUM-1]         done_q, done_k, done_v;
    wire signed [2*DW-1:0]    yq   [0:PE_NUM-1],
                             yk   [0:PE_NUM-1],
                             yv   [0:PE_NUM-1];

    pe_array #(.N(N), .DW(DW), .PE_NUM(PE_NUM)) pe_q (
      .clk    (clk),
      .rst    (rst),
      .valid  (in_valid),
      .A_block(wq_mat),
      .x      (in_vec),
      .y_out  (yq),
      .done_out(done_q)
    );

    pe_array #(.N(N), .DW(DW), .PE_NUM(PE_NUM)) pe_k (
      .clk    (clk),
      .rst    (rst),
      .valid  (in_valid),
      .A_block(wk_mat),
      .x      (in_vec),
      .y_out  (yk),
      .done_out(done_k)
    );

    pe_array #(.N(N), .DW(DW), .PE_NUM(PE_NUM)) pe_v (
      .clk    (clk),
      .rst    (rst),
      .valid  (in_valid),
      .A_block(wv_mat),
      .x      (in_vec),
      .y_out  (yv),
      .done_out(done_v)
    );

    //========================================================================
    // 3) collect outputs when all three arrays are done
    //========================================================================
    reg collecting;
    always @(posedge clk or posedge rst) begin
      if (rst) begin
        collecting <= 1'b0;
        out_valid  <= 1'b0;
      end else begin
        if (start) begin
          collecting <= 1'b1;
          out_valid  <= 1'b0;
        end else if (collecting && &done_q && &done_k && &done_v) begin
          // all PEs in Q/K/V arrays have finished
          out_valid  <= 1'b1;
          collecting <= 1'b0;
        end else begin
          out_valid  <= 1'b0;
        end
      end
    end

    // hook up outputs
    genvar p;
    generate
      for (p = 0; p < PE_NUM; p = p + 1) begin
        assign out_q[p] = yq[p];
        assign out_k[p] = yk[p];
        assign out_v[p] = yv[p];
      end
    endgenerate

endmodule
