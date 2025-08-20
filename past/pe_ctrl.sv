module matvec_fpga_parallel #(
    parameter N = 786,
    parameter DW = 16,
    parameter PE_NUM = 8
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    output reg  done,

    input  wire signed [DW-1:0] A [0:N*N-1],  // Flattened A
    input  wire signed [DW-1:0] x [0:N-1],
    output reg  signed [2*DW-1:0] y [0:N-1]
);

    // FSM states
    localparam IDLE = 2'd0, LOAD = 2'd1, WAIT_PE = 2'd2, DONE = 2'd3;
    reg [1:0] state;

    reg [$clog2(N):0] row_idx;

    wire signed [DW-1:0] A_block [0:PE_NUM-1][0:N-1];
    wire signed [2*DW-1:0] y_partial [0:PE_NUM-1];
    wire [0:PE_NUM-1] done_flags;

    reg pe_valid;

    // A 블록 준비
    genvar i, j;
    generate
        for (i = 0; i < PE_NUM; i = i + 1) begin : blk_rows
            for (j = 0; j < N; j = j + 1) begin : blk_cols
                assign A_block[i][j] = A[(row_idx + i) * N + j];
            end
        end
    endgenerate

    // PE 배열 인스턴스
    pe_array #(.N(N), .DW(DW), .PE_NUM(PE_NUM)) pe_array_inst (
        .clk(clk),
        .rst(rst),
        .valid(pe_valid),
        .A_block(A_block),
        .x(x),
        .y_out(y_partial),
        .done_out(done_flags)
    );

    integer l, k;  // loop indices

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state    <= IDLE;
            row_idx  <= 0;
            done     <= 0;
            pe_valid <= 0;
            for (l = 0; l < N; l = l + 1) begin
                y[l] <= {2*DW{1'b0}};
            end
        end else begin
            case (state)
                IDLE: begin
                    done     <= 0;
                    pe_valid <= 0;
                    row_idx  <= 0;
                    for (l = 0; l < N; l = l + 1) begin
                        y[l] <= {2*DW{1'b0}};
                    end
                    if (start) begin
                        pe_valid <= 1;
                        state <= WAIT_PE;
                    end
                end

                WAIT_PE: begin
                    pe_valid <= 0;
                    if (&done_flags) begin
                        for (k = 0; k < PE_NUM; k = k + 1) begin
                            y[row_idx + k] <= y_partial[k];
                        end
                        if (row_idx + PE_NUM >= N) begin
                            state <= DONE;
                        end else begin
                            row_idx <= row_idx + PE_NUM;
                            pe_valid <= 1;
                        end
                    end
                end

                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
