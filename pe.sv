module pe #(
    parameter N = 786,
    parameter DW = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire valid,
    input  wire signed [DW-1:0] row_data [0:N-1],
    input  wire signed [DW-1:0] x       [0:N-1],
    output reg  signed [2*DW-1:0] y,
    output reg  done
);

    reg [$clog2(N):0] col_idx;
    reg signed [2*DW-1:0] acc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col_idx <= 0;
            acc     <= 0;
            y       <= 0;
            done    <= 0;
        end else begin
            if (valid) begin
                acc     <= 0;
                col_idx <= 0;
                done    <= 0;
                y       <= 0;
            end else if (col_idx < N) begin
                acc     <= acc + row_data[col_idx] * x[col_idx];
                col_idx <= col_idx + 1;
                if (col_idx == N-1) begin
                    y    <= acc + row_data[N-1] * x[N-1];
                    done <= 1;
                end
            end else begin
                // ? 결과 출력 이후에는 자동 클리어
                y    <= 0;
                done <= 0;
            end
        end
    end
endmodule
