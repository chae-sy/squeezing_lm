module pe_array #(
    parameter N = 786,
    parameter DW = 16,
    parameter PE_NUM = 8
)(
    input  wire clk,
    input  wire rst,
    input  wire valid,
    input  wire signed [DW-1:0] A_block [0:PE_NUM-1][0:N-1],  // PE_NUM rows
    input  wire signed [DW-1:0] x       [0:N-1],
    output wire signed [2*DW+$clog2(N)-1:0] y_out [0:PE_NUM-1],
    output wire [0:PE_NUM-1] done_out
);

    genvar i;
    generate
        for (i = 0; i < PE_NUM; i = i + 1) begin : pe_inst
            pe #(.N(N), .DW(DW)) pe_unit (
                .clk(clk),
                .rst(rst),
                .valid(valid),
                .row_data(A_block[i]),
                .x(x),
                .y(y_out[i]),
                .done(done_out[i])
            );
        end
    endgenerate

endmodule