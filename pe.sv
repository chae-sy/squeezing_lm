// -----------------------------------------------------------------------------
// pe.sv  —  Systolic Processing Element (signed MAC)
// A moves to the right, B moves downward. Accumulator resets on clear.
// Compute: acc <= acc + (A * B) when both v_a_in and v_b_in are 1.
// -----------------------------------------------------------------------------
module pe #(
    parameter int DW = 16,          // data width for A and B
    parameter int AW = 48           // accumulator width
)(
    input  logic                 clk,
    input  logic                 rstn,       // active-low reset

    input  logic                 en,         // clock-enable for this PE
    input  logic                 clear,      // synchronous acc clear (tile start)

    // A stream (flows left -> right)
    input  logic signed [DW-1:0] a_in,
    input  logic                 v_a_in,     // valid for A
    output logic signed [DW-1:0] a_out,
    output logic                 v_a_out,

    // B stream (flows top -> bottom)
    input  logic signed [DW-1:0] b_in,
    input  logic                 v_b_in,     // valid for B
    output logic signed [DW-1:0] b_out,
    output logic                 v_b_out,

    // Accumulator output (continuously visible)
    output logic signed [AW-1:0] acc_out
);

    // Forwarding regs (one-cycle shift to neighbors)
    logic signed [DW-1:0] a_reg, b_reg;
    logic                 v_a_reg, v_b_reg;

    // Accumulator
    logic signed [AW-1:0] acc_q, acc_d;

    // Do MAC only when both streams carry valid data in this cycle
    wire do_mac = v_a_in & v_b_in;

    // Accumulate with current-cycle inputs (combinational multiply, DSP mapped)
    always_comb begin
        acc_d = acc_q;
        if (clear) begin
            acc_d = '0;
        end else if (do_mac) begin
            acc_d = acc_q + ( $signed(a_in) * $signed(b_in) );
        end
    end

    // Registers
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            a_reg   <= '0;
            b_reg   <= '0;
            v_a_reg <= 1'b0;
            v_b_reg <= 1'b0;
            acc_q   <= '0;
        end else if (en) begin
            a_reg   <= a_in;
            b_reg   <= b_in;
            v_a_reg <= v_a_in;
            v_b_reg <= v_b_in;
            acc_q   <= acc_d;
        end
    end

    // Forward to neighbors & expose accumulator
    assign a_out  = a_reg;
    assign b_out  = b_reg;
    assign v_a_out = v_a_reg;
    assign v_b_out = v_b_reg;
    assign acc_out = acc_q;

endmodule
