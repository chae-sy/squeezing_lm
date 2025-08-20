`timescale 1ns/1ps

module top_model_ctrl #(
    parameter integer NUM_LAYER = 12
)(
    input  wire clk,
    input  wire reset,

    input  wire start,
    input  wire linear2_done,
    input  wire out_done,

    output reg  done,
    output reg  ln_start,
    output reg  out_start
);

localparam [1:0]
    IDLE = 2'd0,
    CAL  = 2'd1,
    OUT  = 2'd2;

reg [1:0] state;
reg [$clog2(NUM_LAYER+1)-1:0] layer_cnt;



always @(posedge clk or posedge reset) begin
    if (reset) begin
        state          <= IDLE;
        done           <= 1'b0;
        ln_start       <= 1'b0;
        out_start      <= 1'b0;
        layer_cnt      <= {($clog2(NUM_LAYER+1)){1'b0}};
    end else begin
        // default: deassert 1-cycle pulses
        ln_start  <= 1'b0;
        out_start <= 1'b0;

        // sample for edge detection EVERY cycle

        case (state)
            IDLE: begin
                done      <= 1'b0;
                layer_cnt <= {($clog2(NUM_LAYER+1)){1'b0}};
                if (start) begin
                    state    <= CAL;
                    ln_start <= 1'b1; // pulse on CAL entry
                end
            end

            CAL: begin
                // Count only on rising edge
                if (linear2_done) begin
                    if (layer_cnt == NUM_LAYER-1) begin
                        layer_cnt <= {($clog2(NUM_LAYER+1)){1'b0}};
                        state     <= OUT;
                        out_start <= 1'b1; // pulse start of OUT
                    end else begin
                        layer_cnt <= layer_cnt + 1'b1;
                        ln_start  <= 1'b1; // kick next layer
                    end
                end
            end

            OUT: begin
                if (out_done) begin
                    state <= IDLE;
                    done  <= 1'b1; // one-cycle done pulse
                end
            end

            default: state <= IDLE;
        endcase
    end
end

endmodule
