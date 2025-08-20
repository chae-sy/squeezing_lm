// Input Buffer Module for ping-pong buffering of input feature vectors
// Parameters: HIDDEN_SIZE = 768, BITWIDTH = 4
// Double-buffer: two register banks (A/B), toggle on each load, allow read from the other

module input_buffer #(
    parameter HIDDEN_SIZE = 768,
    parameter BITWIDTH     = 4
)(
    input  wire                          clk,
    input  wire                          rst,
    // Write interface
    input  wire                          load,       // Assert to load new input_data
    input  wire [HIDDEN_SIZE*BITWIDTH-1:0] in_data,
    // Read interface (to Projection unit)
    input  wire                          proj_req,   // Assert to read buffered data
    output reg  [HIDDEN_SIZE*BITWIDTH-1:0] out_data,
    output reg                           ready       // Indicates out_data is valid
);

    // Bank select: 0 -> next write goes to bankA, read from bankB
    //              1 -> next write goes to bankB, read from bankA
    reg bank_sel;

    // Two banks of registers
    reg [HIDDEN_SIZE*BITWIDTH-1:0] bankA;
    reg [HIDDEN_SIZE*BITWIDTH-1:0] bankB;

    // Toggle bank selection on each load
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            bank_sel <= 1'b0;
        end else if (load) begin
            bank_sel <= ~bank_sel;
        end
    end

    // Write into the inactive bank
    always @(posedge clk) begin
        if (rst) begin
            bankA <= {HIDDEN_SIZE*BITWIDTH{1'b0}};
            bankB <= {HIDDEN_SIZE*BITWIDTH{1'b0}};
        end else if (load) begin
            if (~bank_sel) // bank_set = 0, write to BankA
                bankA <= in_data;
            else // bank_set = 1, write to BankB
                bankB <= in_data;
        end
    end

    // Read from the active bank on projection request
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_data <= {HIDDEN_SIZE*BITWIDTH{1'b0}};
            ready    <= 1'b0;
        end else if (proj_req) begin
            // Active bank holds the most recent data for projection
            if (bank_sel) // bank_sel = 1, read bankA
                out_data <= bankA;
            else // bank_sel = 0, read bankB
                out_data <= bankB;
            ready <= 1'b1; // set ready = 1
        end else begin
            ready <= 1'b0;
        end
    end

endmodule
