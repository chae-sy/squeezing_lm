//======================================================================
// kv_cache.v
//  Dual-port KV Cache for Transformer inference
//    - Depth = MAX_SEQ_LEN
//    - Width = HEAD_DIM ¡¿ DW (flattened per head-vector)
//    - Port A: write-only
//    - Port B: read-only with read-after-write bypass
//======================================================================
`timescale 1ns/1ps

module kv_cache #(
  parameter MAX_SEQ_LEN = 8,                      // maximum sequence length
  parameter HEAD_DIM    = 4,                      // per-head vector length
  parameter DW          = 4                       // bitwidth per element
)(
  input  wire                           clk,
  input  wire                           rst,        // synchronous reset

  // ----- write port (Port A) -----
  input  wire                           write_en,   // write enable
  input  wire [$clog2(MAX_SEQ_LEN)-1:0] write_addr, // index to write
  input  wire [HEAD_DIM*DW-1:0]         data_in,    // flattened head vector

  // ----- read port (Port B) -----
  input  wire [$clog2(MAX_SEQ_LEN)-1:0] read_addr,  // index to read
  output reg  [HEAD_DIM*DW-1:0]         data_out,   // flattened head vector
  output wire                          valid_out   // high if location has been written
);

  // internal memory and write-tracking
  reg [HEAD_DIM*DW-1:0] mem     [0:MAX_SEQ_LEN-1];
  reg                   written [0:MAX_SEQ_LEN-1];

  integer i;

  // synchronous reset & read/write logic
  always @(posedge clk) begin
    if (rst) begin
      // Initialize memory and written flags
      for (i = 0; i < MAX_SEQ_LEN; i = i + 1) begin
        mem[i]     <= {HEAD_DIM*DW{1'b0}};
        written[i] <= 1'b0;
      end
      data_out <= {HEAD_DIM*DW{1'b0}};
    end else begin
      // Write port (Port A)
      if (write_en) begin
        mem[write_addr]     <= data_in;
        written[write_addr] <= 1'b1;
      end

      // Read port (Port B) with read-after-write bypass
      if (write_en && (write_addr == read_addr)) begin
        // If writing and reading same address, bypass new data immediately
        data_out <= data_in;
      end else begin
        // Otherwise, read stored memory
        data_out <= mem[read_addr];
      end
    end
  end

  // valid_out is high if slot has been written before, or is being written this cycle
  assign valid_out = written[read_addr]
                   || (write_en && (write_addr == read_addr));

endmodule
