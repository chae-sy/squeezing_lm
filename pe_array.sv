// -----------------------------------------------------------------------------
// pe_array.sv  —  R x C systolic MAC array (signed INT MAC)
// Protocol (per tile):
//   • Pulse start (one cycle). Provide K (inner dimension length).
//   • Keep en=1 while running. Assert clear_acc for one cycle at/after start.
//   • For the next K cycles, drive a_left_data[r] with A(r,k) for all rows r,
//     and b_top_data[c] with B(k,c) for all cols c, with valids high.
//   • After feeds stop, array flushes for (R+C-2) cycles; 'done' will assert.
// At 'done', acc_out[r][c] holds C(r,c) = sum_k A(r,k) * B(k,c).
// -----------------------------------------------------------------------------
module pe_array #(
    parameter int R  = 16,     // rows
    parameter int C  = 12,     // cols
    parameter int DW = 16,     // data width for A/B
    parameter int AW = 48,     // accumulator width
    parameter int KW = 16      // width of K (inner dimension) input
)(
    input  logic                         clk,
    input  logic                         rstn,        // active-low reset
    input  logic                         en,          // global clock-enable

    // Tile control
    input  logic                         start,       // 1-cycle pulse to start a tile
    input  logic                         clear_acc,   // 1-cycle pulse to clear all accums
    input  logic       [KW-1:0]          K,           // number of MAC steps (inner dim)

    // Left-edge A injections (one element per row, per cycle)
    input  logic signed [R-1:0][DW-1:0]  a_left_data,
    input  logic         [R-1:0]         a_left_valid,

    // Top-edge B injections (one element per column, per cycle)
    input  logic signed [C-1:0][DW-1:0]  b_top_data,
    input  logic         [C-1:0]         b_top_valid,

    // Accumulator matrix snapshot (valid at 'done')
    output logic signed [R-1:0][C-1:0][AW-1:0] acc_out,

    // Done for this tile (1-cycle pulse)
    output logic                         done
);

    // -------------------------------------------------------------------------
    // Internal buses (unpacked arrays for readability; Vivado synthesizes them)
    // A shifts left->right across columns: index [row][col_stage]
    // B shifts top->down across rows    : index [row_stage][col]
    // -------------------------------------------------------------------------
    logic signed [R-1:0][C:0][DW-1:0]  a_bus;   // C+1 taps horizontally
    logic         [R-1:0][C:0]         va_bus;

    logic signed [R:0][C-1:0][DW-1:0]  b_bus;   // R+1 taps vertically
    logic         [R:0][C-1:0]         vb_bus;

    // Connect array edges
    genvar r, c;
    generate
        for (r = 0; r < R; r++) begin : EDGE_A
            // Left edge injection
            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    a_bus[r][0]  <= '0;
                    va_bus[r][0] <= 1'b0;
                end else if (en) begin
                    a_bus[r][0]  <= a_left_data[r];
                    va_bus[r][0] <= a_left_valid[r];
                end
            end
        end
        for (c = 0; c < C; c++) begin : EDGE_B
            // Top edge injection
            always_ff @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    b_bus[0][c]  <= '0;
                    vb_bus[0][c] <= 1'b0;
                end else if (en) begin
                    b_bus[0][c]  <= b_top_data[c];
                    vb_bus[0][c] <= b_top_valid[c];
                end
            end
        end
    endgenerate

    // PE fabric
    logic signed [R-1:0][C-1:0][AW-1:0] acc_mat;
    generate
        for (r = 0; r < R; r++) begin : ROWS
            for (c = 0; c < C; c++) begin : COLS
                // Valid at this PE is the AND of A/B valids entering the PE
                wire v_here = va_bus[r][c] & vb_bus[r][c];

                pe #(.DW(DW), .AW(AW)) u_pe (
                    .clk     (clk),
                    .rstn    (rstn),
                    .en      (en),
                    .clear   (clear_acc),          // broadcast clear at tile start

                    .a_in    (a_bus[r][c]),
                    .v_a_in  (va_bus[r][c]),
                    .a_out   (a_bus[r][c+1]),
                    .v_a_out (va_bus[r][c+1]),

                    .b_in    (b_bus[r][c]),
                    .v_b_in  (vb_bus[r][c]),
                    .b_out   (b_bus[r+1][c]),
                    .v_b_out (vb_bus[r+1][c]),

                    .acc_out (acc_mat[r][c])
                );
            end
        end
    endgenerate

    // Expose the accumulator matrix continuously; snapshot is valid at 'done'
    assign acc_out = acc_mat;

    // -----------------------------------------------------------------------------
    // Tile timing: when we stream for K cycles, the systolic wavefront needs
    // (R-1)+(C-1) additional cycles to flush. 'done' pulses after that delay.
    // -----------------------------------------------------------------------------
    localparam int FLUSH_LAT = (R-1) + (C-1);

    typedef enum logic [1:0] {IDLE, RUN, FLUSH} state_t;
    state_t state, nstate;

    logic [KW-1:0] k_cnt;
    logic [$clog2(FLUSH_LAT+1)-1:0] flush_cnt;

    // State transition
    always_comb begin
        nstate = state;
        unique case (state)
            IDLE:  if (start)                 nstate = RUN;
            RUN:   if (k_cnt == K && en)      nstate = FLUSH;
            FLUSH: if (flush_cnt == FLUSH_LAT && en) nstate = IDLE;
        endcase
    end

    // Counters & done
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state     <= IDLE;
            k_cnt     <= '0;
            flush_cnt <= '0;
            done      <= 1'b0;
        end else if (en) begin
            state <= nstate;
            done  <= 1'b0;

            unique case (state)
                IDLE: begin
                    k_cnt     <= '0;
                    flush_cnt <= '0;
                end
                RUN: begin
                    if (k_cnt < K) k_cnt <= k_cnt + 1;
                end
                FLUSH: begin
                    if (flush_cnt < FLUSH_LAT) flush_cnt <= flush_cnt + 1;
                    if (flush_cnt == FLUSH_LAT) done <= 1'b1; // one-cycle pulse
                end
            endcase
        end
    end

endmodule
