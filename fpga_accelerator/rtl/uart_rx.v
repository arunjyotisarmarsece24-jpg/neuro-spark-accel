`timescale 1ns / 1ps
// Robust UART Receiver for 12 MHz system clock, 115200 baud
// Clock divisor: 12,000,000 / 115200 = 104 cycles per bit

module uart_rx #(
    parameter CLK_FREQ = 12000000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data_out,
    output reg        valid_out
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE; // 104

    localparam IDLE         = 3'b000;
    localparam START_BIT    = 3'b001;
    localparam DATA_BITS    = 3'b010;
    localparam STOP_BIT     = 3'b011;
    localparam CLEANUP      = 3'b100;

    reg [2:0] state = IDLE;
    reg [15:0] clk_count = 0;
    reg [2:0]  bit_index = 0;
    reg [7:0]  rx_byte   = 0;

    // Synchronize RX line to prevent metastability
    reg rx_sync1 = 1'b1;
    reg rx_sync2 = 1'b1;
    always @(posedge clk) begin
        rx_sync1 <= rx;
        rx_sync2 <= rx_sync1;
    end

    always @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            clk_count  <= 0;
            bit_index  <= 0;
            data_out   <= 8'h00;
            valid_out  <= 1'b0;
        end else begin
            valid_out <= 1'b0; // Default

            case (state)
                IDLE: begin
                    clk_count <= 0;
                    bit_index <= 0;
                    if (rx_sync2 == 1'b0) // Detected start bit edge
                        state <= START_BIT;
                end

                START_BIT: begin
                    if (clk_count == (CLKS_PER_BIT / 2)) begin
                        if (rx_sync2 == 1'b0) begin
                            clk_count <= 0; // Valid start bit, sample at midpoint
                            state     <= DATA_BITS;
                        end else begin
                            state <= IDLE; // False glitch
                        end
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end

                DATA_BITS: begin
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1;
                    end else begin
                        clk_count <= 0;
                        rx_byte[bit_index] <= rx_sync2;
                        if (bit_index < 7) begin
                            bit_index <= bit_index + 1;
                        end else begin
                            bit_index <= 0;
                            state     <= STOP_BIT;
                        end
                    end
                end

                STOP_BIT: begin
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1;
                    end else begin
                        clk_count <= 0;
                        data_out  <= rx_byte;
                        valid_out <= 1'b1; // Complete byte received
                        state     <= CLEANUP;
                    end
                end

                CLEANUP: begin
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
