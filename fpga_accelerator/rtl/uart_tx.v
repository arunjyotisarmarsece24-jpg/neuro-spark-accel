`timescale 1ns / 1ps
// Robust UART Transmitter for 12 MHz system clock, 115200 baud
// Clock divisor: 12,000,000 / 115200 = 104 cycles per bit

module uart_tx #(
    parameter CLK_FREQ = 12000000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data_in,
    input  wire       valid_in,
    output reg        tx,
    output reg        ready_out
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
    reg [7:0]  tx_byte   = 0;

    always @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            clk_count  <= 0;
            bit_index  <= 0;
            tx         <= 1'b1;
            ready_out  <= 1'b1;
        end else begin
            case (state)
                IDLE: begin
                    tx        <= 1'b1; // Idle high
                    clk_count <= 0;
                    bit_index <= 0;
                    if (valid_in) begin
                        tx_byte   <= data_in;
                        ready_out <= 1'b0;
                        state     <= START_BIT;
                    end else begin
                        ready_out <= 1'b1;
                    end
                end

                START_BIT: begin
                    tx <= 1'b0; // Start bit is low
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1;
                    end else begin
                        clk_count <= 0;
                        state     <= DATA_BITS;
                    end
                end

                DATA_BITS: begin
                    tx <= tx_byte[bit_index];
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1;
                    end else begin
                        clk_count <= 0;
                        if (bit_index < 7) begin
                            bit_index <= bit_index + 1;
                        end else begin
                            bit_index <= 0;
                            state     <= STOP_BIT;
                        end
                    end
                end

                STOP_BIT: begin
                    tx <= 1'b1; // Stop bit is high
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 1;
                    end else begin
                        clk_count <= 0;
                        state     <= CLEANUP;
                    end
                end

                CLEANUP: begin
                    ready_out <= 1'b1;
                    state     <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
