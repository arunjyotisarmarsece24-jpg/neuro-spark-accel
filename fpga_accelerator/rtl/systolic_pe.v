`timescale 1ns / 1ps
// Systolic Processing Element (PE) for Quantized Matrix Multiplication
// Supports INT8 MAC and BitNet b1.58 Ternary Accumulation {-1, 0, +1}

module systolic_pe (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    input  wire        clear_acc,
    input  wire        is_ternary_mode, // 1 = BitNet {-1, 0, 1}, 0 = Standard INT8
    input  wire signed [7:0]  a_in,     // Activation input
    input  wire signed [7:0]  w_in,     // Weight input (or 2-bit encoded ternary: 01=+1, 11=-1, 00=0)
    output reg  signed [7:0]  a_out,    // Systolic pass-through
    output reg  signed [7:0]  w_out,    // Systolic pass-through
    output reg  signed [31:0] acc_out   // 32-bit Accumulator
);

    wire signed [15:0] mult_result;
    assign mult_result = a_in * w_in;

    always @(posedge clk) begin
        if (rst || clear_acc) begin
            a_out   <= 8'sh00;
            w_out   <= 8'sh00;
            acc_out <= 32'sh00000000;
        end else if (enable) begin
            a_out <= a_in;
            w_out <= w_in;

            if (is_ternary_mode) begin
                // BitNet b1.58 logic: zero multiplier, only addition/subtraction
                if (w_in == 8'sd1) begin
                    acc_out <= acc_out + {{24{a_in[7]}}, a_in};
                end else if (w_in == -8'sd1) begin
                    acc_out <= acc_out - {{24{a_in[7]}}, a_in};
                end
                // if w_in == 0, keep acc_out unchanged
            end else begin
                // Standard INT8 MAC (DSP or logic)
                acc_out <= acc_out + {{16{mult_result[15]}}, mult_result};
            end
        end
    end

endmodule
