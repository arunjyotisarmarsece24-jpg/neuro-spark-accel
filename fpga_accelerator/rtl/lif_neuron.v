`timescale 1ns / 1ps
// Leaky Integrate-and-Fire (LIF) Neuromorphic Neuron Core
// Part of NeuroSpark Spiking Neural Network Engine for Spartan-7
// Biological Dynamics: Leaky Integration, Threshold Evaluation, Refractory Reset

module lif_neuron #(
    parameter MEMBRANE_WIDTH = 16,
    parameter LEAK_SHIFT     = 3,        // Decay factor: V = V - (V >> 3)
    parameter V_THRESHOLD    = 16'sd100, // Spike firing threshold
    parameter V_RESET        = 16'sd0,   // Post-spike reset potential
    parameter REFRACTORY_LEN = 2         // Cycles neuron remains non-responsive after firing
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          step_en,      // Step simulation clock
    input  wire                          clear_state,  // Reset potential & spike counters
    input  wire signed [MEMBRANE_WIDTH-1:0] syn_current,  // Integrated synaptic input: sum(W_i * S_i)
    output reg                           spike_out,    // Fired action potential
    output reg  signed [MEMBRANE_WIDTH-1:0] v_mem,        // Current membrane potential
    output reg  [7:0]                    spike_count   // Cumulative spike accumulator
);

    reg [2:0] refrac_timer = 0;

    // Leak calculation using arithmetic right shift (leak = V >> LEAK_SHIFT)
    wire signed [MEMBRANE_WIDTH-1:0] v_leak;
    assign v_leak = v_mem >>> LEAK_SHIFT;

    // Potential after synaptic integration and biological leak
    wire signed [MEMBRANE_WIDTH-1:0] v_decayed;
    assign v_decayed = v_mem - v_leak + syn_current;

    always @(posedge clk) begin
        if (rst || clear_state) begin
            v_mem        <= V_RESET;
            spike_out    <= 1'b0;
            spike_count  <= 8'd0;
            refrac_timer <= 0;
        end else if (step_en) begin
            if (refrac_timer > 0) begin
                // In biological refractory period: cannot integrate or fire
                refrac_timer <= refrac_timer - 1;
                spike_out    <= 1'b0;
                v_mem        <= V_RESET;
            end else begin
                if (v_decayed >= V_THRESHOLD) begin
                    // Membrane potential exceeded threshold: FIRE ACTION POTENTIAL!
                    spike_out    <= 1'b1;
                    v_mem        <= V_RESET; // Hard reset to resting potential
                    spike_count  <= spike_count + 1'b1;
                    refrac_timer <= REFRACTORY_LEN;
                end else begin
                    // Sub-threshold dynamics: retain integrated potential
                    spike_out <= 1'b0;
                    if (v_decayed < 16'sd0)
                        v_mem <= 16'sd0; // Clamp hyperpolarization floor to 0
                    else
                        v_mem <= v_decayed;
                end
            end
        end else begin
            spike_out <= 1'b0; // Pulse lasts 1 clock cycle
        end
    end

endmodule
