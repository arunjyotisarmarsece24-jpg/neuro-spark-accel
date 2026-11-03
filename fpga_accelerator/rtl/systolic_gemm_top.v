`timescale 1ns / 1ps
// NeuroSpark Tri-Mode AI Silicon Engine for Arty S7-25
// Target Board: Digilent Arty S7-25 (Spartan-7 XC7S25-CSGA324)
// Clock: 12 MHz Onboard Oscillator (Pin F14)
// UART: 115200 Baud (RX: V12, TX: R12)
// Modes Supported:
//   1. 4x4 INT8 Systolic GEMM (Opcode 0x02)
//   2. 4x4 BitNet b1.58 Multiplierless Ternary GEMM (Opcode 0x03)
//   3. 4-Neuron Neuromorphic LIF Spiking Neural Network (Opcode 0x05)
//   4. Nanosecond Precision Hardware Cycle Profiling Counter

module systolic_gemm_top (
    input  wire       CLK12MHZ,       // 12 MHz onboard oscillator (F14)
    input  wire [3:0] btn,            // Buttons: btn[0] is Reset (G15)
    input  wire [3:0] sw,             // Switches
    input  wire       uart_txd_in,    // UART RX from PC (V12)
    output wire       uart_rxd_out,   // UART TX to PC (R12)
    output wire [3:0] led,            // 4 User LEDs
    output wire       led0_r,         // RGB LED 0
    output wire       led0_g,
    output wire       led0_b,
    output wire       led1_r,         // RGB LED 1
    output wire       led1_g,
    output wire       led1_b
);

    wire clk = CLK12MHZ;
    wire rst = btn[0];

    // -----------------------------------------------------------
    // 1. Heartbeat generator (1 Hz blink on LED 0 when idle)
    // -----------------------------------------------------------
    reg [23:0] heartbeat_cnt = 0;
    always @(posedge clk) begin
        if (rst) heartbeat_cnt <= 0;
        else heartbeat_cnt <= heartbeat_cnt + 1;
    end
    wire heartbeat = heartbeat_cnt[23]; // ~0.7 Hz toggle

    // -----------------------------------------------------------
    // 2. UART RX & TX modules
    // -----------------------------------------------------------
    wire [7:0] rx_byte;
    wire       rx_valid;
    uart_rx #(
        .CLK_FREQ(12000000),
        .BAUD_RATE(115200)
    ) u_rx (
        .clk(clk),
        .rst(rst),
        .rx(uart_txd_in),
        .data_out(rx_byte),
        .valid_out(rx_valid)
    );

    reg  [7:0] tx_byte  = 8'h00;
    reg        tx_valid = 1'b0;
    wire       tx_ready;
    uart_tx #(
        .CLK_FREQ(12000000),
        .BAUD_RATE(115200)
    ) u_tx (
        .clk(clk),
        .rst(rst),
        .data_in(tx_byte),
        .valid_in(tx_valid),
        .tx(uart_rxd_out),
        .ready_out(tx_ready)
    );

    // -----------------------------------------------------------
    // 3. Matrix & SNN Buffers (16 bytes weights, 16 bytes inputs)
    // -----------------------------------------------------------
    reg signed [7:0]  mat_a [0:15]; // Matrix A OR SNN Synaptic Weights W[4x4]
    reg signed [7:0]  mat_b [0:15]; // Matrix B OR SNN Temporal Spike Trains S_in[16 steps]
    reg signed [15:0] mat_c [0:15]; // Matrix C result

    // State machine definition
    localparam S_IDLE       = 4'd0;
    localparam S_RX_DATA    = 4'd1;
    localparam S_COMPUTE    = 4'd2;
    localparam S_TX_PING    = 4'd3;
    localparam S_TX_PING_2  = 4'd4;
    localparam S_TX_STATUS  = 4'd5;
    localparam S_TX_RESULT  = 4'd6;
    localparam S_TX_SNN     = 4'd7;

    reg [3:0] state = S_IDLE;
    reg [5:0] rx_byte_idx  = 0;
    reg [4:0] compute_idx  = 0;
    reg [5:0] tx_byte_idx  = 0;
    reg       is_bitnet    = 1'b0;
    reg       is_snn       = 1'b0;

    // Hardware Nanosecond Cycle Profiler Counter
    reg [31:0] hw_cycle_counter = 0;
    reg [31:0] final_cycle_count = 0;

    // Status registers for LEDs
    reg is_computing = 1'b0;
    reg is_rxing     = 1'b0;
    reg is_txing     = 1'b0;

    // -----------------------------------------------------------
    // 4. Matrix Multiplication Compute Engine (4x4 = 16 dot products)
    // -----------------------------------------------------------
    wire [1:0] row = compute_idx[3:2];
    wire [1:0] col = compute_idx[1:0];

    wire signed [7:0] a0 = mat_a[{row, 2'd0}];
    wire signed [7:0] a1 = mat_a[{row, 2'd1}];
    wire signed [7:0] a2 = mat_a[{row, 2'd2}];
    wire signed [7:0] a3 = mat_a[{row, 2'd3}];

    wire signed [7:0] b0 = mat_b[{2'd0, col}];
    wire signed [7:0] b1 = mat_b[{2'd1, col}];
    wire signed [7:0] b2 = mat_b[{2'd2, col}];
    wire signed [7:0] b3 = mat_b[{2'd3, col}];

    function signed [15:0] term_mult;
        input signed [7:0] a;
        input signed [7:0] b;
        input              bitnet_en;
        begin
            if (bitnet_en) begin
                if (b == 8'sd1)       term_mult = {{8{a[7]}}, a};
                else if (b == -8'sd1) term_mult = -({{8{a[7]}}, a});
                else                  term_mult = 16'sd0;
            end else begin
                term_mult = a * b;
            end
        end
    endfunction

    wire signed [15:0] p0 = term_mult(a0, b0, is_bitnet);
    wire signed [15:0] p1 = term_mult(a1, b1, is_bitnet);
    wire signed [15:0] p2 = term_mult(a2, b2, is_bitnet);
    wire signed [15:0] p3 = term_mult(a3, b3, is_bitnet);
    wire signed [15:0] dot_product = p0 + p1 + p2 + p3;

    // -----------------------------------------------------------
    // 5. Neuromorphic SNN LIF Neuron Core Array (4 Neurons)
    // -----------------------------------------------------------
    reg        snn_step_en    = 1'b0;
    reg        snn_clear_all  = 1'b0;
    wire [3:0] snn_spikes;
    wire signed [15:0] snn_v_mem [0:3];
    wire [7:0] snn_spike_counts [0:3];

    // Synaptic current computation for Neuron j at time step t
    // Current_j = sum_{i=0..3} W[j,i] * S_in[i]
    wire [3:0] cur_time_spikes = mat_b[compute_idx[3:0]][3:0];

    function signed [15:0] calc_syn_current;
        input [1:0] neuron_id;
        input [3:0] in_spikes;
        reg signed [15:0] sum;
        begin
            sum = 16'sd0;
            if (in_spikes[0]) sum = sum + {{8{mat_a[{neuron_id, 2'd0}][7]}}, mat_a[{neuron_id, 2'd0}]};
            if (in_spikes[1]) sum = sum + {{8{mat_a[{neuron_id, 2'd1}][7]}}, mat_a[{neuron_id, 2'd1}]};
            if (in_spikes[2]) sum = sum + {{8{mat_a[{neuron_id, 2'd2}][7]}}, mat_a[{neuron_id, 2'd2}]};
            if (in_spikes[3]) sum = sum + {{8{mat_a[{neuron_id, 2'd3}][7]}}, mat_a[{neuron_id, 2'd3}]};
            calc_syn_current = sum;
        end
    endfunction

    wire signed [15:0] syn_cur0 = calc_syn_current(2'd0, cur_time_spikes);
    wire signed [15:0] syn_cur1 = calc_syn_current(2'd1, cur_time_spikes);
    wire signed [15:0] syn_cur2 = calc_syn_current(2'd2, cur_time_spikes);
    wire signed [15:0] syn_cur3 = calc_syn_current(2'd3, cur_time_spikes);

    lif_neuron #(.V_THRESHOLD(16'sd60)) u_lif0 (
        .clk(clk), .rst(rst), .step_en(snn_step_en), .clear_state(snn_clear_all),
        .syn_current(syn_cur0), .spike_out(snn_spikes[0]), .v_mem(snn_v_mem[0]), .spike_count(snn_spike_counts[0])
    );
    lif_neuron #(.V_THRESHOLD(16'sd60)) u_lif1 (
        .clk(clk), .rst(rst), .step_en(snn_step_en), .clear_state(snn_clear_all),
        .syn_current(syn_cur1), .spike_out(snn_spikes[1]), .v_mem(snn_v_mem[1]), .spike_count(snn_spike_counts[1])
    );
    lif_neuron #(.V_THRESHOLD(16'sd60)) u_lif2 (
        .clk(clk), .rst(rst), .step_en(snn_step_en), .clear_state(snn_clear_all),
        .syn_current(syn_cur2), .spike_out(snn_spikes[2]), .v_mem(snn_v_mem[2]), .spike_count(snn_spike_counts[2])
    );
    lif_neuron #(.V_THRESHOLD(16'sd60)) u_lif3 (
        .clk(clk), .rst(rst), .step_en(snn_step_en), .clear_state(snn_clear_all),
        .syn_current(syn_cur3), .spike_out(snn_spikes[3]), .v_mem(snn_v_mem[3]), .spike_count(snn_spike_counts[3])
    );

    // TX Indexing
    wire [3:0] tx_elem_idx = tx_byte_idx[4:1];
    wire       tx_is_msb   = tx_byte_idx[0];

    // -----------------------------------------------------------
    // 6. Main Control FSM
    // -----------------------------------------------------------
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            state             <= S_IDLE;
            rx_byte_idx       <= 0;
            compute_idx       <= 0;
            tx_byte_idx       <= 0;
            tx_valid          <= 1'b0;
            is_bitnet         <= 1'b0;
            is_snn            <= 1'b0;
            is_computing      <= 1'b0;
            is_rxing          <= 1'b0;
            is_txing          <= 1'b0;
            snn_step_en       <= 1'b0;
            snn_clear_all     <= 1'b1;
            hw_cycle_counter  <= 0;
            final_cycle_count <= 0;
            for (i=0; i<16; i=i+1) begin
                mat_a[i] <= 8'sh00;
                mat_b[i] <= 8'sh00;
                mat_c[i] <= 16'sh0000;
            end
        end else begin
            tx_valid      <= 1'b0;
            snn_step_en   <= 1'b0;
            snn_clear_all <= 1'b0;

            case (state)
                S_IDLE: begin
                    is_computing     <= 1'b0;
                    is_rxing         <= 1'b0;
                    is_txing         <= 1'b0;
                    rx_byte_idx      <= 0;
                    compute_idx      <= 0;
                    tx_byte_idx      <= 0;
                    hw_cycle_counter <= 0;

                    if (rx_valid) begin
                        case (rx_byte)
                            8'h01: begin // PING command
                                state <= S_TX_PING;
                            end
                            8'h02: begin // INT8 4x4 GEMM command
                                is_bitnet <= 1'b0;
                                is_snn    <= 1'b0;
                                is_rxing  <= 1'b1;
                                state     <= S_RX_DATA;
                            end
                            8'h03: begin // BitNet Ternary 4x4 GEMM command
                                is_bitnet <= 1'b1;
                                is_snn    <= 1'b0;
                                is_rxing  <= 1'b1;
                                state     <= S_RX_DATA;
                            end
                            8'h04: begin // STATUS command
                                state <= S_TX_STATUS;
                            end
                            8'h05: begin // SNN LIF Spiking Neural Network command
                                is_bitnet     <= 1'b0;
                                is_snn        <= 1'b1;
                                is_rxing      <= 1'b1;
                                snn_clear_all <= 1'b1;
                                state         <= S_RX_DATA;
                            end
                            default: state <= S_IDLE;
                        endcase
                    end
                end

                // Receive 32 bytes (16 bytes weights, 16 bytes inputs/spikes)
                S_RX_DATA: begin
                    is_rxing <= 1'b1;
                    if (rx_valid) begin
                        if (rx_byte_idx < 16)
                            mat_a[rx_byte_idx] <= rx_byte;
                        else
                            mat_b[rx_byte_idx - 16] <= rx_byte;

                        if (rx_byte_idx == 31) begin
                            is_rxing         <= 1'b0;
                            is_computing     <= 1'b1;
                            compute_idx      <= 0;
                            hw_cycle_counter <= 0;
                            if (is_snn) snn_step_en <= 1'b1;
                            state            <= S_COMPUTE;
                        end else begin
                            rx_byte_idx <= rx_byte_idx + 1;
                        end
                    end
                end

                // Compute Engine: 16 clock cycles for GEMM or 16 biological time-steps for SNN
                S_COMPUTE: begin
                    is_computing     <= 1'b1;
                    hw_cycle_counter <= hw_cycle_counter + 1;

                    if (is_snn) begin
                        if (compute_idx == 15) begin
                            snn_step_en       <= 1'b0;
                            is_computing      <= 1'b0;
                            is_txing          <= 1'b1;
                            tx_byte_idx       <= 0;
                            final_cycle_count <= hw_cycle_counter + 1;
                            state             <= S_TX_SNN;
                        end else begin
                            snn_step_en <= 1'b1;
                            compute_idx <= compute_idx + 1;
                        end
                    end else begin
                        // Matrix Multiplication step
                        mat_c[compute_idx] <= dot_product;
                        if (compute_idx == 15) begin
                            is_computing      <= 1'b0;
                            is_txing          <= 1'b1;
                            tx_byte_idx       <= 0;
                            final_cycle_count <= hw_cycle_counter + 1;
                            state             <= S_TX_RESULT;
                        end else begin
                            compute_idx <= compute_idx + 1;
                        end
                    end
                end

                // PING response: 0xA5, 0x5A
                S_TX_PING: begin
                    if (tx_ready) begin
                        tx_byte  <= 8'hA5;
                        tx_valid <= 1'b1;
                        state    <= S_TX_PING_2;
                    end
                end

                S_TX_PING_2: begin
                    if (tx_ready && !tx_valid) begin
                        tx_byte  <= 8'h5A;
                        tx_valid <= 1'b1;
                        state    <= S_IDLE;
                    end
                end

                // STATUS response: 'A', 'R', 'T', 'Y', 0x07 (Spartan-7)
                S_TX_STATUS: begin
                    if (tx_ready && !tx_valid) begin
                        case (tx_byte_idx)
                            0: tx_byte <= 8'h41; // 'A'
                            1: tx_byte <= 8'h52; // 'R'
                            2: tx_byte <= 8'h54; // 'T'
                            3: tx_byte <= 8'h59; // 'Y'
                            4: tx_byte <= 8'h07; // Spartan 7
                        endcase
                        tx_valid <= 1'b1;
                        if (tx_byte_idx == 4) state <= S_IDLE;
                        else tx_byte_idx <= tx_byte_idx + 1;
                    end
                end

                // GEMM Result: 32 bytes (16 elements * 2 bytes)
                S_TX_RESULT: begin
                    is_txing <= 1'b1;
                    if (tx_ready && !tx_valid) begin
                        if (!tx_is_msb)
                            tx_byte <= mat_c[tx_elem_idx][7:0];
                        else
                            tx_byte <= mat_c[tx_elem_idx][15:8];

                        tx_valid <= 1'b1;

                        if (tx_byte_idx == 31) begin
                            state <= S_IDLE;
                        end else begin
                            tx_byte_idx <= tx_byte_idx + 1;
                        end
                    end
                end

                // SNN Result: 12 bytes
                // [0..3]: 4 Spike counts
                // [4..7]: 4 Final membrane potentials (low bytes)
                // [8..11]: 32-bit hardware cycle count
                S_TX_SNN: begin
                    is_txing <= 1'b1;
                    if (tx_ready && !tx_valid) begin
                        case (tx_byte_idx)
                            0:  tx_byte <= snn_spike_counts[0];
                            1:  tx_byte <= snn_spike_counts[1];
                            2:  tx_byte <= snn_spike_counts[2];
                            3:  tx_byte <= snn_spike_counts[3];
                            4:  tx_byte <= snn_v_mem[0][7:0];
                            5:  tx_byte <= snn_v_mem[1][7:0];
                            6:  tx_byte <= snn_v_mem[2][7:0];
                            7:  tx_byte <= snn_v_mem[3][7:0];
                            8:  tx_byte <= final_cycle_count[7:0];
                            9:  tx_byte <= final_cycle_count[15:8];
                            10: tx_byte <= final_cycle_count[23:16];
                            11: tx_byte <= final_cycle_count[31:24];
                        endcase
                        tx_valid <= 1'b1;

                        if (tx_byte_idx == 11) begin
                            state <= S_IDLE;
                        end else begin
                            tx_byte_idx <= tx_byte_idx + 1;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------
    // 7. Visual Pulse Stretchers for Human-Visible LEDs (~87 ms)
    // -----------------------------------------------------------
    reg [19:0] rx_stretch   = 0;
    reg [19:0] comp_stretch = 0;
    reg [19:0] tx_stretch   = 0;
    reg        last_is_bitnet = 0;
    reg        last_is_snn    = 0;

    // Biological Spike Flash Stretchers (so human eye sees individual spikes!)
    reg [18:0] spike_stretch [0:3];
    integer s;
    always @(posedge clk) begin
        if (rst) begin
            rx_stretch     <= 0;
            comp_stretch   <= 0;
            tx_stretch     <= 0;
            last_is_bitnet <= 0;
            last_is_snn    <= 0;
            for (s=0; s<4; s=s+1) spike_stretch[s] <= 0;
        end else begin
            if (is_rxing) rx_stretch <= 20'hFFFFF;
            else if (rx_stretch > 0) rx_stretch <= rx_stretch - 1;

            if (is_computing) begin
                comp_stretch   <= 20'hFFFFF;
                last_is_bitnet <= is_bitnet;
                last_is_snn    <= is_snn;
            end else if (comp_stretch > 0) begin
                comp_stretch <= comp_stretch - 1;
            end

            if (is_txing) tx_stretch <= 20'hFFFFF;
            else if (tx_stretch > 0) tx_stretch <= tx_stretch - 1;

            // Biological spike stretchers (~43 ms per spike)
            for (s=0; s<4; s=s+1) begin
                if (snn_spikes[s]) spike_stretch[s] <= 19'h7FFFF;
                else if (spike_stretch[s] > 0) spike_stretch[s] <= spike_stretch[s] - 1;
            end
        end
    end

    wire rx_vis   = (rx_stretch > 0);
    wire comp_vis = (comp_stretch > 0);
    wire tx_vis   = (tx_stretch > 0);
    wire [3:0] spike_vis;
    assign spike_vis[0] = (spike_stretch[0] > 0);
    assign spike_vis[1] = (spike_stretch[1] > 0);
    assign spike_vis[2] = (spike_stretch[2] > 0);
    assign spike_vis[3] = (spike_stretch[3] > 0);

    // -----------------------------------------------------------
    // 8. LED Status Mapping
    // -----------------------------------------------------------
    // In SNN mode: LEDs 0..3 display real-time biological spike raster!
    // In GEMM mode: LED 0 = Heartbeat, LED 1 = RX, LED 2 = Compute, LED 3 = TX
    assign led[0] = last_is_snn ? spike_vis[0] : heartbeat;
    assign led[1] = last_is_snn ? spike_vis[1] : rx_vis;
    assign led[2] = last_is_snn ? spike_vis[2] : comp_vis;
    assign led[3] = last_is_snn ? spike_vis[3] : tx_vis;

    // RGB LED 0: Blue on RX, Red on Reset, Green when Idle
    assign led0_r = rst;
    assign led0_g = (!rst && !rx_vis);
    assign led0_b = rx_vis;

    // RGB LED 1:
    // Cyan (G+B) = INT8 GEMM
    // Magenta (R+B) = BitNet Ternary GEMM
    // Gold/Amber (R+G) = Neuromorphic SNN LIF Spiking Core Active!
    assign led1_r = comp_vis && (last_is_bitnet || last_is_snn);
    assign led1_g = comp_vis && (!last_is_bitnet || last_is_snn);
    assign led1_b = comp_vis && !last_is_snn;

endmodule
