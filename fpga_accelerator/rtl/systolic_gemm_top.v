`timescale 1ns / 1ps
// Arty S7-25 Quantized Systolic Matrix Multiply Engine (BitGEMM-S7)
// Target Board: Digilent Arty S7-25 (Spartan-7 XC7S25-CSGA324)
// Frequency: 12 MHz oscillator (Pin F14)
// UART: 115200 Baud (RX: V12, TX: R12)

module systolic_gemm_top (
    input  wire       CLK12MHZ,       // 12 MHz onboard oscillator (F14)
    input  wire [3:0] btn,            // Buttons: btn[0] is Reset (G15)
    input  wire [3:0] sw,             // Switches: sw[0] can select BitNet mode manually
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
    // 1. Heartbeat generator (1 Hz blink on LED 0)
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
    // 3. Matrix Buffers (4x4 INT8 matrices = 16 bytes each)
    // -----------------------------------------------------------
    reg signed [7:0]  mat_a [0:15];
    reg signed [7:0]  mat_b [0:15];
    reg signed [15:0] mat_c [0:15];

    // State machine definition
    localparam S_IDLE       = 4'd0;
    localparam S_RX_DATA    = 4'd1;
    localparam S_COMPUTE    = 4'd2;
    localparam S_TX_PING    = 4'd3;
    localparam S_TX_PING_2  = 4'd4;
    localparam S_TX_STATUS  = 4'd5;
    localparam S_TX_RESULT  = 4'd6;
    localparam S_TX_WAIT    = 4'd7;

    reg [3:0] state = S_IDLE;
    reg [5:0] rx_byte_idx = 0;
    reg [4:0] compute_idx = 0;
    reg [5:0] tx_byte_idx = 0;
    reg       is_bitnet   = 1'b0;

    // Status registers for LEDs
    reg is_computing = 1'b0;
    reg is_rxing     = 1'b0;
    reg is_txing     = 1'b0;

    // -----------------------------------------------------------
    // 4. Matrix Multiplication Compute Engine (4x4 = 16 dot products)
    // -----------------------------------------------------------
    // C[row, col] = sum_{k=0..3} A[row*4 + k] * B[k*4 + col]
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

    // Function for multiplication/ternary logic
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

    // Wire for TX indexing
    wire [3:0] tx_elem_idx = tx_byte_idx[4:1];
    wire       tx_is_msb   = tx_byte_idx[0];

    // -----------------------------------------------------------
    // 5. Main Control FSM
    // -----------------------------------------------------------
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            rx_byte_idx  <= 0;
            compute_idx  <= 0;
            tx_byte_idx  <= 0;
            tx_valid     <= 1'b0;
            is_bitnet    <= 1'b0;
            is_computing <= 1'b0;
            is_rxing     <= 1'b0;
            is_txing     <= 1'b0;
            for (i=0; i<16; i=i+1) begin
                mat_a[i] <= 8'sh00;
                mat_b[i] <= 8'sh00;
                mat_c[i] <= 16'sh0000;
            end
        end else begin
            tx_valid <= 1'b0; // Pulse tx_valid

            case (state)
                S_IDLE: begin
                    is_computing <= 1'b0;
                    is_rxing     <= 1'b0;
                    is_txing     <= 1'b0;
                    rx_byte_idx  <= 0;
                    compute_idx  <= 0;
                    tx_byte_idx  <= 0;

                    if (rx_valid) begin
                        case (rx_byte)
                            8'h01: begin // PING command
                                state <= S_TX_PING;
                            end
                            8'h02: begin // INT8 4x4 GEMM command
                                is_bitnet <= 1'b0;
                                is_rxing  <= 1'b1;
                                state     <= S_RX_DATA;
                            end
                            8'h03: begin // BitNet Ternary 4x4 GEMM command
                                is_bitnet <= 1'b1;
                                is_rxing  <= 1'b1;
                                state     <= S_RX_DATA;
                            end
                            8'h04: begin // STATUS command
                                state <= S_TX_STATUS;
                            end
                            default: state <= S_IDLE;
                        endcase
                    end
                end

                // Receive 32 bytes (16 bytes matrix A, 16 bytes matrix B)
                S_RX_DATA: begin
                    is_rxing <= 1'b1;
                    if (rx_valid) begin
                        if (rx_byte_idx < 16)
                            mat_a[rx_byte_idx] <= rx_byte;
                        else
                            mat_b[rx_byte_idx - 16] <= rx_byte;

                        if (rx_byte_idx == 31) begin
                            is_rxing     <= 1'b0;
                            is_computing <= 1'b1;
                            compute_idx  <= 0;
                            state        <= S_COMPUTE;
                        end else begin
                            rx_byte_idx <= rx_byte_idx + 1;
                        end
                    end
                end

                // Compute 16 elements of Matrix C (1 per clock cycle = 16 clock cycles)
                S_COMPUTE: begin
                    is_computing <= 1'b1;
                    mat_c[compute_idx] <= dot_product;
                    if (compute_idx == 15) begin
                        is_computing <= 1'b0;
                        is_txing     <= 1'b1;
                        tx_byte_idx  <= 0;
                        state        <= S_TX_RESULT;
                    end else begin
                        compute_idx <= compute_idx + 1;
                    end
                end

                // PING response: Byte 1 = 0xA5
                S_TX_PING: begin
                    if (tx_ready) begin
                        tx_byte  <= 8'hA5;
                        tx_valid <= 1'b1;
                        state    <= S_TX_PING_2;
                    end
                end

                // PING response: Byte 2 = 0x5A
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

                // Send 32 bytes (16 elements * 2 bytes each, Little Endian)
                S_TX_RESULT: begin
                    is_txing <= 1'b1;
                    if (tx_ready && !tx_valid) begin
                        if (!tx_is_msb)
                            tx_byte <= mat_c[tx_elem_idx][7:0];   // LSB
                        else
                            tx_byte <= mat_c[tx_elem_idx][15:8];  // MSB

                        tx_valid <= 1'b1;

                        if (tx_byte_idx == 31) begin
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
    // 6. Visual Pulse Stretchers for Human-Visible LEDs (~87 ms)
    // -----------------------------------------------------------
    // Hardware compute is 16 cycles (1.3 microseconds at 12 MHz).
    // Without pulse stretching, 1.3 microseconds is invisible to human eyes!
    // A 20-bit counter (2^20 = 1,048,576 cycles / 12 MHz = ~87 ms).
    reg [19:0] rx_stretch   = 0;
    reg [19:0] comp_stretch = 0;
    reg [19:0] tx_stretch   = 0;
    reg        last_is_bitnet = 0;

    always @(posedge clk) begin
        if (rst) begin
            rx_stretch     <= 0;
            comp_stretch   <= 0;
            tx_stretch     <= 0;
            last_is_bitnet <= 0;
        end else begin
            if (is_rxing)
                rx_stretch <= 20'hFFFFF;
            else if (rx_stretch > 0)
                rx_stretch <= rx_stretch - 1;

            if (is_computing) begin
                comp_stretch   <= 20'hFFFFF;
                last_is_bitnet <= is_bitnet;
            end else if (comp_stretch > 0) begin
                comp_stretch <= comp_stretch - 1;
            end

            if (is_txing)
                tx_stretch <= 20'hFFFFF;
            else if (tx_stretch > 0)
                tx_stretch <= tx_stretch - 1;
        end
    end

    wire rx_vis   = (rx_stretch > 0);
    wire comp_vis = (comp_stretch > 0);
    wire tx_vis   = (tx_stretch > 0);

    // -----------------------------------------------------------
    // 7. LED Status Mapping
    // -----------------------------------------------------------
    // LD2 (led[0]): Heartbeat (blinks continuously at ~1 Hz)
    // LD3 (led[1]): UART RX Activity (stretched ~87 ms)
    // LD4 (led[2]): Hardware Compute Active (stretched ~87 ms)
    // LD5 (led[3]): UART TX Activity (stretched ~87 ms)
    assign led[0] = heartbeat;
    assign led[1] = rx_vis;
    assign led[2] = comp_vis;
    assign led[3] = tx_vis;

    // RGB LED 0 (LD0):
    // Red on Reset, Blue during RX activity, Green when Idle
    assign led0_r = rst;
    assign led0_g = (!rst && !rx_vis);
    assign led0_b = rx_vis;

    // RGB LED 1 (LD1):
    // Cyan (G+B) for INT8 compute, Magenta (R+B) for BitNet compute
    assign led1_r = (comp_vis && last_is_bitnet);
    assign led1_g = (comp_vis && !last_is_bitnet);
    assign led1_b = comp_vis;

endmodule
