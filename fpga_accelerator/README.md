# BitGEMM-S7: Quantized Matrix Multiplication Accelerator for Arty S7-25
**Author:** Arunjyoti Sarma (IIT BHU Research Lab)  
**Target Hardware:** Digilent Arty S7-25 (AMD / Xilinx Spartan-7 `XC7S25-CSGA324-1`)  
**Clock:** 12 MHz Onboard Oscillator (Pin `F14`)  
**Interface:** USB-UART via FTDI FT2232HQ @ 115200 Baud  

---

## 1. Overview & Co-Design Link to Open Kernel Challenge
In modern LLM inference (e.g. Meta Muse Spark Open, BitNet b1.58, and AWQ/W4A16), **quantized matrix multiplication (GEMM)** is the primary bottleneck.

While our GPU track kernels (`01_dense_gemm_fp8.py`, `03_moe_routed_experts.py`) target Hopper H200 Tensor Cores with asynchronous TMA pipelines, **BitGEMM-S7** implements the exact hardware equivalent on physical silicon:
* **Systolic Compute Array:** 16 parallel dot-product accumulators computing $4 \times 4$ GEMM.
* **INT8 Standard MAC Mode:** Hardware DSP multiplication for 8-bit quantized weights.
* **BitNet b1.58 Ternary Mode:** Zero-multiplier arithmetic for ternary weights $\{-1, 0, +1\}$, computing matrix multiplies using only additions, subtractions, and multiplexers.
* **Full Hardware Telemetry:** Real-time visual status on Arty S7's 4 discrete LEDs and 2 dual RGB LEDs.

---

## 2. Arty S7-25 Hardware Connection Guide

1. **Jumper Configuration:**
   * Verify jumper **JP3** is set to **USB** (pins 1-2 shorted).
2. **Cable Connection:**
   * Connect a micro-USB cable from your Windows PC to connector **J10 (PROG/UART)** on the Arty S7 board.
   * This single cable provides:
     - 5V Power
     - JTAG programming
     - High-speed UART serial communication (COM port)
3. **Power-On Verification:**
   * The red **PWR** LED next to J10 should illuminate immediately.

---

## 3. Quick Start (1-Click Build & Run)

### Step 1: Build the Bitstream (Automated Vivado Batch Mode)
Double-click `build.bat` or run:
```cmd
build.bat
```
*(Runs synthesis, placement, routing, and generates `build/bitgemm_s7_arty.bit`)*

### Step 2: Flash the FPGA
Double-click `flash.bat` or run:
```cmd
flash.bat
```
* **Confirmation on Board:** As soon as flashing completes:
  - Green **DONE** LED illuminates.
  - **LED 0** starts blinking at 1 Hz (Heartbeat generator).
  - **RGB LED 0** glows solid Green (System Ready / Idle).

### Step 3: Run Hardware Verification from Python
Run the test harness:
```bash
python host/test_arty_s7.py
```
* The script auto-detects the Arty S7 COM port, queries the FPGA device tag, sends test matrices over UART, reads back the hardware output, and verifies 100% bit-exact correctness against NumPy!

---

## 4. Hardware Port Map & Pin Assignments

| Signal Name | Arty S7 Pin | IO Standard | Description |
| :--- | :--- | :--- | :--- |
| `CLK12MHZ` | **F14** | LVCMOS33 | 12 MHz System Clock |
| `btn[0]` | **G15** | LVCMOS33 | Active-High System Reset |
| `uart_rxd_out` | **R12** | LVCMOS33 | UART TX (FPGA $\rightarrow$ PC) |
| `uart_txd_in` | **V12** | LVCMOS33 | UART RX (PC $\rightarrow$ FPGA) |
| `led[0]` | **E18** | LVCMOS33 | Heartbeat (1 Hz Blink) |
| `led[1]` | **F13** | LVCMOS33 | UART RX Activity |
| `led[2]` | **E13** | LVCMOS33 | Hardware Compute Active |
| `led[3]` | **H15** | LVCMOS33 | UART TX Activity |
| `led0_r/g/b` | **J15 / G17 / F15** | LVCMOS33 | RGB 0: Red=Reset, Green=Idle, Blue=RX |
| `led1_r/g/b` | **E15 / F18 / E14** | LVCMOS33 | RGB 1: Cyan=INT8 Compute, Magenta=BitNet Compute |

---

## 5. Serial Command Protocol (115200 Baud, 8-N-1)

* **`0x01` (PING):** Returns `0xA5`, `0x5A`.
* **`0x02` (INT8 GEMM):** Followed by 16 bytes (Matrix A) + 16 bytes (Matrix B). Returns 32 bytes ($4 \times 4$ INT16 result).
* **`0x03` (BitNet GEMM):** Followed by 16 bytes (Matrix A) + 16 bytes (Ternary B). Returns 32 bytes ($4 \times 4$ INT16 result).
* **`0x04` (STATUS):** Returns `ARTY` (ASCII) + `0x07` (Spartan-7).
