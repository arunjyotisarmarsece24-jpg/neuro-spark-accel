# NeuroSpark: Tri-Mode AI Silicon Engine & Datacenter GPU Accelerators
**Author:** Arunjyoti Sarma  
**Institution:** Indian Institute of Technology (BHU) Varanasi — Research Lab  
**Target Hardware:** Digilent Arty S7-25 (AMD / Xilinx Spartan-7 `XC7S25-CSGA324`) & NVIDIA H200 SXM (141GB HBM3e)  
**Target Benchmark:** [Open Kernel Challenge](https://openkernelchallenge.dev) (Meta Muse Spark Open / SGLang)

---

## ⚡ Executive Summary
**NeuroSpark** is an end-to-end hardware-software co-design ecosystem demonstrating high-performance AI acceleration from **physical edge silicon** to **datacenter scale**:

1. **Edge Silicon (Arty S7-25 FPGA):**
   * **BitNet b1.58 Multiplierless Engine:** Zero-multiplier matrix multiplication for ternary weights $\{-1, 0, +1\}$, executing using only adders, subtractors, and multiplexers.
   * **INT8 Systolic Array:** Hardware-pipelined matrix multiplication engine utilizing DSP48E1 slices, verified against CPU NumPy.
   * **Neuromorphic SNN Foundation:** Architecture designed for temporal spike integration and Leaky Integrate-and-Fire (LIF) neurons.
   * **Hardware-Stretched Visual Telemetry:** Integrated ~90 ms pulse-stretchers driving onboard RGB and discrete LEDs to visualize memory transfers and compute bursts in real time.

2. **Datacenter Scale (NVIDIA H200 SXM):**
   * Custom **OpenAI Triton** kernels targeting Hopper SM90a architecture.
   * Multi-stage software pipelining simulating Hopper **Tensor Memory Accelerator (TMA)** asynchronous global-to-shared memory copies.
   * **Split-K FlashDecoding** over paged FP8 KV caches to eliminate memory-bandwidth bottlenecks during low-batch LLM decoding.

---

## 📐 System Architecture

```
                                  NEUROSPARK DUAL-TARGET ECOSYSTEM
                                  
      ┌────────────────────────────────────────────────────────────────────────────────────────┐
      │                                    PYTHON HOST LAYER                                   │
      │                  (PyTorch / Triton / Serial Test Harness / NumPy Reference)            │
      └───────────────────────────┬────────────────────────────────┬───────────────────────────┘
                                  │                                │
                [ USB-UART / COM5 @ 115200 Baud ]                  │ [ CUDA / Triton PyTorch Extension ]
                                  │                                │
                                  ▼                                ▼
┌──────────────────────────────────────────────────┐  ┌──────────────────────────────────────────────────┐
│             EDGE SILICON ACCELERATOR             │  │            DATACENTER GPU KERNELS                │
│       Digilent Arty S7-25 (Spartan-7 XC7S25)     │  │          NVIDIA H200 SXM (Hopper SM90a)          │
├──────────────────────────────────────────────────┤  ├──────────────────────────────────────────────────┤
│ • 12 MHz Master Clock & System Reset             │  │ • 141 GB HBM3e at 4,800 GB/s Memory Bandwidth    │
│ • UART RX/TX with 2-Stage Anti-Glitch Sync       │  │ • Asynchronous TMA Bulk Tensor Transfers        │
│ • 4x4 INT8 Systolic GEMM Core                    │  │ • WGMMA (Warp Group Matrix Multiply Accumulate) │
│ • BitNet b1.58 Zero-Multiplier Ternary Logic     │  │ • Split-K Paged Attention Decode (FlashDecode)   │
│ • 20-bit Optical Pulse Stretchers (~90 ms)       │  │ • Fused MoE W4A16 Epilogue & Activation Pipeline │
│ • 4 User LEDs (Heartbeat, RX, Compute, TX)       │  │ • Strictly CUDA Graph Capturable                 │
│ • Dual RGB LEDs for Real-Time State Telemetry    │  │ • Zero Host Synchronizations                     │
└──────────────────────────────────────────────────┘  └──────────────────────────────────────────────────┘
```

---

## 🏆 Official Open Kernel Challenge Results (Meta H200 SXM Cluster)

Verified on Meta's official evaluation cluster (`openkernelchallenge.dev` / Meta Muse Spark Open):

```text
✓ muse_bf16_proj passed  job d587705cc0174940
  1.024×  ━━━━━━━━━╋──────────────────  faster than SGLang  95%: 1.009–1.039×

  time        4,717.1 µs
  tests       9/9 passed
  output      muse_bf16_proj/profiles/submission.d587705cc0174940.log

  Official Leaderboard Standing:
  #1658  arunjyotisarmarsece24-jpg (IIT BHU Research Lab)   1.024×
         sglang reference                                    1.000×
```

### Detailed Microbenchmark Breakdown vs SGLang Reference

| Benchmark Case | Tokens ($M$) | SGLang Latency | NeuroSpark Latency | Measured Speedup | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **verify 1 · router** | 1 | 26.4 µs | **16.3 µs** | **1.626×** | 🚀 **Passed** |
| **decode 1 · o** | 1 | 7.1 µs | **6.8 µs** | **1.037×** | 🚀 **Passed** |
| **decode 32 · o** | 32 | 6.8 µs | **6.7 µs** | **1.023×** | 🚀 **Passed** |
| **decode 1 · unpooler** | 1 | 6.9 µs | **6.8 µs** | **1.018×** | 🚀 **Passed** |
| **decode 1 · pooler** | 1 | 21.6 µs | **21.3 µs** | **1.014×** | 🚀 **Passed** |
| **decode 16 · router** | 16 | 14.8 µs | **14.7 µs** | **1.013×** | 🚀 **Passed** |
| **decode 16 · o** | 16 | 6.7 µs | **6.6 µs** | **1.011×** | 🚀 **Passed** |
| **verify 32 · qkv_gate** | 32 | 19.7 µs | **19.5 µs** | **1.010×** | 🚀 **Passed** |
| **verify 16 · pooler** | 16 | 25.8 µs | **25.6 µs** | **1.009×** | 🚀 **Passed** |

---

## 🔬 Hardware Verification Results (Live Silicon Run on COM5)

The hardware bitstream (`bitgemm_s7_arty.bit`) was synthesized with **AMD Vivado 2025.2**, programmed over JTAG, and verified over physical serial interface (`COM5`):

```text
=================================================================
  NeuroSpark AI Silicon Engine Hardware Test Harness
  IIT BHU Research Lab | Open Kernel Challenge Co-Design
=================================================================
[*] Scanning 2 serial port(s)...
    Checking COM3: Intel(R) Active Management Technology - SOL (COM3)
    Checking COM5: USB Serial Port (COM5)
    [+] LOCKED onto Arty S7-25 on COM5 (Valid PING response 0xA5 0x5A)!

Connecting to COM5 at 115200 baud...
Hardware Info: Device Tag: 'ARTY', Architecture Family: Spartan-7

=================================================================
  TEST 1: 4x4 INT8 MATRIX MULTIPLICATION (HARDWARE DSP)
=================================================================
Matrix A (INT8):
[[ 7 -4  6  6]
 [ 0  1  2  1]
 [-6 -8 -5 -1]
 [ 3 -5  4  5]]

Matrix B (INT8):
[[ 7 -4  6  1]
 [-3 -1  3 -6]
 [ 4 -7 -5  1]
 [-3  0  6  3]]

Expected Result (CPU NumPy):
[[ 67 -66  36  55]
 [  2 -15  -1  -1]
 [-35  67 -41  34]
 [ 37 -35  13  52]]

Hardware Result (Spartan-7 FPGA):
[[ 67 -66  36  55]
 [  2 -15  -1  -1]
 [-35  67 -41  34]
 [ 37 -35  13  52]]

Roundtrip Latency (UART + Hardware Compute): 11.49 ms

>>> [PASSED] 100% BIT-EXACT NUMERICAL MATCH! <<<

=================================================================
  TEST 2: 4x4 BitNet b1.58 TERNARY GEMM {-1, 0, +1}
=================================================================
Activation Matrix A (INT8):
[[-6 -8  7  8]
 [ 0  8  7  7]
 [-5  2 -2 -3]
 [ 7  6 -7  3]]

Ternary Weight Matrix B (-1, 0, 1):
[[-1  1 -1  0]
 [ 0  1  0  0]
 [ 1  0 -1  1]
 [-1  0  0  1]]

Expected Result (CPU NumPy):
[[  5 -14  -1  15]
 [  0   8  -7  14]
 [  6  -3   7  -5]
 [-17  13   0  -4]]

Hardware Result (BitNet Spartan-7 Engine):
[[  5 -14  -1  15]
 [  0   8  -7  14]
 [  6  -3   7  -5]
 [-17  13   0  -4]]

Roundtrip Latency: 15.39 ms

>>> [PASSED] 100% BIT-EXACT MATCH WITH ZERO MULTIPLIERS! <<<

=================================================================
  TEST 3: 4-NEURON NEUROMORPHIC SNN (LEAKY INTEGRATE-AND-FIRE)
  Biological Spike Integration across 16 Temporal Time-Steps
=================================================================
Synaptic Weight Matrix W (4 Neurons x 4 Inputs):
[[ -8 -13  33  24]
 [ 42  43  -9  23]
 [ -5  22   8  -1]
 [ 27  29 -14  23]]

Temporal Input Spike Train (16 Time Steps):
['0110', '1000', '0100', '0110', '1111', '1111', '0010', '0011', '1111', '1000', '0011', '0100', '0010', '0011', '0011', '0010']

Expected Spike Counts (CPU Golden LIF):  [1 4 2 3]
Hardware Spike Counts (Spartan-7 Silicon): [1 4 2 3]

On-Chip Hardware Profiling Telemetry:
  * Execution Clock Cycles: 16 cycles
  * Hardware Pure Compute Latency: 1333.3 ns (1.33 µs at 12 MHz)
  * Total Roundtrip (Serial + Hardware): 15.12 ms

>>> [PASSED] 100% BIT-EXACT NEUROMORPHIC SNN SPIKE MATCH! <<<

=================================================================
  TEST 4: LIVE LED HARDWARE STREAMING BURST
  Watch your board: LD0, LD1, LD2, LD3, LD4, LD5 will pulse and dance!
=================================================================
  [Streaming Packet 01/15] Mode: BitNet b1.58 Ternary   -> LED Pulse Active
  [Streaming Packet 02/15] Mode: INT8 Standard GEMM     -> LED Pulse Active
  [Streaming Packet 03/15] Mode: Neuromorphic SNN (LIF) -> LED Pulse Active
  [Streaming Packet 04/15] Mode: BitNet b1.58 Ternary   -> LED Pulse Active
  [Streaming Packet 05/15] Mode: INT8 Standard GEMM     -> LED Pulse Active
  [Streaming Packet 06/15] Mode: Neuromorphic SNN (LIF) -> LED Pulse Active
  [Streaming Packet 07/15] Mode: BitNet b1.58 Ternary   -> LED Pulse Active
  [Streaming Packet 08/15] Mode: INT8 Standard GEMM     -> LED Pulse Active
  [Streaming Packet 09/15] Mode: Neuromorphic SNN (LIF) -> LED Pulse Active
  [Streaming Packet 10/15] Mode: BitNet b1.58 Ternary   -> LED Pulse Active
  [Streaming Packet 11/15] Mode: INT8 Standard GEMM     -> LED Pulse Active
  [Streaming Packet 12/15] Mode: Neuromorphic SNN (LIF) -> LED Pulse Active
  [Streaming Packet 13/15] Mode: BitNet b1.58 Ternary   -> LED Pulse Active
  [Streaming Packet 14/15] Mode: INT8 Standard GEMM     -> LED Pulse Active
  [Streaming Packet 15/15] Mode: Neuromorphic SNN (LIF) -> LED Pulse Active
  [Done] 15/15 Packets Streamed & Verified across Tri-Mode Silicon!

=================================================================
  ALL HARDWARE ACCELERATION TESTS PASSED (4/4)!
  Your Arty S7-25 is running verified SNN, BitNet & INT8 Silicon!
=================================================================
```

---

## 📂 Repository Structure

```
.
├── README.md                           # Main Architecture & System Documentation
├── TOMORROW_WORK_PLAN.md               # Detailed Roadmap for Next Upgrades
├── h200_hardware_cheat_sheet.md        # FPGA BRAM/DSP to Hopper SRAM/WGMMA Mapping
├── setup_wsl_and_colab.md              # Cloud & Local Linux Environment Setup
├── colab_starter_notebook.ipynb        # 1-Click Interactive Google Colab Notebook
├── test.bat                            # 1-Click Hardware Test Runner (PowerShell/CMD)
├── RUN_TEST.bat                        # Legacy Interactive Test Runner
│
├── kernels/                            # Datacenter GPU Kernels (NVIDIA H200)
│   ├── 01_dense_gemm_fp8.py            # Hopper TMA Pipelined FP8 GEMM
│   ├── 02_paged_attention_decode.py    # Split-K FlashDecode over Paged FP8 KV Cache
│   ├── 03_moe_routed_experts.py        # Grouped GEMM for MoE (W4A16 Quantization)
│   └── 04_fused_decode_layer.py        # Fused RMSNorm + Attention + MoE Epilogue
│
├── fpga_accelerator/                   # Physical Silicon FPGA Implementation
│   ├── README.md                       # Detailed Hardware Specification
│   ├── build.bat                       # Automated Vivado Batch Synthesis & Bitstream
│   ├── flash.bat                       # 1-Click JTAG Flashing Script
│   ├── constraints/
│   │   └── arty_s7_25.xdc              # Constraints targeting XC7S25-CSGA324-1
│   ├── rtl/
│   │   ├── systolic_gemm_top.v         # Top FSM, Matrix Engine & Pulse Stretchers
│   │   ├── systolic_pe.v               # INT8 & BitNet Ternary Processing Element
│   │   ├── lif_neuron.v                # Neuromorphic Leaky Integrate-and-Fire Core
│   │   ├── uart_rx.v                   # 115200 Baud Deglitched UART Receiver
│   │   └── uart_tx.v                   # 115200 Baud UART Transmitter
│   ├── host/
│   │   └── test_arty_s7.py             # Python Host Harness with Auto-Port & Auto-Flash
│   ├── scripts/
│   │   ├── build_bitstream.tcl         # Batch Synthesis & Implementation Script
│   │   └── program_fpga.tcl            # Batch JTAG Programming Script
│   └── build/
│       └── bitgemm_s7_arty.bit         # Production Hardware Bitstream (1.24 MB)
│
└── scripts/
    └── run_okc.sh                      # Shell Script for Open Kernel Challenge CLI
```

---

## 🚀 Quick Start Guide

### 1. Test Physical FPGA (Zero-Installation)
Connect your Digilent Arty S7-25 via micro-USB (Port `J10`, Jumper `JP3` set to USB).
From PowerShell in the project directory, run:
```powershell
.\test
```

### 2. Recompile Hardware Bitstream (AMD Vivado 2025.2)
To re-synthesize after editing Verilog RTL:
```cmd
cd fpga_accelerator
build.bat
flash.bat
```

### 3. Run GPU Kernels
Kernels are written in [OpenAI Triton](https://github.com/openai/triton) and can be executed directly inside PyTorch:
```python
import torch
from kernels import 01_dense_gemm_fp8

# Benchmark kernel against PyTorch baseline
```

---

## 🔒 Security & Integrity
* **Sanitized Repository:** Hardened `.gitignore` blocks all sensitive keys, tokens, `.okc` credentials, and build checkpoints.
* **Hardware Resilience:** RTL includes 2-stage input deglitching, statically bounded registers, and safe FSM fallback transitions to prevent physical lockups.

---

## 📜 License
MIT License. Built for the Open Kernel Challenge & Neuromorphic Edge Systems Research at IIT BHU.
