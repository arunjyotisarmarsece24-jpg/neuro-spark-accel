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

## 🔬 Hardware Verification Results (Live Silicon Run on COM5)

The hardware bitstream (`bitgemm_s7_arty.bit`) was synthesized with **AMD Vivado 2025.2**, programmed over JTAG, and verified over physical serial interface (`COM5`):

```text
=================================================================
  Arty S7-25 BitGEMM Accelerator Hardware Test Harness
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
[[-5  1  6 -5]
 [-8  0  3 -3]
 [-4 -6  4 -5]
 [-2  1  2  5]]

Matrix B (INT8):
[[ 3  5 -8 -4]
 [ 1 -4  0 -2]
 [ 0 -2 -5  6]
 [ 2  4  5 -8]]

Expected Result (CPU NumPy):
[[-24 -61 -15  94]
 [-30 -58  34  74]
 [-28 -24 -13  92]
 [  5   2  31 -22]]

Hardware Result (Spartan-7 FPGA):
[[-24 -61 -15  94]
 [-30 -58  34  74]
 [-28 -24 -13  92]
 [  5   2  31 -22]]

Roundtrip Latency (UART + Hardware Compute): 15.60 ms
>>> [PASSED] 100% BIT-EXACT NUMERICAL MATCH! <<<

=================================================================
  TEST 2: 4x4 BitNet b1.58 TERNARY GEMM {-1, 0, +1}
=================================================================
Activation Matrix A (INT8):
[[  9  -1  -2  -6]
 [ -5  10  -5  -8]
 [ -9  -5 -10   1]
 [  6  -9   1  -3]]

Ternary Weight Matrix B (-1, 0, 1):
[[-1  1  1  0]
 [-1  1 -1  0]
 [ 1  1 -1  0]
 [-1  0  0 -1]]

Expected Result (CPU NumPy):
[[ -4   6  12   6]
 [ -2   0 -10   8]
 [  3 -24   6  -1]
 [  7  -2  14   3]]

Hardware Result (BitNet Spartan-7 Engine):
[[ -4   6  12   6]
 [ -2   0 -10   8]
 [  3 -24   6  -1]
 [  7  -2  14   3]]

Roundtrip Latency: 15.43 ms
>>> [PASSED] 100% BIT-EXACT MATCH WITH ZERO MULTIPLIERS! <<<

=================================================================
  TEST 3: LIVE LED HARDWARE STREAMING (25 Matrix Burst)
  Watch your board: LD0, LD1, LD3, LD4, LD5 will pulse and dance!
=================================================================
  [Done] 25/25 Streaming Packets Verified 100% Bit-Exact!

=================================================================
  ALL HARDWARE ACCELERATION TESTS PASSED (3/3)!
  Your Arty S7-25 is running verified BitNet & INT8 GEMM.
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
│   │   ├── uart_rx.v                   # 115200 Baud Deglitched UART Receiver
│   │   └── uart_tx.v                   # 115200 Baud UART Transmitter
│   ├── host/
│   │   └── test_arty_s7.py             # Python Host Harness with Auto-Port Lock
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
