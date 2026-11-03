# 🚀 NeuroSpark & Open Kernel Challenge Master Execution Plan
**Author:** Arunjyoti Sarma (IIT BHU Research Lab)  
**Project Path:** `C:\Users\IITBHU RESEARCH LAB\OneDrive\Desktop\Arunjyoti_Sarma\My_Coding_files\My_contest_Project\Muse Spark Open_Kernel_challenge`  
**GitHub Repository:** [https://github.com/arunjyotisarmarsece24-jpg/neuro-spark-accel](https://github.com/arunjyotisarmarsece24-jpg/neuro-spark-accel)

---

## 🏆 Current Status (Verified Live on Physical Silicon)
* [x] **Arty S7-25 Silicon Verification:** Verified 100% bit-exact parity across all modes on **COM5**.
* [x] **Mode 1 — INT8 Systolic GEMM:** Hardware DSP-backed 4x4 matrix multiplication (`0x02`).
* [x] **Mode 2 — BitNet b1.58 Ternary GEMM:** Zero-multiplier ternary matrix multiplication $\{-1, 0, +1\}$ (`0x03`).
* [x] **Mode 3 — Neuromorphic SNN LIF Core:** 4-Neuron Leaky Integrate-and-Fire core with membrane leak, firing threshold, refractory timer, and biological spike integration (`0x05`).
* [x] **Hardware Telemetry Profiler:** 32-bit cycle counter measuring exact execution nanoseconds (16 cycles = 1,333.3 ns at 12 MHz).
* [x] **Live Optical Spike Raster:** Pulse-stretchers driving LEDs LD2–LD5 to display live biological spike rasters; RGB LED 1 indicates active mode (Cyan = INT8, Magenta = BitNet, Amber = SNN).
* [x] **Self-Healing Auto-Flashing:** Host harness detects if FPGA lost SRAM volatile state and automatically flashes over Vivado JTAG.
* [x] **1-Click Test Script:** Running `.\test` in PowerShell or CMD executes the full 4-stage hardware test suite.
* [x] **Hopper H200 GPU Kernels:** 4 Triton kernels optimized for TMA software pipelining, Split-K paged attention decode, MoE grouped GEMM, and fused decode layer.

---

## 📅 Action Steps & Submission Workflow

### Step 1: Commit and Push to GitHub
```powershell
cd "C:\Users\IITBHU RESEARCH LAB\OneDrive\Desktop\Arunjyoti_Sarma\My_Coding_files\My_contest_Project\Muse Spark Open_Kernel_challenge"
git add .
git commit -m "feat: complete tri-mode AI silicon engine (INT8, BitNet, SNN LIF) and H200 Triton kernels"
git push origin main
```

---

### Step 2: Open Kernel Challenge Leaderboard Submission (Colab / Linux)
Because the `okc` CLI evaluation harness runs on Linux against Meta's remote 8× H200 SXM GPU cluster, use **Google Colab** (Option 2 in `setup_wsl_and_colab.md`):

1. Open `colab_starter_notebook.ipynb` in [Google Colab](https://colab.research.google.com).
2. Run Step 2 cell to install `okc`:
   ```bash
   !curl -fsSL https://openkernelchallenge.dev/install.sh | sh
   ```
3. Run Step 3 cell to authenticate:
   ```bash
   !okc login
   ```
   (Click the link and input your GitHub one-time code).
4. Initialize the problem:
   ```bash
   !okc init muse_dense_projections
   ```
5. Copy the optimized kernel from `kernels/01_dense_gemm_fp8.py` into `submission.py`.
6. Run `!okc check` to validate formatting.
7. Submit to the official leaderboard:
   ```bash
   !okc submit submission.py
   ```

---

## 🛠 Quick Reference Commands

* **Run Physical Hardware Verification:**
  ```powershell
  .\test
  ```
* **Re-flash Arty S7 over JTAG:**
  ```cmd
  cd fpga_accelerator
  flash.bat
  ```
* **Rebuild FPGA Bitstream (Vivado Batch):**
  ```cmd
  cd fpga_accelerator
  build.bat
  ```
