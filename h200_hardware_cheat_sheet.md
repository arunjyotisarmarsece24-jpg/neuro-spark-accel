# FPGA Engineer's Rosetta Stone to NVIDIA H200 (Hopper SM90a)
**Target:** Mastering GPU Kernel Optimization using Hardware Intuition

---

## 1. Hardware Concept Mapping

| FPGA Concept (Xilinx/Altera) | NVIDIA Hopper H200 GPU Equivalent | Why This Matters for Performance |
| :--- | :--- | :--- |
| **LUTs & Flip-Flops (Registers)** | **GPU Registers (RF)** | Fastest storage on chip (0-cycle stall). Hopper has 64K 32-bit registers per SM. Excessive register usage leads to "register spilling" into slow local memory. |
| **Block RAM (BRAM / URAM)** | **Shared Memory (SRAM / L1)** | 228 KB per SM on Hopper. Fast on-chip scratchpad shared across all 128/256 threads in a Thread Block. |
| **Dual-Port BRAM Bank Contention** | **Shared Memory Bank Conflicts** | Hopper shared memory is divided into **32 banks (4 bytes wide)**. If multiple threads access different addresses in the same bank simultaneously, accesses serialize! Use **Swizzling (XOR indexing)** to avoid conflicts. |
| **DSP48E2 Slices (MAC units)** | **Tensor Cores (WGMMA)** | 4th Gen Tensor Cores execute matrix-multiply-accumulate instructions (`wgmma.mma_async`). They execute $128 \times N \times K$ math directly from Shared Memory without passing through registers! |
| **AXI DMA / AXI DataMover** | **TMA (Tensor Memory Accelerator)** | Dedicated hardware copy engine on Hopper. Asynchronously moves multidimensional tensor tiles between HBM3e and Shared Memory with zero register usage. |
| **FIFO Handshake (`valid` / `ready`)** | **`mbarrier` (Hardware Memory Barrier)** | Hardware register on Hopper that coordinates producer-consumer pipelines between TMA loads and Tensor Core math. |
| **Systolic Array / Pipeline Stages** | **Software Pipelining (`num_stages`)** | Multi-buffering shared memory tiles (Double-buffering / Triple-buffering). While stage $N$ is being computed on Tensor Cores, stage $N+1$ is being transferred via TMA. |
| **Custom Fixed-Point (e.g. Q4.4, Q8.8)** | **FP8 (E4M3 / E5M2) & NVFP4** | Sub-byte precision arithmetic. FP8 E4M3 delivers $2\times$ throughput over BF16 and $4\times$ over FP32. |

---

## 2. NVIDIA H200 SXM Specifications (Evaluation Node)

* **Architecture:** Hopper (Compute Capability 9.0a / SM90a)
* **Streaming Multiprocessors (SMs):** 132 SMs
* **High Bandwidth Memory (HBM3e):** 141 GB
* **Memory Bandwidth:** **4.8 TB/s** (4,800 GB/second)
* **Shared Memory Capacity:** Up to **228 KB per SM** (configurable dynamically)
* **L2 Cache:** 50 MB (Shared across all SMs, high bandwidth)
* **Peak Tensor Core Compute:** ~4,000 TFLOPS (FP8)

---

## 3. The 3 Golden Rules of Kernel Optimization

### Rule 1: Arithmetic Intensity & The Roofline Model
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred from HBM}}$$
* **If Arithmetic Intensity is LOW** (e.g. Attention Decode with $M=1$ token, or Vector Add, RMSNorm):
  * You are **Memory Bandwidth Bound**.
  * Tensor Cores sit idle waiting for memory.
  * **Brutal Solution:** Maximize memory bus saturation! Use vectorized 128-bit loads (`float4` or `int4`), coalesce memory accesses, fuse operations (e.g. fuse RMSNorm + RoPE + Softmax), and use **Split-K** to split the reduction across multiple SMs so all 132 SMs are busy.

* **If Arithmetic Intensity is HIGH** (e.g. Large MatMul, Prefill Attention with $M > 512$):
  * You are **Compute Bound**.
  * Memory bus sits idle waiting for ALUs.
  * **Brutal Solution:** Maximize Tensor Core occupancy! Use large tile sizes ($128 \times 128 \times 64$), maximize `wgmma` utilization, avoid register spilling, and use FP8 precision.

---

### Rule 2: Shared Memory Swizzling (Bank Conflict Elimination)
Hopper shared memory has 32 banks, each 4 bytes wide.  
Address to bank mapping:
$$\text{Bank ID} = \left(\frac{\text{Byte Address}}{4}\right) \pmod{32}$$

If a $32 \times 32$ half-precision matrix is stored in shared memory in row-major order:
* Row 0: Bytes 0 to 63 (Banks 0 to 15, then wraps)
* Stride of 64 bytes between rows causes vertical accesses (columns) to hit the exact same bank!
* **The FPGA Fix:** Bitwise XOR permutation (Swizzle):
```python
# Swizzled Shared Memory offset calculation:
phase = (row // 8) % 8
swizzled_col = col ^ phase
```
In Triton, `triton.language` handles swizzling automatically via `triton.language.swizzle2d` or optimal layout heuristics when targeting Hopper.

---

### Rule 3: Warp Specialization on Hopper
Traditional GPUs: All threads in a block execute:
1. Load tile from DRAM to Registers
2. Write Registers to Shared Memory
3. Synchronize `__syncthreads()`
4. Compute on Tensor Cores
5. Repeat

**Hopper "Brutal" Way (Warp Specialization):**
* **Producer Warp (e.g. Warp 0):** Issues asynchronous TMA copy instructions. Writes to `mbarrier`. It never does math!
* **Consumer Warps (Warps 1, 2, 3):** Waits on `mbarrier`. When data arrives, executes `wgmma` (Tensor Core MMA) directly from Shared Memory.
* **Zero pipeline stalls:** Loading and computing happen 100% in parallel, just like a hardware FIFO in an FPGA!
