"""
Open Kernel Challenge - Problem 1: Dense Projections (FP8 GEMM)
Target Hardware: NVIDIA H200 SXM (Hopper SM90a)
Evaluation Engine: SGLang (Meta Muse Spark Open)

Key Techniques:
1. TMA (Tensor Memory Accelerator) simulation via multi-stage software pipelining.
2. Hopper L2 Cache Locality via 2D Block Swizzling (GROUP_SIZE_M).
3. FP8 / BF16 mixed-precision Tensor Core accumulation (wgmma style).
4. Strictly CUDA Graph capturable (no allocations or synchronizations).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Optimized for Hopper H200's massive 228 KB shared memory per SM:
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp8_gemm_kernel(
    # Pointers to Matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride variables for memory navigation
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton Kernel executing: C = A @ B
    A: (M, K) in FP8 / BF16
    B: (K, N) in FP8 / BF16
    C: (M, N) in BF16 / FP32
    """
    # -----------------------------------------------------------
    # 1. 2D Block Swizzling (Hopper L2 Cache Reuse)
    # -----------------------------------------------------------
    # Maps 1D program ID to a 2D grid with super-grouping to maximize L2 hits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # 2. Compute Tile Offsets
    # -----------------------------------------------------------
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers to first block
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # 3. Main Loop: Pipelined K-dimension Accumulation
    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Asynchronously load tile A and B into Shared Memory
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        
        # Tensor Core Matrix Multiply and Accumulate
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers along K dimension
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Cast back to BF16 for transformer layer consumption
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # 4. Write back result tile to HBM3e
    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def kernel_fn(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Standard interface expected by OKC evaluation harness.
    Must be CUDA graph capturable (no torch.cuda.synchronize, no .item()).
    """
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Shape mismatch: {a.shape} vs {b.shape}"

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    # 1D launch grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _fp8_gemm_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )
    return out
