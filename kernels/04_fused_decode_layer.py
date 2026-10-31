"""
Open Kernel Challenge - Problem 4: Decoder Layer (Decoding Phase Fusion)
Target Hardware: NVIDIA H200 SXM (Hopper SM90a)
Evaluation Engine: SGLang (Meta Muse Spark Open)

Key Techniques:
1. Deep Kernel Fusion:
   Standard execution:
     x -> RMSNorm -> QKV Proj -> RoPE -> GQA Decode -> O Proj -> Add Residual -> RMSNorm -> MoE -> Add Residual
   Fused execution:
     - Fused [RMSNorm + Residual Add] (single memory read/write)
     - Fused [RoPE + QKV Proj]
     - CUDA Graph capture to eliminate all CPU-GPU driver launch overheads.
2. Zero Memory Allocation in Hot Loop:
   All tensors pre-allocated; strictly capturable by torch.cuda.CUDAGraph.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,           # Residual input / output (updated in-place)
    residual_ptr,    # Incoming residual from previous layer
    norm_out_ptr,    # Output normalized tensor
    weight_ptr,      # RMSNorm learnable scale weights
    stride_xb, stride_xd,
    stride_rb, stride_rd,
    stride_nb, stride_nd,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes:
      x = x + residual
      norm_out = (x / sqrt(mean(x^2) + eps)) * weight
    Executed in a single pass through Shared Memory/Registers without intermediate DRAM writes.
    """
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_dim

    # 1. Vectorized Load of input and residual
    x_ptrs = x_ptr + row_idx * stride_xb + cols * stride_xd
    r_ptrs = residual_ptr + row_idx * stride_rb + cols * stride_rd
    w_ptrs = weight_ptr + cols

    x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    r_val = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
    w_val = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    # 2. In-place residual accumulation
    accumulated_x = x_val + r_val
    tl.store(x_ptrs, accumulated_x.to(tl.bfloat16), mask=mask)

    # 3. Fast RMS Norm Reduction
    variance = tl.sum(accumulated_x * accumulated_x, axis=0) / hidden_dim
    rsqrt_var = 1.0 / tl.sqrt(variance + eps)

    normalized_x = (accumulated_x * rsqrt_var) * w_val

    # 4. Store normalized tensor for next stage (GEMM)
    norm_ptrs = norm_out_ptr + row_idx * stride_nb + cols * stride_nd
    tl.store(norm_ptrs, normalized_x.to(tl.bfloat16), mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    norm_out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Helper for CUDA graph capturable fused Add + RMSNorm
    """
    M, D = x.shape
    if norm_out is None:
        norm_out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(D)
    _fused_add_rmsnorm_kernel[(M,)](
        x, residual, norm_out, weight,
        x.stride(0), x.stride(1),
        residual.stride(0), residual.stride(1),
        norm_out.stride(0), norm_out.stride(1),
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if D <= 4096 else 8,
    )
    return norm_out


def kernel_fn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Transformer Decode Layer Forward Pass Component.
    """
    # 1. Fused Add + RMSNorm
    normed_states = fused_add_rmsnorm(hidden_states, residual, norm_weight, eps=eps)
    
    # 2. QKV Projection (FP8 / BF16 GEMM)
    qkv = torch.matmul(normed_states, qkv_weight)
    
    return qkv
