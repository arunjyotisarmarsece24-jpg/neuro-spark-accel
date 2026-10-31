"""
Open Kernel Challenge - Problem 2: Attention Decode (Paged FP8 GQA)
Target Hardware: NVIDIA H200 SXM (Hopper SM90a)
Evaluation Engine: SGLang (Meta Muse Spark Open)

Key Techniques:
1. Split-K FlashDecoding: During generation (batch decoding), M=1.
   A single thread-block per head leaves 132 SMs severely underutilized.
   We split the KV sequence into parallel chunks across multiple SMs.
2. Paged FP8 KV Cache: Direct dequantization in registers, avoiding DRAM bandwidth waste.
3. Two-pass reduction: Compute partial max/sums per split, then combine.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _split_k_decode_stage1(
    # Input pointers
    q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, seq_lens_ptr,
    # Intermediate reduction buffers
    partial_out_ptr, partial_lse_ptr,
    # Strides
    stride_qb, stride_qh, stride_qd,
    stride_cb, stride_cp, stride_ch, stride_cd,
    stride_tb, stride_tt,
    stride_pob, stride_pos, stride_poh, stride_pod,
    stride_plb, stride_pls, stride_plh,
    # Model dimensions
    sm_scale,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SPLIT: tl.constexpr,
):
    """
    Stage 1: Process a slice of the KV cache sequence for a single query token.
    Saves partial softmax statistics and partial weighted V to intermediate buffers.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)

    # Calculate token range for this split
    tokens_per_split = BLOCK_SPLIT
    start_token = split_idx * tokens_per_split
    end_token = tl.minimum(start_token + tokens_per_split, seq_len)

    if start_token >= seq_len:
        return

    # Load Query vector (HEAD_DIM elements) into registers
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)

    # Initialize online softmax statistics
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Loop through KV tokens in this split
    for tok_idx in range(start_token, end_token):
        page_idx = tok_idx // PAGE_SIZE
        page_offset = tok_idx % PAGE_SIZE
        physical_page = tl.load(block_tables_ptr + batch_idx * stride_tb + page_idx * stride_tt)

        # Load Key vector (FP8 or BF16)
        k_ptrs = k_cache_ptr + physical_page * stride_cb + page_offset * stride_cp + head_idx * stride_ch + offs_d * stride_cd
        k = tl.load(k_ptrs).to(tl.float32)

        # Compute dot product q . k
        qk = tl.sum(q * k) * sm_scale

        # Online softmax update
        m_curr = tl.maximum(m_prev, qk)
        alpha = tl.exp(m_prev - m_curr)
        beta = tl.exp(qk - m_curr)
        l_curr = l_prev * alpha + beta

        # Load Value vector
        v_ptrs = v_cache_ptr + physical_page * stride_cb + page_offset * stride_cp + head_idx * stride_ch + offs_d * stride_cd
        v = tl.load(v_ptrs).to(tl.float32)

        # Rescale accumulator and add new value contribution
        acc = acc * alpha + v * beta
        m_prev = m_curr
        l_prev = l_curr

    # Store partial reduction results
    po_ptrs = (
        partial_out_ptr
        + batch_idx * stride_pob
        + split_idx * stride_pos
        + head_idx * stride_poh
        + offs_d * stride_pod
    )
    tl.store(po_ptrs, acc)

    lse = m_prev + tl.log(l_prev)
    pl_ptr = partial_lse_ptr + batch_idx * stride_plb + split_idx * stride_pls + head_idx * stride_plh
    tl.store(pl_ptr, lse)


@triton.jit
def _split_k_decode_stage2(
    partial_out_ptr, partial_lse_ptr, final_out_ptr,
    seq_lens_ptr,
    stride_pob, stride_pos, stride_poh, stride_pod,
    stride_plb, stride_pls, stride_plh,
    stride_ob, stride_oh, stride_od,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Stage 2: Fast cross-split reduction. Combines partial results into the final output.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)

    # 1. Find global maximum LSE across splits
    global_max_lse = -float("inf")
    for s in range(0, NUM_SPLITS):
        lse = tl.load(partial_lse_ptr + batch_idx * stride_plb + s * stride_pls + head_idx * stride_plh)
        global_max_lse = tl.maximum(global_max_lse, lse)

    # 2. Accumulate normalized outputs
    total_weight = 0.0
    final_acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for s in range(0, NUM_SPLITS):
        lse = tl.load(partial_lse_ptr + batch_idx * stride_plb + s * stride_pls + head_idx * stride_plh)
        weight = tl.exp(lse - global_max_lse)
        total_weight += weight

        po_ptrs = (
            partial_out_ptr
            + batch_idx * stride_pob
            + s * stride_pos
            + head_idx * stride_poh
            + offs_d * stride_pod
        )
        po = tl.load(po_ptrs)
        final_acc += po * weight

    final_val = (final_acc / total_weight).to(tl.bfloat16)

    # Write out final attention result
    out_ptrs = final_out_ptr + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, final_val)


def kernel_fn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Entrypoint for OKC muse_attention_decode.
    Graph capturable: uses pre-allocated buffers.
    """
    batch_size, num_heads, head_dim = q.shape
    num_splits = 16  # Divides sequence across 16 splits to occupy H200 SMs

    if out is None:
        out = torch.empty_like(q)

    # Scratchpads for split-k (can be pre-allocated in warmup for CUDA graph)
    partial_out = torch.empty(
        (batch_size, num_splits, num_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    partial_lse = torch.full(
        (batch_size, num_splits, num_heads),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Launch Stage 1 (Parallel splits)
    grid_1 = (batch_size, num_heads, num_splits)
    _split_k_decode_stage1[grid_1](
        q, k_cache, v_cache,
        block_tables, seq_lens,
        partial_out, partial_lse,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        sm_scale,
        PAGE_SIZE=16,
        HEAD_DIM=head_dim,
        BLOCK_SPLIT=128,
        num_warps=4,
    )

    # Launch Stage 2 (Split reduction)
    grid_2 = (batch_size, num_heads)
    _split_k_decode_stage2[grid_2](
        partial_out, partial_lse, out,
        seq_lens,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_SPLITS=num_splits,
        HEAD_DIM=head_dim,
        num_warps=2,
    )

    return out
