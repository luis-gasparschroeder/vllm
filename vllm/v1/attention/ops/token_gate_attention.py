# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute TokenGate prefill attention from retained Q, K, and V tensors.

The kernel reuses each key's learned bias across a tile of queries and performs
causal softmax without materializing an attention matrix. Pruning, position
correction, and query scaling remain the model's responsibility. Subsequent
decoding reads vLLM's paged cache through the existing attention backend.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["length", "bias_length"])
def _token_gate_prefill(
    Q,
    K,
    V,
    Bias,
    Output,
    q_stride_n: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_h: tl.constexpr,
    bias_stride_n: tl.constexpr,
    bias_stride_h,
    o_stride_n: tl.constexpr,
    o_stride_h: tl.constexpr,
    length,
    bias_length,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PADDED_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    kv_head = head // GROUP_SIZE
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    dimensions = tl.arange(0, PADDED_SIZE)
    q = tl.load(
        Q
        + rows.to(tl.int64)[:, None] * q_stride_n
        + head * q_stride_h
        + dimensions[None, :],
        (rows[:, None] < length) & (dimensions[None, :] < HEAD_SIZE),
        0,
    )
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.full((BLOCK_M, PADDED_SIZE), 0.0, tl.float32)
    for start in range(0, tl.minimum((block + 1) * BLOCK_M, length), BLOCK_N):
        keys = start + columns
        k = tl.load(
            K
            + keys.to(tl.int64)[None, :] * k_stride_n
            + kv_head * k_stride_h
            + dimensions[:, None],
            (keys[None, :] < length) & (dimensions[:, None] < HEAD_SIZE),
            0,
        )
        # One vector load per key tile; broadcast across the query rows.
        bias = tl.load(
            Bias + keys.to(tl.int64) * bias_stride_n + head * bias_stride_h,
            keys < bias_length,
            0,
        ).to(tl.float32)
        logits = tl.dot(q, k).to(tl.float32) * SCALE + bias[None, :]
        logits = tl.where(
            (rows[:, None] >= keys[None, :]) & (keys[None, :] < length),
            logits * 1.4426950408889634,
            -float("inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        probabilities = tl.exp2(logits - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        normalizer = normalizer * correction + tl.sum(probabilities, 1)
        accumulator = accumulator * correction[:, None]
        v = tl.load(
            V
            + keys.to(tl.int64)[:, None] * v_stride_n
            + kv_head * v_stride_h
            + dimensions[None, :],
            (keys[:, None] < length) & (dimensions[None, :] < HEAD_SIZE),
            0,
        )
        accumulator = tl.dot(probabilities.to(v.dtype), v, accumulator)
        maximum = next_maximum
    result = accumulator / normalizer[:, None]
    tl.store(
        Output
        + rows.to(tl.int64)[:, None] * o_stride_n
        + head * o_stride_h
        + dimensions[None, :],
        result.to(Output.dtype.element_ty),
        (rows[:, None] < length) & (dimensions[None, :] < HEAD_SIZE),
    )


def token_gate_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_bias: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    tile: tuple[int, int, int, int] | None = None,
) -> None:
    """Write full-sequence causal attention with TokenGate's prompt-key bias.

    Args:
        query: Scaled queries [tokens, query_heads, head_size], in BF16.
        key: Keys [tokens, kv_heads, head_size], in query's dtype.
        value: Values with key's shape and dtype.
        key_bias: Bias [prompt_tokens, query_heads], in query's dtype.
            Later keys receive zero bias. Arbitrary strides are supported.
        output: Result buffer with query's shape and dtype.
        scale: QK softmax scale; bias is added after scaling.
        tile: Optional (query_rows, key_rows, warps, stages) for kernel tuning.
            Ordinary calls use a shape-based launch policy.

    Head elements must be contiguous. The input tensors contain the complete
    retained sequence, with query and KV heads grouped in the usual GQA order.
    The caller still writes K/V to vLLM's cache for subsequent decoding.

    """
    length, heads, size = query.shape
    if tile is None:
        tile = (
            (64, 32, 4, 2)
            if size > 128
            else ((128, 64, 4, 2) if length >= 16384 else (64, 64, 4, 2))
        )
    block_m, block_n, warps, stages = tile
    _token_gate_prefill[(triton.cdiv(length, block_m), heads)](
        query,
        key,
        value,
        key_bias,
        output,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_bias.stride(0),
        key_bias.stride(1),
        output.stride(0),
        output.stride(1),
        length,
        key_bias.shape[0],
        heads // key.shape[1],
        scale,
        size,
        triton.next_power_of_2(size),
        block_m,
        block_n,
        num_warps=warps,
        num_stages=stages,
    )
