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
    QueryStart,
    q_stride_n: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_h: tl.constexpr,
    bias_stride_n: tl.constexpr,
    bias_stride_h,
    bias_stride_seq: tl.constexpr,
    o_stride_n: tl.constexpr,
    o_stride_h: tl.constexpr,
    length,
    bias_length,
    VARLEN: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PADDED_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    if VARLEN:
        request = tl.program_id(2)
        begin = tl.load(QueryStart + request).to(tl.int64)
        length = tl.load(QueryStart + request + 1) - begin
        Q += begin * q_stride_n
        K += begin * k_stride_n
        V += begin * v_stride_n
        Output += begin * o_stride_n
        Bias += request.to(tl.int64) * bias_stride_seq
    if block * BLOCK_M >= length:
        return
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
    # Whole key tiles before the first query row need no causal or length mask.
    boundary = block * BLOCK_M // BLOCK_N * BLOCK_N
    for stage in tl.static_range(2):
        if stage == 0:
            begin, end = 0, boundary
        else:
            begin, end = boundary, tl.minimum((block + 1) * BLOCK_M, length)
        for start in range(begin, end, BLOCK_N):
            keys = start + columns
            k = tl.load(
                K
                + keys.to(tl.int64)[None, :] * k_stride_n
                + kv_head * k_stride_h
                + dimensions[:, None],
                ((stage == 0) | (keys[None, :] < length))
                & (dimensions[:, None] < HEAD_SIZE),
                0,
            )
            # One vector load per key tile; broadcast across the query rows.
            bias = tl.load(
                Bias + keys.to(tl.int64) * bias_stride_n + head * bias_stride_h,
                keys < bias_length,
                0,
            ).to(tl.float32)
            logits = tl.dot(q, k).to(tl.float32) * SCALE + bias[None, :]
            if stage == 1:
                logits = tl.where(
                    (rows[:, None] >= keys[None, :]) & (keys[None, :] < length),
                    logits,
                    -float("inf"),
                )
            logits *= 1.4426950408889634
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
                ((stage == 0) | (keys[:, None] < length))
                & (dimensions[None, :] < HEAD_SIZE),
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
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int | None = None,
    tile: tuple[int, int, int, int] | None = None,
) -> None:
    """Write full-sequence causal attention with TokenGate's prompt-key bias.

    Args:
        query: Scaled queries [tokens, query_heads, head_size], in BF16.
        key: Keys [tokens, kv_heads, head_size], in query's dtype.
        value: Values with key's shape and dtype.
        key_bias: Bias [requests, max_prompt_tokens, query_heads] for packed
            sequences, or [prompt_tokens, query_heads] for a single sequence.
            Later keys receive zero bias. Arbitrary strides are supported.
        output: Result buffer with query's shape and dtype.
        scale: QK softmax scale; bias is added after scaling.
        query_start_loc: Cumulative request boundaries [requests + 1] on the
            query device; None means one contiguous sequence.
        max_query_len: Longest request in the packed inputs; required with
            query_start_loc. Taken from CPU scheduling metadata without GPU reads.
        tile: Optional (query_rows, key_rows, warps, stages) for kernel tuning.
            Ordinary calls use a shape-based launch policy.

    Head elements must be contiguous. The input tensors contain the complete
    retained sequence, with query and KV heads grouped in the usual GQA order.
    The caller still writes K/V to vLLM's cache for subsequent decoding.

    """
    length, heads, size = query.shape
    packed = query_start_loc is not None
    if query_start_loc is not None:
        if max_query_len is None or max_query_len < 1:
            raise ValueError("Packed prefill requires a positive max_query_len")
        requests = query_start_loc.numel() - 1
        if key_bias.ndim != 3 or key_bias.shape[0] != requests:
            raise ValueError("Packed key_bias must have one row per request")
        bias_stride_seq = key_bias.stride(0)
    else:
        requests, max_query_len, bias_stride_seq = 1, length, 0
        if key_bias.ndim != 2:
            raise ValueError("Single-sequence key_bias must be two-dimensional")
    if tile is None:
        # One pipeline stage leaves room for wider key tiles with large heads.
        tile = (
            (64, 64, 4, 1)
            if size > 128
            else ((128, 64, 4, 2) if max_query_len >= 16384 else (64, 64, 4, 2))
        )
    block_m, block_n, warps, stages = tile
    _token_gate_prefill[(triton.cdiv(max_query_len, block_m), heads, requests)](
        query,
        key,
        value,
        key_bias,
        output,
        query_start_loc,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_bias.stride(-2),
        key_bias.stride(-1),
        bias_stride_seq,
        output.stride(0),
        output.stride(1),
        length,
        key_bias.shape[-2],
        packed,
        heads // key.shape[1],
        scale,
        size,
        triton.next_power_of_2(size),
        block_m,
        block_n,
        num_warps=warps,
        num_stages=stages,
    )
