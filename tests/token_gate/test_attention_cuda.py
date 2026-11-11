# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check TokenGate attention during full prefill and paged-cache generation.

The learned token scores add a bias to attention for the prompt tokens that
remain after filtering. The tiled and paged Triton kernels are compared with
a direct dense calculation during prompt processing and generation, including
keys already in the cache. They also check that a zero bias leaves ordinary
attention outputs unchanged.
"""

import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@pytest.mark.parametrize("size,kv_heads", [(64, 2), (96, 1), (256, 4)])
@pytest.mark.parametrize("lengths", [(1, 17, 129), (65, 513, 3)])
def test_packed_prefill_isolates_unequal_requests(size, kv_heads, lengths):
    from vllm.v1.attention.ops.token_gate_attention import token_gate_prefill

    torch.manual_seed(19)
    heads = 4
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    qkv = torch.randn(
        offsets[-1],
        heads + 2 * kv_heads,
        size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    query, key, value = qkv.split([heads, kv_heads, kv_heads], dim=1)
    bias = torch.zeros(
        len(lengths), heads, max(lengths), device="cuda", dtype=query.dtype
    ).transpose(1, 2)
    for i, length in enumerate(lengths):
        bias[i, : max(0, length - 3)].normal_()
    output = torch.empty_like(query)
    starts = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    token_gate_prefill(
        query,
        key,
        value,
        bias,
        output,
        size**-0.5,
        query_start_loc=starts,
        max_query_len=max(lengths),
    )
    for i, length in enumerate(lengths):
        start, end = offsets[i : i + 2]
        dense_key = key[start:end].repeat_interleave(heads // kv_heads, dim=1)
        dense_value = value[start:end].repeat_interleave(heads // kv_heads, dim=1)
        scores = (
            torch.einsum("qhd,khd->hqk", query[start:end].float(), dense_key.float())
            / size**0.5
        )
        scores += bias[i, :length].T[:, None].float()
        causal = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
        scores.masked_fill_(~causal[None], float("-inf"))
        expected = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1).to(query.dtype), dense_value
        )
        torch.testing.assert_close(
            output[start:end], expected, atol=1.5e-2, rtol=1.5e-2
        )
    # Changing a neighboring request must not affect the first request at all.
    unchanged = output[: lengths[0]].clone()
    qkv[lengths[0] :].mul_(7)
    bias[1:].add_(3)
    token_gate_prefill(
        query,
        key,
        value,
        bias,
        output,
        size**-0.5,
        query_start_loc=starts,
        max_query_len=max(lengths),
    )
    torch.testing.assert_close(output[: lengths[0]], unchanged, atol=0, rtol=0)


@pytest.mark.parametrize(
    "heads,kv_heads,size",
    [(4, 2, 64), (4, 4, 64), (4, 1, 96), (16, 8, 128), (4, 1, 192), (8, 2, 256)],
)
@pytest.mark.parametrize("length", [1, 17, 64, 65, 129, 513])
@pytest.mark.parametrize("bias_scale", [0.0, 1.0, 8.0])
@pytest.mark.parametrize("tile", [None, (128, 32, 4, 1), (32, 64, 4, 2)])
def test_tiled_prefill_matches_dense_reference(
    heads, kv_heads, size, length, bias_scale, tile
):
    from vllm.v1.attention.ops.token_gate_attention import token_gate_prefill

    torch.manual_seed(19)
    device, dtype = "cuda", torch.bfloat16
    # Interleaved views reproduce QKV projection strides, without copying K/V.
    qkv = torch.randn(length, heads + 2 * kv_heads, size, device=device, dtype=dtype)
    query, key, value = qkv.split([heads, kv_heads, kv_heads], dim=1)
    prefix = max(0, length - 3)
    bias = torch.randn(heads, prefix + 1, device=device, dtype=dtype)[:, 1:].T
    bias.mul_(bias_scale)
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    token_gate_prefill(query, key, value, bias, output, size**-0.5, tile=tile)
    dense_key = key.repeat_interleave(heads // kv_heads, dim=1)
    dense_value = value.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), dense_key.float()) / size**0.5
    scores[:, :, :prefix] += bias.T[:, None].float()
    causal = (
        torch.arange(length, device=device)[:, None]
        >= torch.arange(length, device=device)[None, :]
    )
    scores.masked_fill_(~causal[None], float("-inf"))
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1).to(dtype), dense_value)
    torch.testing.assert_close(output, expected, atol=1.5e-2, rtol=1.5e-2)


@pytest.mark.parametrize("size", [64, 256])
@pytest.mark.parametrize("query_length", [1, 5, 17])
@pytest.mark.parametrize("split_threshold", [0, 8])
@pytest.mark.parametrize("prefix_length", [7, 37])
def test_paged_prompt_bias_covers_cached_keys(
    query_length, split_threshold, prefix_length, size
):
    torch.manual_seed(19)
    device, dtype = "cuda", torch.bfloat16
    heads, kv_heads, block_size, length = 4, 2, 16, 37
    query = torch.randn(query_length, heads, size, device=device, dtype=dtype)
    key = torch.randn(5, block_size, kv_heads, size, device=device, dtype=dtype)
    value = torch.randn_like(key)
    blocks = torch.tensor([[3, 1, 4]], device=device, dtype=torch.int32)
    bias = torch.randn(1, prefix_length, heads, device=device, dtype=dtype)
    output = torch.empty_like(query)
    scratch = [
        torch.empty(query_length, heads, 16, size, device=device),
        torch.empty(query_length, heads, 16, device=device),
        torch.empty(query_length, heads, 16, device=device),
    ]
    kwargs = dict(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=torch.tensor([0, query_length], device=device, dtype=torch.int32),
        max_seqlen_q=query_length,
        seqused_k=torch.tensor([length], device=device, dtype=torch.int32),
        max_seqlen_k=length,
        softmax_scale=size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=blocks,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=split_threshold,
        num_par_softmax_segments=16,
        softmax_segm_output=scratch[0],
        softmax_segm_max=scratch[1],
        softmax_segm_expsum=scratch[2],
    )
    unified_attention(**kwargs, key_bias=bias)
    dense_key = key[blocks[0].long()].flatten(0, 1)[:length].repeat_interleave(2, dim=1)
    dense_value = (
        value[blocks[0].long()].flatten(0, 1)[:length].repeat_interleave(2, dim=1)
    )
    scores = torch.einsum("qhd,khd->hqk", query.float(), dense_key.float()) / size**0.5
    scores[:, :, :prefix_length] += bias[0].T[:, None].float()
    causal = (
        torch.arange(length, device=device)[None, :]
        <= torch.arange(length - query_length, length, device=device)[:, None]
    )
    scores.masked_fill_(~causal[None], float("-inf"))
    probabilities = scores.softmax(-1).to(dtype)
    expected = torch.einsum("hqk,khd->qhd", probabilities, dense_value)
    torch.testing.assert_close(output, expected, atol=1.5e-2, rtol=1.5e-2)
    # The optional branch must preserve ordinary attention when the bias is zero.
    unified_attention(**kwargs)
    ordinary = output.clone()
    unified_attention(**kwargs, key_bias=torch.zeros_like(bias))
    torch.testing.assert_close(output, ordinary, atol=0, rtol=0)
