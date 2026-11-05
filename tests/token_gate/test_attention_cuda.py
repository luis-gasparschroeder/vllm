# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that TokenGate's attention adjustments work with a paged GPU cache.

The learned token scores add a bias to attention for the prompt tokens that
remain after filtering. These tests compare the Triton kernel with a direct
dense calculation during prompt processing and next-token generation, including
keys already in the cache. They also check that a zero bias leaves ordinary
attention outputs unchanged.
"""

import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


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
