# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the extra GPU time needed for TokenGate's attention-score adjustments.

TokenGate changes attention weights as well as removing tokens. This benchmark
times the Triton attention kernel with and without the added prompt-key bias,
using identical sequence lengths. It excludes the work of computing that bias,
so the result measures this kernel's overhead rather than the speedup from
removing tokens or the latency of answering a request.
"""

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@torch.inference_mode()
def measure(
    query_length: int, sequence_length: int, heads: int, kv_heads: int, head_size: int
) -> dict:
    device, dtype, block_size = "cuda", torch.bfloat16, 16
    blocks = (sequence_length + block_size - 1) // block_size
    query = torch.randn(query_length, heads, head_size, device=device, dtype=dtype)
    key = torch.randn(
        blocks, block_size, kv_heads, head_size, device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    # An offset view exercises the pointer alignment seen with request buffers.
    bias = torch.randn(sequence_length * heads + 1, device=device, dtype=dtype)[1:]
    bias = bias.view(1, sequence_length, heads)
    kwargs = dict(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=torch.tensor([0, query_length], device=device, dtype=torch.int32),
        max_seqlen_q=query_length,
        seqused_k=torch.tensor([sequence_length], device=device, dtype=torch.int32),
        max_seqlen_k=sequence_length,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.arange(blocks, device=device, dtype=torch.int32)[None],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=8,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(
            query_length, heads, 16, head_size, device=device
        ),
        softmax_segm_max=torch.empty(query_length, heads, 16, device=device),
        softmax_segm_expsum=torch.empty(query_length, heads, 16, device=device),
    )
    rows = torch.tensor(sorted({0, query_length // 2, query_length - 1}), device=device)
    dense_key = key.flatten(0, 1)[:sequence_length].repeat_interleave(
        heads // kv_heads, 1
    )
    dense_value = value.flatten(0, 1)[:sequence_length].repeat_interleave(
        heads // kv_heads, 1
    )
    scores = torch.einsum("qhd,khd->hqk", query[rows].float(), dense_key.float())
    scores *= head_size**-0.5
    causal = torch.arange(sequence_length, device=device)[None] <= (
        sequence_length - query_length + rows[:, None]
    )
    result = dict(
        query_length=query_length,
        sequence_length=sequence_length,
        heads=heads,
        kv_heads=kv_heads,
        head_size=head_size,
    )
    for label, key_bias in (("ordinary", None), ("prompt_bias", bias)):

        def run(key_bias=key_bias):
            unified_attention(**kwargs, key_bias=key_bias)

        torch.accelerator.synchronize()
        start = time.perf_counter()
        run()
        torch.accelerator.synchronize()
        result[label + "_first_call_seconds"] = time.perf_counter() - start
        expected_scores = scores.clone()
        if key_bias is not None:
            expected_scores += bias[0].T[:, None].float()
        expected_scores.masked_fill_(~causal[None], -float("inf"))
        probabilities = expected_scores.softmax(-1).to(dtype)
        expected = torch.einsum("hqk,khd->qhd", probabilities, dense_value)
        torch.testing.assert_close(output[rows], expected, atol=1.5e-2, rtol=1.5e-2)
        for _ in range(25):
            run()
        milliseconds = statistics.median(
            bench_gpu_time_with_cupti(run, use_cuda_graph=True, cold_l2_cache=True)
        )
        result[label + "_microseconds"] = milliseconds * 1000
        # Two matrix products over the causal query/key pairs; excludes softmax.
        pairs = query_length * sequence_length - query_length * (query_length - 1) / 2
        result[label + "_tflops"] = 4 * heads * head_size * pairs / (milliseconds * 1e9)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Never label a silent CUDA-event fallback as a CUPTI measurement.
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
    torch.manual_seed(19)
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        dtype="bfloat16",
        timing="CUPTI, CUDA graph, cold L2; production launch heuristic",
        cases=[],
    )
    for heads, kv_heads, size in ((8, 2, 64), (16, 8, 128)):
        for sequence_length in (2048, 16384):
            for query_length in (1, 128, 2048):
                row = measure(query_length, sequence_length, heads, kv_heads, size)
                result["cases"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
