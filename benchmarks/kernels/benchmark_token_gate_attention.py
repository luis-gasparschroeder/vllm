# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FlashAttention, paged and tiled TokenGate attention on matched tensors.

Time ordinary paged attention, biased paged attention, and the dedicated tiled
prefill kernel. Full prefill also measures vLLM's ordinary FlashAttention.
Check a dense reference before using CUPTI, CUDA graphs, and
cold L2 timing. Query scaling, bias preparation, and KV writes are excluded;
end-to-end benchmarks must include them. Optional tiles support controlled tuning.
"""

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.triton_utils import triton
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.ops.token_gate_attention import token_gate_prefill
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.vllm_flash_attn import flash_attn_varlen_func


@torch.inference_mode()
def measure(
    query_length: int,
    sequence_length: int,
    heads: int,
    kv_heads: int,
    head_size: int,
    tiles: list | None = None,
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
            query_length, heads, 16, triton.next_power_of_2(head_size), device=device
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
    variants = [("ordinary", None, None), ("prompt_bias", bias, None)]
    if query_length == sequence_length:
        direct_key = key.flatten(0, 1)[:sequence_length]
        direct_value = value.flatten(0, 1)[:sequence_length]
        packed_bias = bias[0].T.contiguous().T
        result["flash_attention_version"] = get_flash_attn_version(head_size=head_size)
        variants.append(("flash_attention", None, None))
        variants.append(("tiled_bias", packed_bias, None))
        for tile in tiles or []:
            label = "tiled_" + "_".join(map(str, tile))
            variants.append((label, packed_bias, tuple(tile)))
    for label, key_bias, tile in variants:

        def run(key_bias=key_bias, label=label, tile=tile):
            if label.startswith("tiled"):
                token_gate_prefill(
                    query,
                    direct_key,
                    direct_value,
                    key_bias,
                    output,
                    head_size**-0.5,
                    tile=tile,
                )
            elif label == "flash_attention":
                flash_attn_varlen_func(
                    query,
                    direct_key,
                    direct_value,
                    query_length,
                    kwargs["cu_seqlens_q"],
                    sequence_length,
                    kwargs["cu_seqlens_q"],
                    softmax_scale=head_size**-0.5,
                    causal=True,
                    fa_version=result["flash_attention_version"],
                    out=output,
                )
            else:
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
        timings = sorted(
            bench_gpu_time_with_cupti(
                run, repeat_iters=15, use_cuda_graph=True, cold_l2_cache=True
            )
        )
        milliseconds = statistics.median(timings)
        result[label + "_p10_microseconds"] = timings[len(timings) // 10] * 1000
        result[label + "_p90_microseconds"] = (
            timings[min(len(timings) - 1, len(timings) * 9 // 10)] * 1000
        )
        result[label + "_microseconds"] = milliseconds * 1000
        # Two matrix products over the causal query/key pairs; excludes softmax.
        pairs = query_length * sequence_length - query_length * (query_length - 1) / 2
        result[label + "_tflops"] = 4 * heads * head_size * pairs / (milliseconds * 1e9)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        action="append",
        metavar=("HEADS", "KV_HEADS", "HEAD_SIZE"),
        help="Repeat to benchmark multiple model attention shapes",
    )
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[2048, 16384]
    )
    parser.add_argument("--query-lengths", type=int, nargs="+", default=[1, 128, 2048])
    parser.add_argument(
        "--full-prefill",
        action="store_true",
        help="Also measure query length equal to each context length",
    )
    parser.add_argument(
        "--tile",
        type=int,
        nargs=4,
        action="append",
        metavar=("QUERY_ROWS", "KEY_ROWS", "WARPS", "STAGES"),
        help="Additional tiled-prefill launch configuration to measure",
    )
    args = parser.parse_args()
    shapes = args.shape or [(8, 2, 64), (16, 8, 128)]
    if any(min(shape) < 1 or shape[0] % shape[1] for shape in shapes):
        parser.error(
            "head counts and dimensions must be positive; "
            "heads must be divisible by KV heads"
        )
    if min(args.sequence_lengths + args.query_lengths) < 1:
        parser.error("sequence and query lengths must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(
            "choose a new output path; existing measurements are not overwritten"
        )
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
    for heads, kv_heads, size in shapes:
        for sequence_length in args.sequence_lengths:
            queries = set(args.query_lengths)
            if args.full_prefill:
                queries.add(sequence_length)
            for query_length in sorted(queries):
                if query_length > sequence_length:
                    continue
                row = measure(
                    query_length, sequence_length, heads, kv_heads, size, args.tile
                )
                result["cases"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
