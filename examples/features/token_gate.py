# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demonstrate document question answering with TokenGate in vLLM.

TokenGate is a learned filter inside a language model. It selects document
tokens for later model layers to process, aiming to reduce computation and
attention-cache use. This example reads a document and question, loads a Qwen3
or text-only Qwen3.5 model prepared by tools/export_token_gate.py, and prints
an answer. --keep-tokens sets how many document tokens survive the filter;
the question and surrounding prompt text are always kept. Omit that option
when loading a model exported without TokenGate.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.renderers.token_gate import tokenize_span


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--keep-tokens", type=int)
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prefix = f"Question: {args.question}\nDocument:\n"
    document = args.document.read_text(encoding="utf-8")
    text = prefix + document + "\nAnswer:"
    extra_args = None
    if args.keep_tokens is None:
        prompt = {"prompt_token_ids": tokenizer.encode(text, add_special_tokens=False)}
    else:
        prompt, extra_args = tokenize_span(
            tokenizer, text, len(prefix), len(prefix) + len(document), args.keep_tokens
        )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_scheduled_tokens=args.max_model_len,
        long_prefill_token_threshold=0,
        max_num_seqs=1,
        async_scheduling=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(temperature=0, max_tokens=128, extra_args=extra_args)
    print(llm.generate(prompt, sampling)[0].outputs[0].text)


if __name__ == "__main__":
    main()
