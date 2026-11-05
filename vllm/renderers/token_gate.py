# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a text prompt with a span that TokenGate may prune.

Tokenize the complete prompt once and map the caller's character boundaries to
token indices. The helper returns token IDs and SamplingParams.extra_args for
TokenGate, preserving tokens outside the span and across its boundaries.
"""

from typing import Any

from vllm.utils.token_gate import TOKEN_GATE_FIELDS, TokenGateRequestOptions


def tokenize_span(
    tokenizer: Any, text: str, span_start: int, span_end: int, keep_tokens: int
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Tokenize a complete prompt and mark the text range eligible for pruning.

    Tokens that cross either character boundary are kept outside the span.
    At least one token must follow the span, preserving the final prompt token.

    Args:
        tokenizer: Fast tokenizer that returns offsets into the original text.
        text: Complete prompt, including any text outside the selected span.
        span_start: Inclusive character offset of the selected text range.
        span_end: Exclusive character offset of the selected text range.
        keep_tokens: Positive maximum number of tokens to keep within the span.

    Returns:
        A TokensPrompt and SamplingParams.extra_args containing the span's
        token offsets and retention budget.

    """
    if not 0 <= span_start < span_end <= len(text):
        raise ValueError("Invalid character span")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    span_rows = [
        index
        for index, (start, end) in enumerate(offsets)
        if start >= span_start and end <= span_end and start < end
    ]
    if not span_rows or span_rows != list(range(span_rows[0], span_rows[-1] + 1)):
        raise ValueError(
            "The character span must map to a nonempty contiguous token span"
        )
    extra_args = dict(
        zip(
            TOKEN_GATE_FIELDS,
            (
                1,
                span_rows[0],
                span_rows[-1] + 1,
                keep_tokens,
            ),
        )
    )
    TokenGateRequestOptions.parse(extra_args, len(encoded["input_ids"]))
    return {"prompt_token_ids": encoded["input_ids"]}, extra_args
