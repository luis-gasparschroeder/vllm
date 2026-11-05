# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate each request's TokenGate pruning span and retained-token budget.

TokenGate may shorten one contiguous range of prompt tokens while preserving
all tokens outside it. The request supplies that range and its budget through
SamplingParams.extra_args. These shared definitions let API entrypoints, the
engine, and GPU workers check the same fields and reject incompatible options.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TOKEN_GATE_FIELDS = (
    "token_gate_version",
    "token_gate_span_start",
    "token_gate_span_end",
    "token_gate_keep_tokens",
)


def validate_token_gate_fields(extra_args: Mapping[str, Any] | None) -> None:
    """Validate raw types before an API schema can coerce booleans."""
    if not extra_args or not any(key in extra_args for key in TOKEN_GATE_FIELDS):
        return
    for key in TOKEN_GATE_FIELDS:
        if key not in extra_args or type(extra_args[key]) is not int:
            raise ValueError(f"{key} must be an integer, excluding booleans")
    if extra_args["token_gate_version"] != 1:
        raise ValueError("Unsupported token_gate_version")


@dataclass(frozen=True)
class TokenGateRequestOptions:
    """A token range [span_start, span_end) and the number of its tokens to keep."""

    span_start: int
    span_end: int
    keep_tokens: int

    @classmethod
    def parse(
        cls, extra_args: Mapping[str, Any] | None, prompt_length: int
    ) -> "TokenGateRequestOptions":
        validate_token_gate_fields(extra_args)
        if not extra_args or not all(key in extra_args for key in TOKEN_GATE_FIELDS):
            raise ValueError("TokenGate requires span boundaries and a budget")
        start = extra_args["token_gate_span_start"]
        end = extra_args["token_gate_span_end"]
        keep = extra_args["token_gate_keep_tokens"]
        if not 0 <= start < end < prompt_length:
            raise ValueError("TokenGate span must exclude the final prompt token")
        if keep < 1:
            raise ValueError("token_gate_keep_tokens must be positive")
        return cls(start, end, min(keep, end - start))

    @property
    def removed_tokens(self) -> int:
        return self.span_end - self.span_start - self.keep_tokens


def validate_token_gate_transformations(
    extra_args: Mapping[str, Any] | None,
    *,
    truncate_prompt_tokens: int | None = None,
    tokenization_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Reject transformations after the caller has aligned span offsets."""
    validate_token_gate_fields(extra_args)
    if not extra_args or not any(key in extra_args for key in TOKEN_GATE_FIELDS):
        return
    kwargs = tokenization_kwargs or {}
    if (
        truncate_prompt_tokens is not None
        or kwargs.get("truncation") not in (None, False, "do_not_truncate")
        or kwargs.get("padding") not in (None, False, "do_not_pad")
        or kwargs.get("truncate_prompt_tokens") is not None
        or kwargs.get("pad_prompt_tokens") is not None
        or kwargs.get("add_special_tokens") not in (None, False)
    ):
        raise ValueError("TokenGate requires unchanged, pretokenized completion inputs")


def validate_token_gate_sampling(params: Any) -> None:
    if params.n != 1:
        raise ValueError("TokenGate supports one completion per request")
    if (
        params.prompt_logprobs is not None
        or params.prompt_logprob_token_ids is not None
    ):
        raise ValueError("TokenGate does not support prompt scoring")
    validate_token_gate_fields(params.extra_args)
