# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size each attention cache for the number of tokens its layers actually keep.

A key/value (KV) cache stores information about earlier tokens for use when
generating the next token. With TokenGate, layers before filtering cache the
full prompt, while later layers cache only the retained prompt. These helpers translate
request lengths into the counts needed to allocate blocks for each group of
layers, including tokens generated after the prompt.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheSequenceLayout:
    original_prompt_length: int
    retained_prompt_length: int

    def __post_init__(self) -> None:
        if not 0 < self.retained_prompt_length <= self.original_prompt_length:
            raise ValueError("Invalid retained prompt length")

    def project(self, endpoint: int) -> int:
        if endpoint == 0:
            return 0
        if endpoint < self.original_prompt_length:
            raise ValueError("This cache layout requires complete prompt execution")
        return endpoint - self.original_prompt_length + self.retained_prompt_length


@dataclass(frozen=True)
class KVCacheTokenCounts:
    num_tokens: int
    num_tokens_main_model: int
    total_computed_tokens: int = 0
    num_local_computed_tokens: int = 0
    prefill_end: int = 0

    def project(self, layout: CacheSequenceLayout) -> "KVCacheTokenCounts":
        return KVCacheTokenCounts(
            *(
                layout.project(value)
                for value in (
                    self.num_tokens,
                    self.num_tokens_main_model,
                    self.total_computed_tokens,
                    self.num_local_computed_tokens,
                    self.prefill_end,
                )
            )
        )


def compute_group_token_counts(
    counts: KVCacheTokenCounts,
    layout: CacheSequenceLayout | None,
    compact_groups: tuple[bool, ...],
) -> tuple[KVCacheTokenCounts, ...]:
    if any(compact_groups) and layout is None:
        raise ValueError("Compact cache groups require a request sequence layout")
    compact = counts.project(layout) if layout is not None else counts
    return tuple(compact if uses_compact else counts for uses_compact in compact_groups)
