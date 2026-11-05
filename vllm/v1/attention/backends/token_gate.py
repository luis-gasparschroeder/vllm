# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run attention with the learned adjustments required by a TokenGate model.

TokenGate both removes prompt tokens and changes how later attention layers
weight the remaining tokens. This backend scales attention queries and adds a
score-based bias for retained prompt keys before calling Triton paged attention.
The adjustments apply during prompt processing and answer generation; newly
generated keys receive no score-based bias.
"""

from dataclasses import dataclass, replace

import torch

from vllm.model_executor.layers.token_gate import TokenGate, TokenGateBatch
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)


@dataclass
class TokenGateAttentionMetadata(TritonAttentionMetadata):
    batch: TokenGateBatch | None = None
    gate: TokenGate | None = None
    upper_layer_index: int = 0


class TokenGateAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TOKEN_GATE"

    @staticmethod
    def get_impl_cls() -> type["TokenGateAttentionImpl"]:
        return TokenGateAttentionImpl


class TokenGateAttentionImpl(TritonAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)
        if not isinstance(attn_metadata, TokenGateAttentionMetadata):
            raise ValueError("Missing TokenGate attention metadata")
        if attn_metadata.batch is None or attn_metadata.gate is None:
            raise ValueError("Missing TokenGate attention state")
        query, bias = attn_metadata.batch.attention_inputs(
            query, attn_metadata.gate, attn_metadata.upper_layer_index
        )
        metadata = replace(attn_metadata, key_bias=bias.unsqueeze(0))
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            output,
            output_scale,
            output_block_scale,
        )
