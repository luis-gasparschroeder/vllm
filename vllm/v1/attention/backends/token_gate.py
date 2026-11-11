# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run attention with the learned adjustments required by a TokenGate model.

TokenGate both removes prompt tokens and changes how later attention layers
weight the remaining tokens. This backend scales attention queries and adds a
score-based bias for retained prompt keys. Full prefill can use a dedicated
tiled kernel; decode and replay read the paged KV cache.
The adjustments apply during prompt processing and answer generation; newly
generated keys receive no score-based bias.
"""

from dataclasses import dataclass, field, replace

import torch

from vllm.config import get_current_vllm_config
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
    prefill_query_start_loc: torch.Tensor | None = field(init=False, default=None)
    paged_metadata: TritonAttentionMetadata | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Prepare views for a cached prefix and an uncached prefill suffix.

        Request and token order are unchanged. Preparing offsets here avoids
        rebuilding them in every attention-layer forward pass.
        """
        if self.batch is None or self.batch.uncached_prefill_start is None:
            return
        first = self.batch.uncached_prefill_start
        tokens = self.batch.output_offsets[first]
        self.prefill_query_start_loc = self.query_start_loc[first:] - tokens
        if first:
            requests = self.batch.requests[:first]
            self.paged_metadata = TritonAttentionMetadata(
                num_actual_tokens=tokens,
                max_query_len=max(r.num_retained_tokens for r in requests),
                query_start_loc=self.query_start_loc[: first + 1],
                max_seq_len=max(
                    r.compact_computed_tokens + r.num_retained_tokens for r in requests
                ),
                seq_lens=self.seq_lens[:first],
                block_table=self.block_table[:first],
                slot_mapping=self.slot_mapping[:tokens],
                seq_threshold_3D=self.seq_threshold_3D,
                num_par_softmax_segments=self.num_par_softmax_segments,
                softmax_segm_output=self.softmax_segm_output,
                softmax_segm_max=self.softmax_segm_max,
                softmax_segm_expsum=self.softmax_segm_expsum,
                causal=self.causal,
            )


class TokenGateAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TOKEN_GATE"

    @staticmethod
    def get_impl_cls() -> type["TokenGateAttentionImpl"]:
        return TokenGateAttentionImpl


class TokenGateAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config().attention_config
        self.use_tiled_prefill = (
            config.token_gate_prefill_backend == "tiled"
            and self.head_size <= 256
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.chunk_lookback == -1
        )

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
        if (
            self.use_tiled_prefill
            and attn_metadata.causal is True
            and attn_metadata.mm_prefix_range_tensor is None
            and output_scale is None
            and output_block_scale is None
            and attn_metadata.prefill_query_start_loc is not None
        ):
            from vllm.v1.attention.ops.token_gate_attention import token_gate_prefill

            # Adjacent keys within a head share cache lines. No values change.
            first = attn_metadata.batch.uncached_prefill_start
            assert first is not None
            tokens = attn_metadata.batch.output_offsets[first]
            prefill_bias = bias[first:].transpose(1, 2).contiguous().transpose(1, 2)
            token_gate_prefill(
                query[tokens:],
                key[tokens:],
                value[tokens:],
                prefill_bias,
                output[tokens:],
                self.scale,
                query_start_loc=attn_metadata.prefill_query_start_loc,
                max_query_len=attn_metadata.max_query_len,
            )
            if attn_metadata.paged_metadata is not None:
                super().forward(
                    layer,
                    query[:tokens],
                    key[:tokens],
                    value[:tokens],
                    kv_cache,
                    replace(attn_metadata.paged_metadata, key_bias=bias[:first]),
                    output[:tokens],
                )
            return output
        metadata = replace(attn_metadata, key_bias=bias)
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
