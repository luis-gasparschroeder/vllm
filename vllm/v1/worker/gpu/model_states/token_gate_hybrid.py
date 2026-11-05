# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Support TokenGate filtering in Qwen3.5's mixture of attention and recurrent layers.

Attention layers cache earlier tokens; recurrent Gated DeltaNet layers instead
carry a fixed-size state forward. After TokenGate removes prompt tokens, both
kinds of layer must process the same shorter sequence. This module combines
TokenGate's sequence bookkeeping with vLLM's existing recurrent-state handling
through prompt processing, answer generation, and request cleanup.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.token_gate import TokenGateHistory
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class TokenGateHybridModelState(MambaHybridModelState):
    """TokenGate sequence views with native hybrid-model lifecycle handling."""

    supports_prompt_embeds = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.history = TokenGateHistory(vllm_config, model, encoder_cache, device)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.history.add_request(req_index, new_req_data)

    def remove_request(self, req_id: str) -> None:
        self.history.remove_request(req_id)
        super().remove_request(req_id)

    def prepare_batch(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        *,
        synthetic: bool = False,
        skip_attention: bool = False,
    ) -> None:
        self.history.prepare_batch(
            input_batch, req_states, synthetic=synthetic, skip_attention=skip_attention
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        return self.history.prepare_inputs(input_batch, req_states)

    def prepare_slot_mappings(
        self,
        input_batch: InputBatch,
        tables: BlockTables,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
    ) -> torch.Tensor:
        return self.history.prepare_slot_mappings(
            input_batch, tables, block_tables, kv_cache_config
        )

    def slot_mappings_by_layer(
        self, slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        return self.history.slot_mappings_by_layer(slot_mappings, kv_cache_config)

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        return self.history.prepare_attn(
            super().prepare_attn,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
