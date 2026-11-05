# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the GPU runner's request state consistent after TokenGate removes tokens.

Earlier model layers still use the full prompt, while later layers use a shorter
sequence. This module stores the selected tokens' scores and position ratios
and prepares the cache locations and attention inputs for both sequences. It
keeps that information across generated tokens, rebuilds it when a request is
replayed, and clears it when the request ends.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.token_gate import TokenGateConfig
from vllm.model_executor.layers.token_gate import TokenGateBatch, position_ratio
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.token_gate import TokenGateRequestOptions
from vllm.v1.attention.backends.token_gate import TokenGateAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class TokenGateRequestState:
    slot: int
    prompt_length: int
    options: TokenGateRequestOptions


class TokenGateHistory:
    """Own request channels and build independent views for each history domain."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.model_config = vllm_config.model_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = self.model_config.max_model_len
        self.device = device
        config = TokenGateConfig.from_hf_config(self.model_config.hf_text_config)
        assert config is not None
        self.gate_layer = config.gate_layer
        self.gain_rows = {
            layer: row for row, layer in enumerate(config.attention_layer_indices)
        }
        self.gate = model.model.layers[self.gate_layer].token_gate
        self.requests: dict[str, TokenGateRequestState] = {}
        self.prompt_channels = torch.zeros(
            (2, self.max_num_reqs, self.max_model_len),
            dtype=torch.float32,
            device=device,
        )
        self.batch: TokenGateBatch | None = None
        self.profile_workspace: torch.Tensor | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        assert new_req_data.sampling_params is not None
        options = TokenGateRequestOptions.parse(
            new_req_data.sampling_params.extra_args, new_req_data.prompt_len
        )
        self.prompt_channels[:, req_index].zero_()
        self.requests[new_req_data.req_id] = TokenGateRequestState(
            req_index, new_req_data.prompt_len, options
        )

    def remove_request(self, req_id: str) -> None:
        state = self.requests.pop(req_id, None)
        if state is not None:
            self.prompt_channels[:, state.slot].zero_()
        self.batch = None

    def prepare_batch(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        *,
        synthetic: bool = False,
        skip_attention: bool = False,
    ) -> None:
        if input_batch.num_reqs != 1:
            raise ValueError("TokenGate requires one active request")
        self.profile_workspace = None
        tokens = input_batch.num_tokens
        if synthetic:
            prompt_length = tokens
            options = (
                TokenGateRequestOptions(0, tokens - 1, tokens - 1)
                if tokens > 1
                else None
            )
            computed = 0
            channels = torch.zeros((2, tokens), device=self.device)
            positions = torch.arange(tokens, device=self.device)
            if skip_attention:
                # Account for scaled Q, key bias, and FP32/BF16 scale temporaries
                # that real paged attention allocates, without touching its caches.
                config = self.model_config.hf_text_config
                heads = config.num_attention_heads
                head_size = self.model_config.get_head_size()
                workspace_bytes = tokens * heads * (2 * head_size + 24)
                self.profile_workspace = torch.empty(
                    workspace_bytes, dtype=torch.uint8, device=self.device
                )
        else:
            state = self.requests[input_batch.req_ids[0]]
            prompt_length, options = state.prompt_length, state.options
            computed = int(input_batch.num_computed_tokens_np[0])
            channels = self.prompt_channels[:, state.slot]
            positions = input_batch.positions[:tokens]
            if computed == 0 and tokens < prompt_length:
                raise ValueError("TokenGate cannot execute a partial prompt")
            if 0 < computed < prompt_length:
                raise ValueError("TokenGate cannot resume a partial prompt")
        removed = options.removed_tokens if options is not None else 0
        retained_prompt = prompt_length - removed
        retained_queries = tokens - removed if computed == 0 else tokens
        upper_positions = (
            torch.empty(retained_queries, device=self.device, dtype=torch.int64)
            if computed == 0
            else positions
        )
        ratios = (
            torch.empty(retained_queries, device=self.device, dtype=torch.float32)
            if computed == 0
            else position_ratio(positions, computed - removed)
        )
        self.batch = TokenGateBatch(
            self.gate_layer,
            prompt_length,
            retained_prompt,
            computed,
            options,
            positions,
            upper_positions,
            torch.empty(retained_queries, device=self.device, dtype=torch.int64),
            ratios,
            channels[0, :retained_prompt],
            channels[1, :retained_prompt],
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        assert self.batch is not None
        return {
            "sequence_layout": self.batch,
            "positions": self.batch.original_positions,
        }

    def prepare_slot_mappings(
        self,
        input_batch: InputBatch,
        tables: BlockTables,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
    ) -> torch.Tensor:
        assert self.batch is not None
        mappings = tables.slot_mappings[:, : input_batch.num_tokens_after_padding]
        mappings.fill_(-1)
        removed = self.batch.original_prompt_length - self.batch.retained_prompt_length
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.first_spec
            if isinstance(spec, MambaSpec):
                continue
            count = (
                self.batch.upper_positions.numel()
                if group.uses_compact_sequence
                else input_batch.num_tokens
            )
            start = self.batch.num_computed_tokens
            if group.uses_compact_sequence and start:
                start -= removed
            positions = torch.arange(start, start + count, device=self.device)
            block_size = group.kv_cache_spec.block_size
            blocks = block_tables[i][0].index_select(0, positions // block_size)
            mappings[i, :count].copy_(
                blocks.long() * block_size + positions % block_size
            )
        return mappings

    def slot_mappings_by_layer(
        self,
        slot_mappings: torch.Tensor,
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, torch.Tensor]:
        assert self.batch is not None
        return {
            name: slot_mappings[
                i,
                : (
                    self.batch.upper_positions.numel()
                    if group.uses_compact_sequence
                    else self.batch.original_positions.numel()
                ),
            ]
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }

    def prepare_attn(
        self,
        prepare: Callable[..., dict[str, Any]],
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
        assert self.batch is not None
        if for_capture or cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError("TokenGate does not support CUDA graphs")
        result = {}
        for compact in (False, True):
            groups = [
                group if spec.uses_compact_sequence == compact else []
                for group, spec in zip(attn_groups, kv_cache_config.kv_cache_groups)
            ]
            batch = replace(
                input_batch,
                num_tokens_after_padding=input_batch.num_tokens,
                is_prefilling_np=np.array([self.batch.is_prefill]),
                has_prefill=self.batch.is_prefill,
                decode_graph_eligible=not self.batch.is_prefill,
                prefill_runs_as_decode_np=None,
            )
            if compact:
                queries = self.batch.upper_positions.numel()
                removed = (
                    self.batch.original_prompt_length
                    - self.batch.retained_prompt_length
                )
                computed = max(0, self.batch.num_computed_tokens - removed)
                seq_len = computed + queries
                query_locs = np.array([0, queries], dtype=np.int32)
                batch = replace(
                    batch,
                    num_tokens=queries,
                    num_tokens_after_padding=queries,
                    query_start_loc=torch.tensor(
                        [0, queries], dtype=torch.int32, device=self.device
                    ),
                    query_start_loc_np=query_locs,
                    max_query_len=queries,
                    num_scheduled_tokens=np.array([queries], dtype=np.int32),
                    seq_lens=torch.tensor(
                        [seq_len], dtype=torch.int32, device=self.device
                    ),
                    seq_lens_cpu_upper_bound=torch.tensor([seq_len], dtype=torch.int32),
                    positions=self.batch.upper_positions,
                    num_computed_tokens_np=np.array([computed], dtype=np.int32),
                    prefill_len_np=np.array(
                        [self.batch.retained_prompt_length], dtype=np.int32
                    ),
                    num_computed_prefill_tokens_np=np.array(
                        [min(computed, self.batch.retained_prompt_length)],
                        dtype=np.int32,
                    ),
                    prompt_lens=(
                        None
                        if input_batch.prompt_lens is None
                        else torch.tensor(
                            [self.batch.retained_prompt_length],
                            dtype=input_batch.prompt_lens.dtype,
                            device=self.device,
                        )
                    ),
                )
            metadata = prepare(
                batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                groups,
                kv_cache_config,
                ubatch_idx=ubatch_idx,
                model_specific_attn_metadata=model_specific_attn_metadata,
            )
            if compact:
                metadata = {
                    name: TokenGateAttentionMetadata(
                        **{
                            field.name: getattr(value, field.name)
                            for field in fields(TritonAttentionMetadata)
                        },
                        batch=self.batch,
                        gate=self.gate,
                        upper_layer_index=self.gain_rows[extract_layer_index(name)],
                    )
                    if extract_layer_index(name) in self.gain_rows
                    else value
                    for name, value in metadata.items()
                }
            result.update(metadata)
        return result


class TokenGateModelState(DefaultModelState):
    """TokenGate sequence views with native dense-model lifecycle handling."""

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
