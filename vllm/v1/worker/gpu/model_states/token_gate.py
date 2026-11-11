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
from vllm.model_executor.layers.token_gate import TokenGateBatch, TokenGateSequence
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.token_gate import TokenGateRequestOptions
from vllm.v1.attention.backends.token_gate import TokenGateAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
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
        self.profile_workspace = None
        requests = []
        positions = input_batch.positions[: input_batch.num_tokens]
        if synthetic:
            positions = torch.cat(
                [
                    torch.arange(int(length), device=self.device)
                    for length in input_batch.num_scheduled_tokens
                ]
            )
        for i, length in enumerate(input_batch.num_scheduled_tokens):
            tokens = int(length)
            if synthetic:
                prompt_length, computed = tokens, 0
                options = (
                    TokenGateRequestOptions(0, tokens - 1, tokens - 1)
                    if tokens > 1
                    else None
                )
                channels = torch.zeros((2, tokens), device=self.device)
            else:
                state = self.requests[input_batch.req_ids[i]]
                if state.slot != int(input_batch.idx_mapping_np[i]):
                    raise ValueError(
                        "TokenGate request slot does not match the scheduled row"
                    )
                prompt_length, options = state.prompt_length, state.options
                computed = int(input_batch.num_computed_tokens_np[i])
                channels = self.prompt_channels[:, state.slot]
                if computed == 0 and tokens < prompt_length:
                    raise ValueError("TokenGate cannot execute a partial prompt")
                if 0 < computed < prompt_length:
                    raise ValueError("TokenGate cannot resume a partial prompt")
            removed = options.removed_tokens if options is not None else 0
            retained_prompt = prompt_length - removed
            requests.append(
                TokenGateSequence(
                    prompt_length,
                    retained_prompt,
                    computed,
                    tokens,
                    options,
                    channels[0, :retained_prompt],
                    channels[1, :retained_prompt],
                )
            )
        self.batch = TokenGateBatch(self.gate_layer, requests, positions)
        if skip_attention:
            config = self.model_config.hf_text_config
            heads = config.num_attention_heads
            size = self.model_config.get_head_size()
            tokens = self.batch.upper_positions.numel()
            prefix = len(requests) * max(r.retained_prompt_length for r in requests)
            # Scaled queries plus FP32/BF16 query scales, key scales, and biases.
            workspace_bytes = heads * (tokens * (2 * size + 12) + prefix * 12)
            self.profile_workspace = torch.empty(
                workspace_bytes, dtype=torch.uint8, device=self.device
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
        full = tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            self.batch.original_positions,
            input_batch.num_tokens_after_padding,
        )
        # The second call must not overwrite full-history mappings.
        compact_positions = torch.cat(
            [
                torch.arange(
                    r.compact_computed_tokens,
                    r.compact_computed_tokens + r.num_retained_tokens,
                    device=self.device,
                )
                for r in self.batch.requests
            ]
        )
        compact = tables.compute_slot_mappings(
            input_batch.idx_mapping,
            self.batch.query_start_loc,
            compact_positions,
            input_batch.num_tokens_after_padding,
            out=torch.empty_like(tables.slot_mappings),
        )
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            if group.uses_compact_sequence:
                full[i].copy_(compact[i])
        return full

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
                is_prefilling_np=np.array([r.is_prefill for r in self.batch.requests]),
                has_prefill=self.batch.is_prefill,
                decode_graph_eligible=not self.batch.is_prefill,
                prefill_runs_as_decode_np=None,
            )
            if compact:
                requests = self.batch.requests
                lengths = np.array(
                    [r.num_retained_tokens for r in requests], dtype=np.int32
                )
                computed = np.array(
                    [r.compact_computed_tokens for r in requests], dtype=np.int32
                )
                prompts = np.array(
                    [r.retained_prompt_length for r in requests], dtype=np.int32
                )
                seq_lens = computed + lengths
                queries = int(lengths.sum())
                batch = replace(
                    batch,
                    num_tokens=queries,
                    num_tokens_after_padding=queries,
                    query_start_loc=self.batch.query_start_loc,
                    query_start_loc_np=np.array(
                        self.batch.output_offsets, dtype=np.int32
                    ),
                    max_query_len=int(lengths.max()),
                    num_scheduled_tokens=lengths,
                    seq_lens=torch.tensor(
                        seq_lens, dtype=torch.int32, device=self.device
                    ),
                    seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens),
                    positions=self.batch.upper_positions,
                    num_computed_tokens_np=computed,
                    prefill_len_np=prompts,
                    num_computed_prefill_tokens_np=np.minimum(computed, prompts),
                    max_seq_len_np=seq_lens
                    if input_batch.max_seq_len_np is not None
                    else None,
                    prompt_lens=(
                        None
                        if input_batch.prompt_lens is None
                        else torch.tensor(
                            prompts,
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
