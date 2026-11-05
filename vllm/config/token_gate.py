# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that a model and serving setup support TokenGate token filtering.

TokenGate removes some prompt tokens partway through the model, so earlier and
later layers process different sequence lengths. This module reads where the
gate is placed and which attention rules its checkpoint requires, then rejects
incompatible models or execution settings. It supports Qwen3 and text-only
Qwen3.5 with a complete prompt processed in one step and one active request.
"""

from dataclasses import dataclass
from typing import Any


def attention_layer_indices(config: Any, gate_layer: int) -> tuple[int, ...]:
    """Validate supported topology and return full-attention layers above the gate."""
    if config.model_type not in ("qwen3", "qwen3_5_text") or not getattr(
        config, "is_causal", True
    ):
        raise ValueError("TokenGate supports causal dense Qwen3 and text Qwen3.5 only")
    if (
        type(gate_layer) is not int
        or not 0 <= gate_layer < config.num_hidden_layers - 1
    ):
        raise ValueError("TokenGate needs a gate layer followed by attention layers")
    if (
        getattr(config, "sliding_window", None) is not None
        or getattr(config, "rswa_window", None) is not None
        or getattr(config, "dual_chunk_attention_config", None)
    ):
        raise ValueError("TokenGate requires ordinary full attention")
    if getattr(config, "layer_scale", False):
        raise ValueError("TokenGate does not support decoder layer scaling")
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None and config.model_type == "qwen3":
        layer_types = ["full_attention"] * config.num_hidden_layers
    supported = {"full_attention"}
    if config.model_type == "qwen3_5_text":
        supported.add("linear_attention")
    if (
        not isinstance(layer_types, (list, tuple))
        or len(layer_types) != config.num_hidden_layers
        or any(
            not isinstance(kind, str) or kind not in supported for kind in layer_types
        )
    ):
        raise ValueError("TokenGate requires a complete supported layer_types list")
    indices = tuple(
        index
        for index, kind in enumerate(layer_types)
        if index > gate_layer and kind == "full_attention"
    )
    if not indices:
        raise ValueError("TokenGate needs full-attention layers above the gate")
    return indices


@dataclass(frozen=True)
class TokenGateConfig:
    gate_layer: int
    attention_layer_indices: tuple[int, ...]
    normalization_epsilon: float = 1e-6

    @classmethod
    def from_hf_config(cls, config: Any) -> "TokenGateConfig | None":
        data = getattr(config, "token_gate", None)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("token_gate must be a versioned configuration object")
        version = data.get("format_version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported TokenGate format_version")
        if version == 1 and config.model_type != "qwen3":
            raise ValueError("TokenGate format_version 1 supports Qwen3 only")
        required = {
            "attention_semantics": "tokengate_code_v1",
            "precision": "bfloat16",
            "selection_mode": "topk",
            "random_topk": False,
            "decode_phase2_form": True,
            "prune_below_gate": False,
        }
        for key, expected in required.items():
            if type(data.get(key)) is not type(expected) or data[key] != expected:
                raise ValueError(f"TokenGate requires {key}={expected!r}")
        layer = data.get("gate_layer")
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise ValueError("TokenGate requires an integer gate_layer")
        indices = attention_layer_indices(config, layer)
        if version == 2:
            declared = data.get("attention_layer_indices")
            if (
                not isinstance(declared, list)
                or any(type(index) is not int for index in declared)
                or tuple(declared) != indices
            ):
                raise ValueError(
                    "TokenGate attention_layer_indices disagree with topology"
                )
        provenance = data.get("provenance")
        if not isinstance(provenance, dict) or not all(
            provenance.get(key)
            for key in (
                "base_model",
                "gate_sha256",
                "adapter_sha256",
                "tokenizer_sha256",
                "reference_revision",
                "selection_policy",
            )
        ):
            raise ValueError(
                "TokenGate export lacks verified artifact/policy provenance"
            )
        return cls(layer, indices)

    def validate_execution(self, config: Any) -> None:
        import torch

        from vllm.platforms import current_platform

        model = config.model_config
        scheduler = config.scheduler_config
        parallel = config.parallel_config
        cache = config.cache_config
        hybrid = model.hf_text_config.model_type == "qwen3_5_text"
        unsupported = {
            "hybrid cache manager disabled": (
                hybrid and scheduler.disable_hybrid_kv_cache_manager
            ),
            "recurrent cache checkpoints": hybrid and cache.mamba_cache_mode != "none",
            "non-CUDA execution": not current_platform.is_cuda(),
            "non-BF16 weights": model.dtype != torch.bfloat16,
            "another model runner": not config.use_v2_model_runner,
            "graphs or compilation": not model.enforce_eager,
            "chunked prefill": scheduler.enable_chunked_prefill,
            "async scheduling": scheduler.async_scheduling,
            "multiple active requests": scheduler.max_num_seqs != 1,
            "prefill threshold": scheduler.long_prefill_token_threshold != 0,
            "prefix caching": cache.enable_prefix_caching,
            "quantized KV cache": cache.cache_dtype != "auto",
            "KV sharing": cache.kv_sharing_fast_prefill,
            "speculative decoding": config.speculative_config is not None,
            "dynamic adapters": config.lora_config is not None,
            "quantization": config.quant_config is not None,
            "prompt embeddings": model.enable_prompt_embeds,
            "multimodal inputs": model.multimodal_config is not None,
            "pooling": model.runner_type != "generate",
            "cache transfer": config.kv_transfer_config is not None,
            "auxiliary outputs": config.aux_output_config.enabled,
            "HiSparse": config.attention_config.hisparse_config is not None,
            "parallel execution": any(
                getattr(parallel, key, 1) != 1
                for key in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "decode_context_parallel_size",
                    "prefill_context_parallel_size",
                )
            ),
            "weight offload": (
                config.offload_config.uva.cpu_offload_gb > 0
                or config.offload_config.prefetch.offload_group_size > 0
            ),
            "cache offload": cache.kv_offloading_size is not None,
        }
        failures = [name for name, enabled in unsupported.items() if enabled]
        if failures:
            raise ValueError("TokenGate does not support: " + ", ".join(failures))
        scheduled = scheduler.max_num_scheduled_tokens
        if scheduled is None:
            scheduled = scheduler.max_num_batched_tokens
        if min(scheduled, scheduler.max_num_batched_tokens) < model.max_model_len:
            raise ValueError("TokenGate requires both token budgets >= max_model_len")
