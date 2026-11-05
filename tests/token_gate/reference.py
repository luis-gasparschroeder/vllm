# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provide an independent calculation of what a TokenGate model should predict.

For small test models, this code selects tokens from a prompt span and computes
attention and next-token scores with PyTorch and Hugging Face model weights.
It recomputes the entire prompt and generated history each time, avoiding
vLLM's cache machinery. Tests use it to check that token filtering, attention
adjustments, and rebuilding a cleared cache preserve the expected calculations.
"""

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb as apply_hybrid_rotary_pos_emb,
)


@torch.no_grad()
def reference_logits(model, gate, token_ids, original_prompt_length, options):
    config = model.config
    device = model.device
    ids = torch.tensor(token_ids, device=device)
    hidden = model.model.embed_tokens(ids)[None]
    positions = torch.arange(len(token_ids), device=device)[None]
    ratios = None
    scores = None
    retained = None
    original_scores = None
    gate_layer = gate["metadata"]["gate_layer"]
    hybrid = config.model_type == "qwen3_5_text"
    gain_rows = {
        layer: row
        for row, layer in enumerate(
            index
            for index, kind in enumerate(config.layer_types)
            if index > gate_layer and kind == "full_attention"
        )
    }
    weights = {key: value.to(device) for key, value in gate["state_dict"].items()}
    for index, layer in enumerate(model.model.layers):
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        if hybrid and config.layer_types[index] == "linear_attention":
            hidden = residual + layer.linear_attn(
                normalized, cache_params=None, attention_mask=None
            )
        else:
            attention = layer.self_attn
            shape = (*normalized.shape[:-1], -1, config.head_dim)
            projected = attention.q_proj(normalized)
            output_gate = None
            if hybrid:
                projected, output_gate = projected.view(
                    *normalized.shape[:-1], -1, 2 * config.head_dim
                ).chunk(2, dim=-1)
                output_gate = output_gate.reshape(*normalized.shape[:-1], -1)
            query = attention.q_norm(projected.reshape(shape)).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(
                1, 2
            )
            value = attention.v_proj(normalized).view(shape).transpose(1, 2)
            cos, sin = model.model.rotary_emb(normalized, positions)
            apply_rope = apply_hybrid_rotary_pos_emb if hybrid else apply_rotary_pos_emb
            query, key = apply_rope(query, key, cos, sin)
            bias = None
            if index > gate_layer:
                assert ratios is not None and scores is not None
                gain = F.softplus(weights["attn_scale"][gain_rows[index]])
                prefill_scale = (
                    gain.to(query.dtype)[:, None] * ratios.to(query.dtype)[None]
                )
                decode_scale = (gain[:, None] * ratios[None]).to(query.dtype)
                scale = torch.where(
                    positions < original_prompt_length, prefill_scale, decode_scale
                )
                query = query * scale[None, :, :, None]
                bias = (
                    scores[:, None]
                    * F.softplus(weights["gate_scale"][gain_rows[index]])
                ).to(query.dtype)
                key_scale = (
                    gain.to(query.dtype)[None]
                    * ratios[: scores.numel()].to(query.dtype)[:, None]
                )
                bias = bias * key_scale.mean(-1, keepdim=True)
            key = key.repeat_interleave(
                config.num_attention_heads // config.num_key_value_heads, 1
            )
            value = value.repeat_interleave(
                config.num_attention_heads // config.num_key_value_heads, 1
            )
            logits = (
                query.float() @ key.float().transpose(-2, -1) / config.head_dim**0.5
            )
            if bias is not None:
                logits[..., : bias.shape[0]] += bias.T[None, :, None].float()
            causal = torch.ones(
                hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=device
            ).tril()
            logits.masked_fill_(~causal, float("-inf"))
            output = logits.softmax(-1).to(value.dtype) @ value
            output = output.transpose(1, 2).reshape(*hidden.shape[:-1], -1)
            if output_gate is not None:
                output = output * output_gate.sigmoid()
            hidden = residual + attention.o_proj(output)
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        if index == gate_layer:
            start, end, keep = options
            span_hidden = hidden[0, start:end]
            dtype = hidden.dtype
            branch = F.silu(
                F.linear(span_hidden, weights["gate_branch.gate_proj.weight"].to(dtype))
            ) * F.linear(span_hidden, weights["gate_branch.up_proj.weight"].to(dtype))
            raw = F.linear(branch, weights["gate_branch.down_proj.weight"].to(dtype))[
                :, 0
            ]
            normalized = (
                raw.float() * torch.rsqrt(raw.float().square().mean() + 1e-6)
            ) * weights["b_norm.weight"]
            first_bias = normalized[:, None] * F.softplus(weights["gate_scale"][0])
            ranking = (first_bias * (1 / config.num_attention_heads)).sum(-1)
            selected = (
                ranking.topk(min(keep, end - start)).indices.sort().values + start
            )
            retained = torch.cat(
                (
                    torch.arange(start, device=device),
                    selected,
                    torch.arange(end, hidden.shape[1], device=device),
                )
            )
            hidden, residual = hidden[:, retained], residual[:, retained]
            positions = positions[:, retained]
            compact = torch.arange(1, retained.numel() + 1, device=device).float()
            ratios = (
                compact.clamp_min(2).log() / (retained.float() + 1).clamp_min(2).log()
            ).clamp_max(1)
            retained_prompt_length = (
                original_prompt_length - (end - start) + selected.numel()
            )
            scores = torch.zeros(retained_prompt_length, device=device)
            scores[start : start + selected.numel()] = normalized[selected - start]
            original_scores = normalized
        hidden = residual + layer.mlp(hidden)
    logits = model.lm_head(model.model.norm(hidden))[0, -1].float()
    return logits, retained, original_scores
