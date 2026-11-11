# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implement learned token pruning and attention adjustments for TokenGate.

During prefill (processing the prompt), TokenGate keeps the highest-scoring
tokens in a caller-selected span and every token outside it. The gate layer's
MLP and later layers then process the shorter sequence in its original order.
During decode (generating new tokens), no tokens are pruned. Retained prompt
scores and position corrections continue to adjust attention in later layers.
"""

from dataclasses import dataclass, field
from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F

from vllm.utils.token_gate import TokenGateRequestOptions


def position_ratio(positions: torch.Tensor, compact_start: int = 0) -> torch.Tensor:
    """Return the position correction used to scale attention after pruning.

    A retained token keeps its original position ID, but its index in the
    shorter sequence can decrease. Divide the log of the retained one-based
    position by the log of the original one-based position, clamping the ratio
    to at most one and both positions to at least two before taking logs.

    Args:
        positions: Original zero-based position IDs of retained tokens, in
            sequence order, with shape [num_tokens].
        compact_start: Number of retained tokens preceding this step; zero
            when processing the prompt from the start.

    Returns:
        FP32 correction factors with the same shape and device as positions.

    """
    compact = torch.arange(
        compact_start + 1,
        compact_start + positions.numel() + 1,
        dtype=torch.float32,
        device=positions.device,
    )
    return (
        compact.clamp_min(2).log() / (positions.float() + 1).clamp_min(2).log()
    ).clamp_max(1)


class TokenGate(nn.Module):
    """Score a prunable token span and retain tokens for later model layers.

    The scorer is a SwiGLU MLP that produces one scalar per token. Parameters
    are stored in FP32; scoring projections run in the input dtype. Scores
    are RMS-normalized across the span in FP32 and reused for attention bias.

    Attributes:
        input_gate: Projection whose SiLU activation gates the MLP values.
        input_up: Projection producing the MLP values.
        output: Projection from the gated MLP values to a scalar token score.
        gate_scale: Raw score gains, shaped [upper_layers, num_heads]; softplus
            makes the gains positive before ranking and attention biasing.
        attn_scale: Raw query gains of the same shape, transformed by softplus
            before adjusting attention strength.
        normalization_weight: Frozen scalar multiplier for normalized scores.
        epsilon: Stabilizer added to the mean squared score before normalization.

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        upper_layers: int,
        num_heads: int,
        epsilon: float = 1e-6,
    ) -> None:
        """Allocate the token scorer and per-layer attention parameters.

        Checkpoint weights must be loaded before use; the gain matrices are
        allocated without initialization.

        Args:
            hidden_size: Width of the model hidden states supplied to the scorer.
            intermediate_size: Width of the scorer's SwiGLU hidden layer.
            upper_layers: Number of full-attention layers after the gate, in model
                order. Layers using other token mixers do not consume gain rows.
            num_heads: Number of query heads in each of those attention layers.
            epsilon: Stabilizer for RMS normalization across the prunable span.

        """
        super().__init__()
        self.input_gate = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=torch.float32
        )
        self.input_up = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=torch.float32
        )
        self.output = nn.Linear(intermediate_size, 1, bias=False, dtype=torch.float32)
        self.gate_scale = nn.Parameter(
            torch.empty(upper_layers, num_heads, dtype=torch.float32)
        )
        self.attn_scale = nn.Parameter(torch.empty_like(self.gate_scale))
        self.normalization_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        self.epsilon = epsilon

    def score(
        self, hidden_states: torch.Tensor, span_lengths: list[int] | None = None
    ) -> torch.Tensor:
        """Compute pruning scores, normalizing each packed span independently.

        Normalization uses the root mean square within each supplied span, so
        changing that span can change its scores. Scores are signed values,
        not probabilities; larger values receive higher retention priority.

        Args:
            hidden_states: Normalized states before the gate layer's MLP,
                restricted to packed nonempty prunable spans. Shape is
                [total_span_tokens, hidden_size], on the gate parameters' device.
            span_lengths: Length of each packed span; None means one span.

        Returns:
            FP32 scores of shape [total_span_tokens], including normalization_weight,
            on the input device.

        """
        dtype = hidden_states.dtype
        gated = F.silu(F.linear(hidden_states, self.input_gate.weight.to(dtype)))
        values = F.linear(hidden_states, self.input_up.weight.to(dtype))
        raw = F.linear(gated * values, self.output.weight.to(dtype)).squeeze(-1)
        # Match prefill_batch_for_decode in the TokenGate reference repository:
        # normalize scalar scores in FP32 to avoid reduced-precision rounding.
        spans = raw.float().split(span_lengths) if span_lengths else (raw.float(),)
        normalized = [
            span * torch.rsqrt(span.square().mean() + self.epsilon) for span in spans
        ]
        return torch.cat(normalized) * self.normalization_weight

    def select(self, scores: torch.Tensor, keep_tokens: int) -> torch.Tensor:
        """Choose the highest-scoring tokens and restore their sequence order.

        Ranking uses the mean score bias across query heads in the first
        full-attention layer after the gate.

        Args:
            scores: Normalized FP32 scores for one span, shaped [span_length].
            keep_tokens: Number of tokens to retain, from one to span_length.
                The request parser clamps the budget before this method is called.

        Returns:
            Sorted int64 indices of shape [keep_tokens] on the scores' device.
            Indices are relative to the span, not the complete prompt.

        """
        bias = scores[:, None] * F.softplus(self.gate_scale[0])
        # Match the reference head mean: multiply by 1/H before summing.
        ranking = (bias * (1.0 / bias.shape[-1])).sum(-1)
        return torch.topk(ranking, keep_tokens).indices.sort().values

    def compact(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        batch: "TokenGateBatch",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prune each new prompt independently and preserve decode/replay rows.

        Args:
            hidden_states: Packed normalized states before the gate MLP.
            residual: Residual states in the same packed input order.
            batch: Per-request lengths, options, and persistent score buffers.

        Returns:
            Hidden and residual states in the packed retained layout. Persistent
            prompt scores are updated only for requests starting or replaying.

        """
        spans = []
        for request, start in zip(batch.requests, batch.input_offsets):
            if request.is_prefill and request.options is not None:
                span_options = request.options
                spans.append(
                    hidden_states[
                        start + span_options.span_start : start + span_options.span_end
                    ]
                )
        scores = iter(
            self.score(torch.cat(spans), [s.shape[0] for s in spans]).split(
                [s.shape[0] for s in spans]
            )
            if spans
            else ()
        )
        for i, request in enumerate(batch.requests):
            start, end = batch.input_offsets[i : i + 2]
            upper_start, upper_end = batch.output_offsets[i : i + 2]
            options = request.options
            device = hidden_states.device
            if request.is_prefill and options is not None:
                normalized = next(scores)
                selected = self.select(normalized, options.keep_tokens)
                indices = torch.cat(
                    (
                        torch.arange(options.span_start, device=device),
                        selected + options.span_start,
                        torch.arange(options.span_end, end - start, device=device),
                    )
                )
                request.prompt_scores.zero_()
                request.prompt_scores[
                    options.span_start : options.span_start + options.keep_tokens
                ].copy_(normalized.index_select(0, selected))
            else:
                indices = torch.arange(end - start, device=device)
                if request.is_prefill:
                    request.prompt_scores.zero_()
            rows = batch.retained_rows[upper_start:upper_end]
            rows.copy_(indices + start)
            positions = batch.upper_positions[upper_start:upper_end]
            positions.copy_(batch.original_positions.index_select(0, rows))
            ratios = batch.query_ratios[upper_start:upper_end]
            ratios.copy_(position_ratio(positions, request.compact_computed_tokens))
            batch.query_is_prompt[upper_start:upper_end].copy_(
                positions < request.original_prompt_length
            )
            if request.is_prefill:
                request.prompt_ratios.copy_(ratios[: request.retained_prompt_length])
        # Materialize in scheduled order once, not once per attention layer.
        width = max(r.retained_prompt_length for r in batch.requests)
        batch.prompt_scores = torch.stack(
            [
                F.pad(r.prompt_scores, (0, width - r.retained_prompt_length))
                for r in batch.requests
            ]
        )
        batch.prompt_ratios = torch.stack(
            [
                F.pad(r.prompt_ratios, (0, width - r.retained_prompt_length))
                for r in batch.requests
            ]
        )
        if not batch.is_prefill:
            return hidden_states, residual
        return (
            hidden_states.index_select(0, batch.retained_rows),
            residual.index_select(0, batch.retained_rows),
        )


@dataclass
class TokenGateSequence:
    """Describe one scheduled request and its persistent retained-prompt state.

    Token counts exclude other requests. Prompt channel tensors are views into
    the worker's request-slot storage; reordering a batch does not move ownership.
    """

    original_prompt_length: int
    retained_prompt_length: int
    num_computed_tokens: int
    num_tokens: int
    options: TokenGateRequestOptions | None
    prompt_scores: torch.Tensor
    prompt_ratios: torch.Tensor

    @property
    def is_prefill(self) -> bool:
        """Whether the complete prompt is being processed from its start."""
        return self.num_computed_tokens == 0

    @property
    def compact_computed_tokens(self) -> int:
        """Number of cached tokens in the upper layers before this step."""
        return max(0, self.num_computed_tokens - self.removed_tokens)

    @property
    def removed_tokens(self) -> int:
        """Number of prompt tokens omitted from upper-layer history."""
        return self.original_prompt_length - self.retained_prompt_length

    @property
    def num_retained_tokens(self) -> int:
        """Number of query rows this request contributes after the gate."""
        return self.num_tokens - (self.removed_tokens if self.is_prefill else 0)


@dataclass
class TokenGateBatch:
    """Map packed model rows to independently pruned requests.

    Input and output offsets use scheduled request order. Positions remain
    request-local original positions, while retained_rows indexes the packed
    input. Prompt scores/ratios are padded to the longest retained prompt with
    zeros, so generated keys and padding never inherit another request's bias.
    """

    gate_layer: int
    requests: list[TokenGateSequence]
    original_positions: torch.Tensor
    input_offsets: list[int] = field(init=False)
    output_offsets: list[int] = field(init=False)
    query_start_loc: torch.Tensor = field(init=False)
    upper_positions: torch.Tensor = field(init=False)
    retained_rows: torch.Tensor = field(init=False)
    query_ratios: torch.Tensor = field(init=False)
    query_is_prompt: torch.Tensor = field(init=False)
    prompt_scores: torch.Tensor = field(init=False)
    prompt_ratios: torch.Tensor = field(init=False)
    uncached_prefill_start: int | None = field(init=False)

    def __post_init__(self) -> None:
        self.input_offsets = list(
            accumulate((r.num_tokens for r in self.requests), initial=0)
        )
        self.output_offsets = list(
            accumulate((r.num_retained_tokens for r in self.requests), initial=0)
        )
        device = self.original_positions.device
        count = self.output_offsets[-1]
        self.query_start_loc = torch.tensor(
            self.output_offsets, dtype=torch.int32, device=device
        )
        self.upper_positions = torch.empty(count, dtype=torch.int64, device=device)
        self.retained_rows = torch.empty_like(self.upper_positions)
        self.query_ratios = torch.empty(count, dtype=torch.float32, device=device)
        self.query_is_prompt = torch.empty(count, dtype=torch.bool, device=device)
        uncached = [
            r.is_prefill and r.num_tokens == r.original_prompt_length
            for r in self.requests
        ]
        first = next((i for i, value in enumerate(uncached) if value), None)
        # The native runner orders decodes before prefills. Unusual layouts can
        # still use paged attention without reordering request or cache state.
        self.uncached_prefill_start = (
            first if first is not None and all(uncached[first:]) else None
        )

    @property
    def is_prefill(self) -> bool:
        """Whether any scheduled request starts or replays a complete prompt."""
        return any(r.is_prefill for r in self.requests)

    def positions_for_layer(self, layer_index: int) -> torch.Tensor:
        """Return original position IDs for the layer's full or retained rows."""
        return (
            self.original_positions
            if layer_index <= self.gate_layer
            else self.upper_positions
        )

    def restore_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore sampler row indices, including decode rows in a mixed step."""
        if not self.is_prefill:
            return hidden_states
        result = hidden_states.new_zeros(
            (self.original_positions.numel(), hidden_states.shape[-1])
        )
        return result.index_copy_(0, self.retained_rows, hidden_states)

    def attention_inputs(
        self,
        query: torch.Tensor,
        gate: TokenGate,
        upper_layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale packed queries and return request-local padded key biases.

        Args:
            query: Retained queries [tokens, heads, head_size], optionally with
                the last two dimensions flattened.
            gate: Loaded parameters on the query device.
            upper_layer_index: Gain row for this full-attention layer.

        Returns:
            Scaled queries with the input shape and a bias tensor shaped
            [requests, max_retained_prompt, heads], both in the query dtype.
            Generated queries retain the reference implementation's cast order.

        """
        dtype = query.dtype
        scale = F.softplus(gate.attn_scale[upper_layer_index])
        prefill_scale = scale.to(dtype)[None, :] * self.query_ratios.to(dtype)[:, None]
        decode_scale = (scale[None, :] * self.query_ratios[:, None]).to(dtype)
        query_scale = torch.where(
            self.query_is_prompt[:, None], prefill_scale, decode_scale
        )
        shaped = query.view(query.shape[0], scale.numel(), -1)
        scaled_query = (shaped * query_scale[:, :, None]).reshape_as(query)
        key_bias = (
            self.prompt_scores[:, :, None]
            * F.softplus(gate.gate_scale[upper_layer_index])
        ).to(dtype)
        key_scale = scale.to(dtype) * self.prompt_ratios.to(dtype)[:, :, None]
        key_bias = key_bias * key_scale.mean(dim=-1, keepdim=True)
        return scaled_query, key_bias
