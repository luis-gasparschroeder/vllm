# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implement learned token pruning and attention adjustments for TokenGate.

During prefill (processing the prompt), TokenGate keeps the highest-scoring
tokens in a caller-selected span and every token outside it. The gate layer's
MLP and later layers then process the shorter sequence in its original order.
During decode (generating new tokens), no tokens are pruned. Retained prompt
scores and position corrections continue to adjust attention in later layers.
"""

from dataclasses import dataclass

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

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute normalized pruning scores for one token span.

        Normalization uses the root mean square across all supplied tokens, so
        passing a different span can change the scores. Scores are signed values,
        not probabilities; larger values receive higher retention priority.

        Args:
            hidden_states: Normalized states before the gate layer's MLP,
                restricted to a nonempty prunable span. Shape is
                [span_length, hidden_size], on the gate parameters' device.

        Returns:
            FP32 scores of shape [span_length], including normalization_weight,
            on the input device.

        """
        dtype = hidden_states.dtype
        gated = F.silu(F.linear(hidden_states, self.input_gate.weight.to(dtype)))
        values = F.linear(hidden_states, self.input_up.weight.to(dtype))
        raw = F.linear(gated * values, self.output.weight.to(dtype)).squeeze(-1)
        # Match prefill_batch_for_decode in the TokenGate reference repository:
        # normalize scalar scores in FP32 to avoid reduced-precision rounding.
        normalized = raw.float() * torch.rsqrt(
            raw.float().square().mean() + self.epsilon
        )
        return normalized * self.normalization_weight

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
        """Prune prompt rows before the gate layer's MLP and prepare attention state.

        On prefill, keep the selected span rows and every row outside the span.
        Update batch's retained rows, original position IDs for those rows, and
        attention correction buffers in place. Protected prompt scores are zero.
        On decode, return the inputs without changing batch. Synthetic warmup
        with no span retains every row.

        Args:
            hidden_states: Normalized states before the gate layer's MLP,
                shaped [num_tokens, hidden_size].
            residual: Residual stream with the same shape and token order.
            batch: This step's layout, with validated span options and GPU buffers
                already allocated for the retained sequence length.

        Returns:
            A tuple (hidden_states, residual) with matching retained rows in
            original order, preserving each input's dtype and device. Decode
            returns the input tensors themselves.

        """
        if not batch.is_prefill:
            return hidden_states, residual
        options = batch.options
        if options is None:
            # The one-token synthetic warmup has no prunable span.
            indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            normalized = hidden_states.new_empty(0, dtype=torch.float32)
        else:
            start, end = options.span_start, options.span_end
            normalized = self.score(hidden_states[start:end])
            selected = self.select(normalized, options.keep_tokens) + start
            indices = torch.cat(
                (
                    torch.arange(start, device=selected.device),
                    selected,
                    torch.arange(end, hidden_states.shape[0], device=selected.device),
                )
            )
        batch.retained_rows.copy_(indices)
        batch.upper_positions.copy_(batch.original_positions.index_select(0, indices))
        batch.query_ratios.copy_(position_ratio(batch.upper_positions))
        batch.prompt_scores.zero_()
        if options is not None:
            begin = options.span_start
            selected = indices[begin : begin + options.keep_tokens] - begin
            batch.prompt_scores[begin : begin + options.keep_tokens].copy_(
                normalized.index_select(0, selected)
            )
        batch.prompt_ratios.copy_(batch.query_ratios[: batch.retained_prompt_length])
        return (
            hidden_states.index_select(0, indices),
            residual.index_select(0, indices),
        )


@dataclass
class TokenGateBatch:
    """Track original and retained token layouts for one request's model step.

    Prefill buffers are populated by TokenGate.compact. Prompt scores and
    ratios are views into worker-owned storage that persists across decode
    steps; the other tensors describe only the current step.

    Attributes:
        gate_layer: Zero-based model layer index containing the pruning gate.
        original_prompt_length: Prompt length before pruning, excluding any
            generated tokens replayed with the prompt.
        retained_prompt_length: Prompt length after pruning.
        num_computed_tokens: Previously processed tokens in the original
            sequence; zero for initial prefill or replay from the start.
        options: Validated prunable span and budget; None for one-token warmup.
        original_positions: Original position IDs for this step's input rows,
            shaped [num_input_tokens].
        upper_positions: Original position IDs for rows after pruning, shaped
            [num_retained_tokens]. Decode uses original_positions unchanged.
        retained_rows: Int64 row indices into the prefill input, shaped
            [num_retained_tokens]. Unused during decode.
        query_ratios: FP32 position corrections for this step's retained query
            rows, shaped [num_retained_tokens].
        prompt_scores: FP32 scores of retained prompt tokens, shaped
            [retained_prompt_length]; zero for tokens outside the prunable span.
        prompt_ratios: FP32 position corrections for retained prompt tokens,
            shaped [retained_prompt_length].

    """

    gate_layer: int
    original_prompt_length: int
    retained_prompt_length: int
    num_computed_tokens: int
    options: TokenGateRequestOptions | None
    original_positions: torch.Tensor
    upper_positions: torch.Tensor
    retained_rows: torch.Tensor
    query_ratios: torch.Tensor
    prompt_scores: torch.Tensor
    prompt_ratios: torch.Tensor

    @property
    def is_prefill(self) -> bool:
        """Whether this step processes the prompt from the start, including replay."""
        return self.num_computed_tokens == 0

    def positions_for_layer(self, layer_index: int) -> torch.Tensor:
        """Return original position IDs for the rows a layer will process.

        The gate layer's token mixer still sees all input rows. Later layers see
        only retained rows, with their original position IDs preserved for RoPE.

        Args:
            layer_index: Zero-based model layer index, not an attention-gain row.

        Returns:
            The original_positions tensor through the gate layer, or the
            upper_positions tensor after it; no position IDs are renumbered.

        """
        return (
            self.original_positions
            if layer_index <= self.gate_layer
            else self.upper_positions
        )

    def restore_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Place final prefill hidden states back at their original input rows.

        This restores the row layout expected by vLLM's sampling indices. Pruned
        rows contain zeros; the final prompt token is protected and retains its
        computed hidden state.

        Args:
            hidden_states: Final states in retained order, shaped
                [num_retained_tokens, hidden_size].

        Returns:
            On prefill, a tensor of shape [num_input_tokens, hidden_size] with
            the input dtype and device. On decode, the input tensor itself.

        """
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
        """Adjust queries and prompt-key biases for attention after the gate.

        Each query is multiplied by its position correction and a learned
        per-head gain. Retained prompt scores supply an additive attention-logit
        bias, weighted by the mean query gain at each prompt key's position.
        Protected prompt keys have zero bias; generated keys have no entries
        in this bias tensor and receive zero bias in the attention kernel.

        Args:
            query: Queries in retained sequence order, shaped
                [num_queries, num_heads, head_size], or with the last two
                dimensions flattened. The dtype is the model's attention dtype.
            gate: Loaded TokenGate parameters on the same device as query.
            upper_layer_index: Zero-based row in the gain matrices, counting only
                full-attention layers after the gate, not all model layers.

        Returns:
            A tuple (scaled_query, key_bias), both in query's dtype and device.
            scaled_query has query's shape; key_bias has shape
            [retained_prompt_length, num_heads]. The input query is not modified.

        """
        dtype = query.dtype
        scale = F.softplus(gate.attn_scale[upper_layer_index])
        # Match the TokenGate reference: prompt rows cast before multiplying;
        # generated rows multiply in FP32 first. The rounding differs.
        prefill_scale = scale.to(dtype)[None, :] * self.query_ratios.to(dtype)[:, None]
        decode_scale = (scale[None, :] * self.query_ratios[:, None]).to(dtype)
        # Replay can mix prompt and generated rows in the same prefill step.
        query_scale = torch.where(
            (self.upper_positions < self.original_prompt_length)[:, None],
            prefill_scale,
            decode_scale,
        )
        shaped = query.view(query.shape[0], scale.numel(), -1)
        scaled_query = (shaped * query_scale[:, :, None]).reshape_as(query)
        key_bias = (
            self.prompt_scores[:, None] * F.softplus(gate.gate_scale[upper_layer_index])
        ).to(dtype)
        key_scale = scale.to(dtype)[None, :] * self.prompt_ratios.to(dtype)[:, None]
        key_bias = key_bias * key_scale.mean(dim=-1, keepdim=True)
        return scaled_query, key_bias
