# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check TokenGate's request rules and token-filtering calculations on the CPU.

These CPU tests verify that requests identify valid token-span boundaries and a
valid number of tokens to keep. They check that filtering preserves the other
tokens, their original positions, and their model state, and that scoring and
attention calculations follow the required floating-point rules. Unsupported
checkpoint settings and prompt transformations must be rejected.
"""

import json

import pytest
import torch
from pydantic import ValidationError
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from vllm.config.token_gate import TokenGateConfig
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.model_executor.layers.token_gate import (
    TokenGate,
    TokenGateBatch,
    TokenGateSequence,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.token_gate import (
    TOKEN_GATE_FIELDS,
    TokenGateRequestOptions,
    validate_token_gate_sampling,
    validate_token_gate_transformations,
)
from vllm.v1.core.kv_cache_layout import (
    CacheSequenceLayout,
    KVCacheTokenCounts,
    compute_group_token_counts,
)


def options(**kwargs):
    return {
        "token_gate_version": 1,
        "token_gate_span_start": 2,
        "token_gate_span_end": 8,
        "token_gate_keep_tokens": 3,
        **kwargs,
    }


@pytest.mark.parametrize("key", TOKEN_GATE_FIELDS)
@pytest.mark.parametrize("value", [True, False, 1.0, "1", [1], None])
def test_api_rejects_raw_non_integer_gate_fields(key, value):
    with pytest.raises(ValidationError):
        CompletionRequest.model_validate_json(
            json.dumps(
                {
                    "model": "gate",
                    "prompt": [1] * 10,
                    "vllm_xargs": options(**{key: value}),
                }
            )
        )


def test_api_preserves_unrelated_extension_fields():
    request = CompletionRequest.model_validate_json(
        json.dumps(
            {
                "model": "gate",
                "prompt": [1] * 10,
                "vllm_xargs": {
                    **options(),
                    "custom_float": 0.25,
                    "custom_text": "hello",
                },
            }
        )
    )
    assert request.vllm_xargs == {
        **options(),
        "custom_float": 0.25,
        "custom_text": "hello",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"truncate_prompt_tokens": 8},
        {"tokenization_kwargs": {"truncation": True}},
        {"tokenization_kwargs": {"padding": "max_length"}},
        {"tokenization_kwargs": {"pad_prompt_tokens": 10}},
    ],
)
def test_transformation_rejected_even_if_stale_span_remains_in_bounds(kwargs):
    assert TokenGateRequestOptions.parse(options(), 10).span_end == 8
    with pytest.raises(ValueError, match="unchanged"):
        validate_token_gate_transformations(options(), **kwargs)


def test_independent_prompt_scoring_is_rejected():
    params = SamplingParams(
        extra_args=options(), prompt_logprob_token_ids=[3], prompt_logprobs=None
    )
    with pytest.raises(ValueError, match="prompt scoring"):
        validate_token_gate_sampling(params)


@pytest.mark.parametrize("prompt_length,keep", [(9, 1), (10, 3), (10, 20)])
def test_lengths_preserve_protected_tokens(prompt_length, keep):
    parsed = TokenGateRequestOptions.parse(
        options(token_gate_keep_tokens=keep), prompt_length
    )
    assert parsed.keep_tokens == min(keep, 6)
    assert prompt_length - parsed.removed_tokens == prompt_length - 6 + min(keep, 6)


@pytest.mark.parametrize("endpoint,expected", [(0, 0), (10, 7), (11, 8), (33, 30)])
def test_group_counts_project_endpoints_not_deltas(endpoint, expected):
    counts = KVCacheTokenCounts(endpoint, endpoint, endpoint, endpoint, endpoint)
    early, upper = compute_group_token_counts(
        counts, CacheSequenceLayout(10, 7), (False, True)
    )
    assert early == counts
    assert upper == KVCacheTokenCounts(expected, expected, expected, expected, expected)


def test_partial_prefill_is_not_projectable():
    with pytest.raises(ValueError, match="complete prompt"):
        CacheSequenceLayout(10, 7).project(9)


@pytest.mark.parametrize("length,keep", [(12, 5), (2048, 64)])
def test_scorer_matches_expanded_qwen_normalization(length, keep):
    torch.manual_seed(2)
    gate = TokenGate(16, 32, 2, 4)
    with torch.no_grad():
        gate.gate_scale.fill_(0.3)
        gate.attn_scale.fill_(0.2)
    hidden = torch.randn(length, 16, dtype=torch.bfloat16)
    branch = torch.nn.functional.silu(
        hidden @ gate.input_gate.weight.to(hidden.dtype).T
    ) * (hidden @ gate.input_up.weight.to(hidden.dtype).T)
    raw = branch @ gate.output.weight.to(hidden.dtype).T
    norm = Qwen3RMSNorm(1, eps=1e-6)
    # The original TokenGate implementation normalizes gate scores in FP32.
    # Expand them across heads as E2EGate.forward does before Qwen3RMSNorm.
    expanded = raw.float()[:, None].expand(-1, 2, 4).movedim(0, -1)
    expected = norm(expanded).movedim(-1, 0)[:, 0, 0]
    torch.testing.assert_close(gate.score(hidden), expected, rtol=0, atol=0)
    bias = expected[:, None] * torch.nn.functional.softplus(gate.gate_scale[0])
    ranking = (bias * torch.full((4,), 0.25)).sum(-1)
    expected_indices = ranking.topk(keep).indices.sort().values
    actual = gate.score(hidden)
    assert actual.dtype == torch.float32
    assert torch.equal(gate.select(actual, keep), expected_indices)


def test_compaction_preserves_residual_positions_and_output_rows():
    torch.manual_seed(11)
    gate = TokenGate(8, 16, 1, 2)
    with torch.no_grad():
        gate.gate_scale.fill_(0.1)
        gate.attn_scale.fill_(0.4)
    # Two already-generated tokens participate in replay but never in selection.
    hidden = torch.randn(12, 8, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    batch = TokenGateBatch(
        0,
        [
            TokenGateSequence(
                10,
                7,
                0,
                12,
                TokenGateRequestOptions(2, 8, 3),
                torch.empty(7),
                torch.empty(7),
            )
        ],
        torch.arange(12),
    )
    compact, compact_residual = gate.compact(hidden, residual, batch)
    assert batch.retained_rows[:2].tolist() == [0, 1]
    assert batch.retained_rows[-4:].tolist() == [8, 9, 10, 11]
    torch.testing.assert_close(compact, hidden[batch.retained_rows], rtol=0, atol=0)
    torch.testing.assert_close(
        compact_residual, residual[batch.retained_rows], rtol=0, atol=0
    )
    restored = batch.restore_output(compact)
    assert torch.equal(restored[batch.retained_rows], compact)
    assert torch.count_nonzero(restored).item() == torch.count_nonzero(compact).item()
    assert torch.equal(batch.upper_positions, batch.retained_rows)


def test_replayed_generated_queries_use_decode_cast_order():
    gate = TokenGate(8, 16, 1, 2)
    with torch.no_grad():
        gate.gate_scale.copy_(torch.tensor([[0.77, 1.21]]))
        gate.attn_scale.copy_(torch.tensor([[0.13, 1.77]]))
    ratios = torch.full((8,), 0.7031)
    batch = TokenGateBatch(
        0,
        [
            TokenGateSequence(
                10,
                7,
                0,
                11,
                TokenGateRequestOptions(2, 8, 3),
                torch.ones(7),
                torch.full((7,), 0.73),
            )
        ],
        torch.arange(11),
    )
    batch.upper_positions.copy_(torch.tensor([0, 1, 3, 5, 7, 8, 9, 10]))
    batch.query_ratios.copy_(ratios)
    batch.query_is_prompt.copy_(batch.upper_positions < 10)
    batch.prompt_scores = torch.ones(1, 7)
    batch.prompt_ratios = torch.full((1, 7), 0.73)
    query = torch.ones(8, 2, 4, dtype=torch.bfloat16)
    transformed, bias = batch.attention_inputs(query, gate, 0)
    gain = torch.nn.functional.softplus(gate.attn_scale[0])
    prefill = gain.bfloat16() * ratios[0].bfloat16()
    decode = (gain * ratios[1]).bfloat16()
    assert torch.equal(transformed[-2, :, 0], prefill)
    assert torch.equal(transformed[-1, :, 0], decode)
    expected_bias = torch.nn.functional.softplus(gate.gate_scale[0]).bfloat16()
    expected_bias = (
        expected_bias * (gain.bfloat16() * torch.tensor(0.73).bfloat16()).mean()
    )
    assert torch.equal(bias[0, 0], expected_bias)


@pytest.mark.parametrize(
    "key,value",
    [
        ("selection_mode", "threshold"),
        ("random_topk", True),
        ("decode_phase2_form", False),
        ("prune_below_gate", True),
        ("format_version", True),
        ("format_version", 1),
        ("attention_semantics", "paper"),
    ],
)
def test_checkpoint_semantics_fail_closed(key, value):
    config = Qwen3Config(num_hidden_layers=3)
    config.token_gate = {
        "format_version": 2,
        "attention_layer_indices": [1, 2],
        "gate_layer": 0,
        "selection_mode": "topk",
        "random_topk": False,
        "decode_phase2_form": True,
        "prune_below_gate": False,
        "attention_semantics": "tokengate_code_v1",
        "precision": "bfloat16",
        "provenance": dict.fromkeys(
            (
                "base_model",
                "gate_sha256",
                "adapter_sha256",
                "tokenizer_sha256",
            ),
            "test-evidence",
        ),
    }
    assert TokenGateConfig.from_hf_config(config) == TokenGateConfig(0, (1, 2))
    config.token_gate[key] = value
    with pytest.raises(ValueError, match=key):
        TokenGateConfig.from_hf_config(config)


def test_hybrid_reference_with_neutral_gate_matches_hf(tmp_path):
    """Qualify the independent oracle's native output gate and partial RoPE."""
    from transformers import AutoModelForCausalLM

    from tests.token_gate.assets import create_artifacts
    from tests.token_gate.reference import reference_logits
    from vllm.platforms import current_platform

    artifacts = create_artifacts(tmp_path, hybrid=True, gate_layer=1)
    device = "cuda" if current_platform.is_cuda() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(
            artifacts["base_path"], dtype=torch.float32, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    gate = torch.load(artifacts["gate_path"], weights_only=True)
    gate["state_dict"]["gate_branch.down_proj.weight"].zero_()
    gate["state_dict"]["attn_scale"].fill_(torch.tensor(1.0).expm1().log().item())
    tokens = list(range(4, 16))
    actual, retained, scores = reference_logits(model, gate, tokens, 12, (2, 10, 8))
    with torch.no_grad():
        expected = model(torch.tensor([tokens], device=device)).logits[0, -1]
    assert retained.tolist() == list(range(12))
    assert torch.count_nonzero(scores) == 0
    torch.testing.assert_close(actual, expected)


def test_valid_sampling_preserves_aligned_coordinates():
    params = SamplingParams(extra_args=options(), temperature=0, max_tokens=3)
    validate_token_gate_sampling(params)
    assert TokenGateRequestOptions.parse(
        params.extra_args, 10
    ) == TokenGateRequestOptions(2, 8, 3)


@pytest.mark.parametrize("reverse", [False, True])
def test_packed_spans_normalize_independently(reverse):
    """A large neighboring span must not change another request's gate scores."""
    torch.manual_seed(19)
    gate = TokenGate(8, 16, 1, 2)
    spans = [torch.randn(9, 8), torch.randn(5, 8) * 20]
    if reverse:
        spans.reverse()
    lengths = [span.shape[0] for span in spans]
    actual = gate.score(torch.cat(spans), lengths).split(lengths)
    for observed, span in zip(actual, spans):
        torch.testing.assert_close(observed, gate.score(span))


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_batch_preserves_request_boundaries_and_persistent_scores(reverse):
    """Reorder a replay, a new prompt, and a decode without mixing their state."""
    torch.manual_seed(17)
    gate = TokenGate(8, 16, 1, 2)
    with torch.no_grad():
        gate.gate_scale.fill_(0.2)
        gate.attn_scale.fill_(0.7)
    # Prompt length, computed tokens, scheduled tokens, span, keep count.
    cases = [(10, 0, 12, (2, 8, 3)), (7, 9, 1, (0, 6, 1)), (9, 0, 9, (1, 7, 2))]
    if reverse:
        cases.reverse()
    requests, positions, hidden, residual = [], [], [], []
    for prompt, computed, count, span in cases:
        opts = TokenGateRequestOptions(*span)
        retained = prompt - opts.removed_tokens
        requests.append(
            TokenGateSequence(
                prompt,
                retained,
                computed,
                count,
                opts,
                torch.linspace(0, 1, retained),
                torch.linspace(0.5, 1, retained),
            )
        )
        positions.append(torch.arange(computed, computed + count))
        hidden.append(torch.randn(count, 8))
        residual.append(torch.randn(count, 8))
    expected = []
    for request, pos, h, r in zip(requests, positions, hidden, residual):
        independent = TokenGateSequence(
            request.original_prompt_length,
            request.retained_prompt_length,
            request.num_computed_tokens,
            request.num_tokens,
            request.options,
            request.prompt_scores.clone(),
            request.prompt_ratios.clone(),
        )
        single = TokenGateBatch(0, [independent], pos)
        compact, compact_residual = gate.compact(h, r, single)
        q, bias = single.attention_inputs(compact.reshape(-1, 2, 4), gate, 0)
        expected.append((single, compact, compact_residual, q, bias))
    batch = TokenGateBatch(0, requests, torch.cat(positions))
    assert batch.uncached_prefill_start == (None if reverse else 2)
    compact, compact_residual = gate.compact(
        torch.cat(hidden), torch.cat(residual), batch
    )
    scaled, bias = batch.attention_inputs(compact.reshape(-1, 2, 4), gate, 0)
    restored = batch.restore_output(compact)
    for i, (single, h, r, q, b) in enumerate(expected):
        lo, hi = batch.output_offsets[i : i + 2]
        in_lo, in_hi = batch.input_offsets[i : i + 2]
        torch.testing.assert_close(compact[lo:hi], h)
        torch.testing.assert_close(compact_residual[lo:hi], r)
        torch.testing.assert_close(scaled[lo:hi], q)
        width = requests[i].retained_prompt_length
        torch.testing.assert_close(bias[i, :width], b[0])
        assert torch.count_nonzero(bias[i, width:]) == 0
        assert torch.equal(batch.retained_rows[lo:hi] - in_lo, single.retained_rows)
        assert torch.equal(restored[in_hi - 1], hidden[i][-1])
        torch.testing.assert_close(
            requests[i].prompt_scores, single.requests[0].prompt_scores
        )
