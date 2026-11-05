# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that separately saved TokenGate training weights are packaged correctly.

The exporter combines a base model, its learned token filter, and adapter
updates into a directory that vLLM can load. These tests verify the resulting
weights and model structure, check the records identifying the training files,
and require invalid or incomplete inputs to be rejected. They also cover
exports without the filter for comparison and span tokenization.
"""

import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)

from tests.token_gate.assets import create_artifacts
from tools.export_token_gate import artifact_sha256, export_checkpoint
from vllm.config.token_gate import TokenGateConfig
from vllm.renderers.token_gate import tokenize_span


@pytest.fixture
def artifacts(tmp_path):
    return create_artifacts(tmp_path)


@pytest.mark.parametrize("gate_layer,indices", [(1, (3, 7)), (3, (7,))])
@pytest.mark.parametrize("head_dim", [64, 256])
def test_hybrid_export_preserves_topology_and_native_query_gate(
    tmp_path, gate_layer, indices, head_dim
):
    artifacts = create_artifacts(
        tmp_path, hybrid=True, gate_layer=gate_layer, head_dim=head_dim
    )
    output = tmp_path / "export"
    export_checkpoint(**artifacts, output_path=output)
    config = AutoConfig.from_pretrained(output)
    assert config.architectures == ["Qwen3_5ForCausalLM"]
    assert config.tie_word_embeddings
    assert config.mamba_ssm_dtype == "float32"
    assert config.rope_parameters["partial_rotary_factor"] == 0.25
    assert TokenGateConfig.from_hf_config(config).attention_layer_indices == indices
    state = load_file(output / "model.safetensors")
    assert state[f"model.layers.{gate_layer}.token_gate.gate_scale"].shape == (
        len(indices),
        2,
    )
    base = load_file(artifacts["base_path"] / "model.safetensors")
    adapter = torch.load(artifacts["adapter_path"], weights_only=True)
    target = "model.layers.3.self_attn.q_proj"
    expected = (
        base[target + ".weight"].float()
        + 2 * (adapter[target + ".lora_A"] @ adapter[target + ".lora_B"]).T
    ).bfloat16()
    assert expected.shape == (4 * head_dim, 128)
    assert torch.equal(state[target + ".weight"], expected)
    assert torch.equal(
        state["model.layers.0.linear_attn.in_proj_qkv.weight"],
        base["model.layers.0.linear_attn.in_proj_qkv.weight"],
    )
    evidence = json.loads((output / "token_gate_export_evidence.json").read_text())
    assert evidence["format_version"] == 1


@pytest.mark.parametrize("declared", [[4, 7], [7, 3], [3], [True, 7], None])
def test_hybrid_rejects_incorrect_gain_layer_map(tmp_path, declared):
    artifacts = create_artifacts(tmp_path, hybrid=True, gate_layer=1)
    output = tmp_path / "export"
    export_checkpoint(**artifacts, output_path=output)
    config = AutoConfig.from_pretrained(output)
    config.token_gate["attention_layer_indices"] = declared
    with pytest.raises(ValueError, match="attention_layer_indices"):
        TokenGateConfig.from_hf_config(config)


def test_hybrid_export_extracts_complete_language_model_from_wrapper(tmp_path):
    artifacts = create_artifacts(tmp_path, hybrid=True, gate_layer=1)
    text = AutoModelForCausalLM.from_pretrained(artifacts["base_path"])
    wrapper = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=text.config.to_dict(),
            tie_word_embeddings=True,
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 128,
            },
        )
    ).bfloat16()
    wrapper.model.language_model.load_state_dict(text.model.state_dict())
    wrapper.save_pretrained(artifacts["base_path"])
    path = artifacts["evidence_path"]
    evidence = json.loads(path.read_text())
    evidence["base_config_sha256"] = artifact_sha256(
        artifacts["base_path"] / "config.json"
    )
    evidence["base_weight_sha256"] = {
        entry.name: artifact_sha256(entry)
        for entry in artifacts["base_path"].glob("*.safetensors")
    }
    path.write_text(json.dumps(evidence))
    output = tmp_path / "export"
    export_checkpoint(**artifacts, output_path=output)
    config = AutoConfig.from_pretrained(output)
    assert config.model_type == "qwen3_5_text"
    assert config.architectures == ["Qwen3_5ForCausalLM"]
    state = load_file(output / "model.safetensors")
    assert not any("visual" in name or "language_model" in name for name in state)
    for name, value in text.model.state_dict().items():
        if name != "layers.3.self_attn.q_proj.weight":
            assert torch.equal(state["model." + name], value.bfloat16()), name


def test_export_preserves_fp32_gate_and_bf16_backbone(artifacts, tmp_path):
    output = tmp_path / "export"
    export_checkpoint(**artifacts, output_path=output)
    state = load_file(output / "model.safetensors")
    assert state["model.layers.0.token_gate.input_gate.weight"].dtype == torch.float32
    assert state["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16
    config = AutoConfig.from_pretrained(output)
    assert config.architectures == ["Qwen3ForCausalLM"]
    assert config.token_gate["attention_semantics"] == "tokengate_code_v1"
    assert config.token_gate["provenance"]["step"] == 7
    base = load_file(artifacts["base_path"] / "model.safetensors")
    adapter = torch.load(artifacts["adapter_path"], weights_only=True)
    target = "model.layers.0.self_attn.q_proj"
    expected = (
        base[target + ".weight"].float()
        + 2 * (adapter[target + ".lora_A"] @ adapter[target + ".lora_B"]).T
    )
    assert torch.equal(state[target + ".weight"], expected.bfloat16())


def test_matched_baseline_uses_existing_qwen3_without_gate(artifacts, tmp_path):
    output = tmp_path / "baseline"
    export_checkpoint(**artifacts, output_path=output, include_gate=False)
    config = AutoConfig.from_pretrained(output)
    state = load_file(output / "model.safetensors")
    assert config.architectures == ["Qwen3ForCausalLM"]
    assert not hasattr(config, "token_gate")
    assert not any(".token_gate." in key for key in state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gate_sha256", "wrong"),
        ("adapter_sha256", "wrong"),
        ("params_sha256", "wrong"),
        ("tokenizer_sha256", "wrong"),
        ("adapter_step", 8),
        ("selection_mode", "threshold"),
        ("random_topk", True),
    ],
)
def test_export_rejects_unverified_or_unsupported_artifacts(
    artifacts, tmp_path, field, value
):
    path = artifacts["evidence_path"]
    evidence = json.loads(path.read_text())
    evidence[field] = value
    path.write_text(json.dumps(evidence))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        export_checkpoint(**artifacts, output_path=output)
    assert not output.exists()


@pytest.mark.parametrize(
    "text,start,end,token_ids,token_span",
    [
        ("word4 word5 word6 word7 word8", 6, 23, [4, 5, 6, 7, 8], (1, 4)),
        ("word4 = word5 + word6\nword7", 8, 21, [4, 3, 5, 3, 6, 7], (2, 5)),
        ("word4: word5 word6 word7\nword8", 7, 24, [4, 3, 5, 6, 7, 8], (2, 5)),
    ],
    ids=["plain-text", "code", "log"],
)
def test_span_helper_aligns_full_tokenization(
    artifacts, text, start, end, token_ids, token_span
):
    tokenizer = AutoTokenizer.from_pretrained(artifacts["base_path"])
    prompt, extra = tokenize_span(
        tokenizer, text, span_start=start, span_end=end, keep_tokens=1
    )
    assert prompt["prompt_token_ids"] == token_ids
    assert extra == {
        "token_gate_version": 1,
        "token_gate_span_start": token_span[0],
        "token_gate_span_end": token_span[1],
        "token_gate_keep_tokens": 1,
    }
    # A token crossing either character boundary remains outside the span.
    _, crossing = tokenize_span(tokenizer, text, start + 2, end - 2, 1)
    assert crossing["token_gate_span_start"] == token_span[0] + 1
    assert crossing["token_gate_span_end"] == token_span[1] - 1


@pytest.mark.parametrize(
    "case,message",
    [
        ("missing_gate", "Missing or incompatible gate tensor"),
        ("unknown_gate", "Unconsumed gate tensors"),
        ("nonfinite_adapter", "Nonfinite or invalid adapter tensor"),
        ("partial_adapter", "Incomplete adapter tensors"),
    ],
)
def test_hashed_artifacts_still_require_complete_valid_tensors(
    artifacts, tmp_path, case, message
):
    kind = "gate" if case.endswith("gate") else "adapter"
    path = artifacts[kind + "_path"]
    artifact = torch.load(path, weights_only=True)
    if case == "missing_gate":
        del artifact["state_dict"]["gate_scale"]
    elif case == "unknown_gate":
        artifact["state_dict"]["unknown.weight"] = torch.ones(1)
    elif case == "nonfinite_adapter":
        next(iter(artifact.values())).fill_(float("nan"))
    else:
        del artifact["model.layers.0.self_attn.q_proj.lora_B"]
    torch.save(artifact, path)
    evidence_path = artifacts["evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence[kind + "_sha256"] = artifact_sha256(path)
    evidence_path.write_text(json.dumps(evidence))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=message):
        export_checkpoint(**artifacts, output_path=output)
    assert not output.exists()


def rehash(artifacts, kind):
    path = artifacts["evidence_path"]
    evidence = json.loads(path.read_text())
    evidence[kind + "_sha256"] = artifact_sha256(artifacts[kind + "_path"])
    path.write_text(json.dumps(evidence))


@pytest.mark.parametrize(
    "field,value",
    [
        ("tanh_gate", True),
        ("qk_norm_scale", True),
        ("softplus_sss", True),
        ("sss_frozen_gate_scale", True),
        ("learned_agg", True),
        ("gate_cond", "boundary"),
        ("use_boundary_cond", True),
        ("n_input_layers", 2),
        ("score_source", "keys"),
    ],
)
def test_export_rejects_removed_research_semantics(artifacts, tmp_path, field, value):
    gate = torch.load(artifacts["gate_path"], weights_only=True)
    gate[field] = value
    torch.save(gate, artifacts["gate_path"])
    rehash(artifacts, "gate")
    with pytest.raises(ValueError, match=field):
        export_checkpoint(**artifacts, output_path=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("kind", ["lora", "delta"])
@pytest.mark.parametrize("missing", ["target", "layer", "default_targets"])
def test_export_rejects_entire_missing_adapter_targets(
    artifacts, tmp_path, kind, missing
):
    gate = torch.load(artifacts["gate_path"], weights_only=True)
    metadata = gate["metadata"]
    metadata["adapter_type"] = kind
    if missing == "layer":
        metadata["lora_layer_ids"] = [0, 1]
    elif missing == "target":
        metadata["lora_targets"] = ["q_proj", "k_proj"]
    else:
        metadata["lora_targets"] = None
    if kind == "delta":
        base = load_file(artifacts["base_path"] / "model.safetensors")
        targets = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        if missing != "layer":
            targets = targets[:1]
        state = {
            f"model.layers.0.{target}.delta": torch.zeros_like(
                base[f"model.layers.0.{target}.weight"]
            )
            for target in targets
        }
        torch.save(state, artifacts["adapter_path"])
        rehash(artifacts, "adapter")
    torch.save(gate, artifacts["gate_path"])
    rehash(artifacts, "gate")
    with pytest.raises(ValueError, match="adapter"):
        export_checkpoint(**artifacts, output_path=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -float("inf"), 1e40])
def test_export_rejects_nonfinite_scaling_or_merged_weights(artifacts, tmp_path, alpha):
    gate = torch.load(artifacts["gate_path"], weights_only=True)
    gate["metadata"]["lora_alpha"] = alpha
    torch.save(gate, artifacts["gate_path"])
    rehash(artifacts, "gate")
    with pytest.raises(ValueError, match="[Nn]onfinite|scaling"):
        export_checkpoint(**artifacts, output_path=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_export_rejects_retrospective_sliding_attention(artifacts, tmp_path):
    config = AutoConfig.from_pretrained(artifacts["base_path"])
    config.rswa_window = 8
    config.save_pretrained(artifacts["base_path"])
    evidence_path = artifacts["evidence_path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["base_config_sha256"] = artifact_sha256(
        artifacts["base_path"] / "config.json"
    )
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="full attention"):
        export_checkpoint(**artifacts, output_path=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("kind", ["lora", "delta"])
def test_export_resolves_complete_adapter_defaults(artifacts, tmp_path, kind):
    gate = torch.load(artifacts["gate_path"], weights_only=True)
    gate["metadata"].update(adapter_type=kind, lora_targets=None)
    del gate["metadata"]["lora_layer_ids"]  # The legacy default is the gate layer.
    base = load_file(artifacts["base_path"] / "model.safetensors")
    targets = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]
    if kind == "delta":
        targets += ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    state = {}
    expected = {}
    for target in targets:
        name = "model.layers.0." + target
        weight = base[name + ".weight"]
        if kind == "lora":
            first = torch.full((weight.shape[1], 4), 0.01)
            second = torch.full((4, weight.shape[0]), 0.01)
            state[name + ".lora_A"], state[name + ".lora_B"] = first, second
            delta = 2 * (first @ second).T
        else:
            delta = torch.full(weight.shape, 0.01)
            state[name + ".delta"] = delta
        expected[name + ".weight"] = (weight.float() + delta).bfloat16()
    torch.save(state, artifacts["adapter_path"])
    torch.save(gate, artifacts["gate_path"])
    rehash(artifacts, "adapter")
    rehash(artifacts, "gate")
    output = tmp_path / "defaults"
    export_checkpoint(**artifacts, output_path=output)
    actual = load_file(output / "model.safetensors")
    for name, wanted in expected.items():
        assert torch.equal(actual[name], wanted)
    if kind == "delta":
        # Finite FP32 values may overflow only when cast to BF16 for saving.
        state["model.layers.0.self_attn.q_proj.delta"].fill_(
            torch.finfo(torch.float32).max
        )
        torch.save(state, artifacts["adapter_path"])
        rehash(artifacts, "adapter")
        with pytest.raises(ValueError, match="Nonfinite merged"):
            export_checkpoint(**artifacts, output_path=tmp_path / "overflow")
        assert not (tmp_path / "overflow").exists()
