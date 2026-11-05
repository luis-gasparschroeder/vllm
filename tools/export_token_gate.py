# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a trained TokenGate model for inference in vLLM.

Training saves the gate (the learned token filter) and adapter (updates to the
base model's weights) separately. This command checks that those files belong
together, merges the adapter into the base model, and packages the gate, model,
and tokenizer in one directory that vLLM can load. --without-token-gate creates
a comparison model with the same merged weights but no token filtering.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.config.token_gate import TokenGateConfig, attention_layer_indices

TOKEN_GATE_FORMAT_VERSION = 2
EXPORT_EVIDENCE_FORMAT_VERSION = 1
GATE_WEIGHT_NAMES = {
    "gate_branch.gate_proj.weight": "input_gate.weight",
    "gate_branch.up_proj.weight": "input_up.weight",
    "gate_branch.down_proj.weight": "output.weight",
    "gate_scale": "gate_scale",
    "attn_scale": "attn_scale",
    "b_norm.weight": "normalization_weight",
}


def artifact_sha256(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_sha256(directory: str | Path) -> str:
    """Hash recognized tokenizer filenames and contents, ignoring directory location.

    Args:
        directory: Local directory containing tokenizer files.

    Returns:
        A hexadecimal SHA-256 digest of the filenames and their content hashes.

    Raises:
        ValueError: No recognized tokenizer files are present.

    """
    root = Path(directory)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    )
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = root / name
        if path.is_file():
            found = True
            digest.update(name.encode() + b"\0")
            digest.update(bytes.fromhex(artifact_sha256(path)))
    if not found:
        raise ValueError("No tokenizer files found")
    return digest.hexdigest()


def validate_export_evidence(
    gate: dict[str, Any],
    evidence: dict[str, Any],
    *,
    gate_path: Path,
    adapter_path: Path,
    params_path: Path,
    base_path: Path,
) -> dict[str, Any]:
    """Verify that the base model, gate, and adapter match the export manifest.

    Checks file hashes, checkpoint identities, training steps, and the supported
    selection policy before the model is loaded or modified.

    Args:
        gate: Loaded gate checkpoint, including its training metadata.
        evidence: Manifest recording expected hashes and training identities.
        gate_path: Gate checkpoint file.
        adapter_path: Adapter checkpoint file.
        params_path: JSON file containing the training parameters.
        base_path: Local base-model directory, including weights and tokenizer.

    Returns:
        Verified file hashes and training identities for the exported config.

    Raises:
        ValueError: Required evidence is missing, inconsistent, or unsupported.

    """
    if (
        type(evidence.get("format_version")) is not int
        or evidence["format_version"] != EXPORT_EVIDENCE_FORMAT_VERSION
    ):
        raise ValueError("Unsupported evidence format")
    hashes = {
        "gate_sha256": artifact_sha256(gate_path),
        "adapter_sha256": artifact_sha256(adapter_path),
        "params_sha256": artifact_sha256(params_path),
        "tokenizer_sha256": tokenizer_sha256(base_path),
    }
    for name, expected_hash in hashes.items():
        if evidence.get(name) != expected_hash:
            raise ValueError(f"Evidence mismatch: {name}")
    params = json.loads(params_path.read_text())
    metadata = gate.get("metadata", {})
    for key, expected in (("selection_mode", "topk"), ("random_topk", False)):
        if type(evidence.get(key)) is not type(expected) or evidence[key] != expected:
            raise ValueError(f"Export requires verified {key}={expected!r}")
        if key in gate and (
            type(gate[key]) is not type(expected) or gate[key] != expected
        ):
            raise ValueError(f"Gate policy conflicts with evidence: {key}")
        params_key = "gate_selection" if key == "selection_mode" else key
        if params_key not in params or type(params[params_key]) is not type(expected):
            raise ValueError(f"Hashed training params must explicitly record {key}")
        if params[params_key] != expected:
            raise ValueError(f"Training policy is unsupported: {key}")
    step = metadata.get("step")
    if (
        type(step) is not int
        or type(evidence.get("gate_step")) is not int
        or type(evidence.get("adapter_step")) is not int
        or step != evidence.get("gate_step")
        or (step != evidence.get("adapter_step"))
    ):
        raise ValueError("Gate and adapter must have the same verified training step")
    for key in ("base_model", "reference_revision", "run_id"):
        if not isinstance(evidence.get(key), str) or not evidence[key]:
            raise ValueError(f"Evidence must identify {key}")
    if metadata.get("base_model") != evidence["base_model"]:
        raise ValueError("Gate base model conflicts with evidence")
    if params.get("base_model") != evidence["base_model"]:
        raise ValueError("Training base model conflicts with evidence")
    if params.get("gate_layer") != metadata.get("gate_layer"):
        raise ValueError("Training gate layer conflicts with checkpoint")
    if "run_id" in metadata and metadata["run_id"] != evidence["run_id"]:
        raise ValueError("Checkpoint run identity conflicts with evidence")
    if "run_id" in params and params["run_id"] != evidence["run_id"]:
        raise ValueError("Training run identity conflicts with evidence")
    if (
        "params_sha256" in metadata
        and metadata["params_sha256"] != hashes["params_sha256"]
    ):
        raise ValueError("Checkpoint training params conflict with evidence")
    # A local snapshot must itself be pinned; filenames alone are not an identity.
    config_hash = artifact_sha256(base_path / "config.json")
    if evidence.get("base_config_sha256") != config_hash:
        raise ValueError("Base configuration conflicts with evidence")
    weights = sorted(base_path.glob("*.safetensors"))
    weight_hashes = {path.name: artifact_sha256(path) for path in weights}
    if not weights or evidence.get("base_weight_sha256") != weight_hashes:
        raise ValueError("Base weights conflict with evidence")
    return {
        **hashes,
        "base_model": evidence["base_model"],
        "base_config_sha256": config_hash,
        "base_weight_sha256": weight_hashes,
        "reference_revision": evidence["reference_revision"],
        "run_id": evidence["run_id"],
        "step": step,
        "selection_policy": "hashed_training_params_and_artifact_manifest",
    }


def export_gate_weights(
    gate: dict[str, Any], config: Any
) -> tuple[int, dict[str, torch.Tensor]]:
    """Validate gate tensors and map them to vLLM parameter names.

    Args:
        gate: Loaded checkpoint containing gate settings, metadata, and tensors.
        config: Hugging Face model configuration defining layer types and sizes.

    Returns:
        The zero-based gate layer and a mapping from model parameter names to
        contiguous FP32 tensors.

    Raises:
        ValueError: The gate architecture, dimensions, or tensors are unsupported
            or inconsistent with the base model.

    """
    constants = {
        "head_agg": "avg",
        "per_layer_gate": True,
        "learnable_sss_scale": True,
        "normed_gate": True,
        "use_attn_scale": True,
        "scaled_softmax_scale": True,
        "softplus_gate_scale": True,
        "shared_readout": True,
        "absorbed_readout": True,
        "readout_residual": False,
        "ste_factor": 0.0,
        "gate_branch_mode": "mlp",
    }
    for name, expected in constants.items():
        if gate.get(name) != expected:
            raise ValueError(f"Unsupported or missing gate architecture field: {name}")
    # The research loader assigns these defaults to older frozen-architecture
    # checkpoints. Explicit declarations of removed variants must still fail.
    defaults = {
        "n_input_layers": 1,
        "score_source": "hidden",
        "learned_agg": False,
        "tanh_gate": False,
        "qk_norm_scale": False,
        "softplus_sss": False,
        "sss_frozen_gate_scale": False,
        "use_boundary_cond": False,
    }
    for name, expected in defaults.items():
        value = gate.get(name, expected)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Unsupported gate architecture field: {name}")
    if gate.get("gate_cond", "none") not in (None, "none"):
        raise ValueError("Unsupported gate architecture field: gate_cond")
    layer = gate["metadata"].get("gate_layer")
    upper = len(attention_layer_indices(config, layer))
    heads = config.num_attention_heads
    if (
        gate.get("d_model") != config.hidden_size
        or gate.get("n_heads") != heads
        or gate.get("n_layers_bias") != upper
    ):
        raise ValueError("Gate dimensions disagree with the base model")
    state = gate["state_dict"]
    unexpected = set(state) - set(GATE_WEIGHT_NAMES) - {"agg_weights"}
    if unexpected:
        raise ValueError(f"Unconsumed gate tensors: {sorted(unexpected)}")
    if "agg_weights" in state and not torch.equal(
        state["agg_weights"].float(), torch.full((heads,), 1.0 / heads)
    ):
        raise ValueError("TokenGate requires uniform head aggregation")
    shapes = {
        "gate_branch.gate_proj.weight": (config.intermediate_size, config.hidden_size),
        "gate_branch.up_proj.weight": (config.intermediate_size, config.hidden_size),
        "gate_branch.down_proj.weight": (1, config.intermediate_size),
        "gate_scale": (upper, heads),
        "attn_scale": (upper, heads),
        "b_norm.weight": (1,),
    }
    result = {}
    for source, target in GATE_WEIGHT_NAMES.items():
        value = state.get(source)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[source]:
            raise ValueError(f"Missing or incompatible gate tensor: {source}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite gate tensor: {source}")
        result[f"model.layers.{layer}.token_gate.{target}"] = value.float().contiguous()
    return layer, result


def _adapter_targets(
    model: nn.Module, metadata: dict[str, Any]
) -> dict[str, nn.Linear]:
    """Resolve the model's linear layers targeted by the trained adapter.

    Args:
        model: Hugging Face model whose parameters will receive the adapter.
        metadata: Checkpoint metadata declaring the adapter type and targets.

    Returns:
        Full module names mapped to linear layers.

    Raises:
        ValueError: Layer or target declarations are invalid, refer to missing
            or non-linear modules, or select unsupported hybrid layers.

    """
    layers = metadata.get("lora_layer_ids")
    if model.config.model_type == "qwen3_5_text" and layers is None:
        raise ValueError("Hybrid adapters require explicit lora_layer_ids")
    if layers is None:
        layers = [metadata.get("gate_layer")]
    num_layers = len(model.get_submodule("model.layers"))
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            type(layer) is not int or not 0 <= layer < num_layers for layer in layers
        )
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("Invalid adapter layer declaration")
    if model.config.model_type == "qwen3_5_text" and any(
        model.config.layer_types[layer] != "full_attention" for layer in layers
    ):
        raise ValueError("Hybrid adapters support full-attention layers only")
    targets = metadata.get("lora_targets")
    if metadata["adapter_type"] == "delta":
        # The trainer's delta adapter always covers attention and the whole MLP.
        targets = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    elif targets is None:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise ValueError("Invalid adapter target declaration")
    targets = [target if "." in target else f"self_attn.{target}" for target in targets]
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate adapter targets")
    result = {}
    for layer in layers:
        for target in targets:
            name = f"model.layers.{layer}.{target}"
            try:
                linear = model.get_submodule(name)
            except AttributeError as error:
                raise ValueError(f"Unknown adapter target: {name}") from error
            if not isinstance(linear, nn.Linear):
                raise ValueError(f"Adapter target is not a linear layer: {name}")
            result[name] = linear
    return result


def _merged_weight(
    weight: torch.Tensor, delta: torch.Tensor, name: str
) -> torch.Tensor:
    """Add an adapter update and check the result in the destination dtype.

    Returns the merged tensor; raises ValueError with the parameter name if
    conversion to the original weight dtype produces a nonfinite value.

    """
    merged = (weight.float() + delta).to(weight.dtype)
    if not torch.isfinite(merged).all():
        raise ValueError(f"Nonfinite merged adapter tensor: {name}")
    return merged


@torch.no_grad()
def merge_fixed_adapter(
    model: nn.Module, state: dict[str, torch.Tensor], metadata: dict[str, Any]
) -> None:
    """Merge the trained LoRA or delta adapter into the model in place.

    Validates the declared tensor set, then adds each update to its target.
    A later validation error can leave earlier targets already updated.

    Args:
        model: Hugging Face model to modify in place.
        state: Adapter tensors keyed by the training checkpoint's names.
        metadata: Adapter type, target layers, and LoRA rank and scaling.

    Raises:
        ValueError: The adapter type, targets, tensors, or scaling are invalid,
            or merging produces nonfinite weights.

    """
    kind = metadata.get("adapter_type")
    if kind not in ("lora", "delta") or not state:
        raise ValueError("Export requires a matched LoRA or delta adapter")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite or invalid adapter tensor: {name}")
    targets = _adapter_targets(model, metadata)
    expected: set[str] = set()
    for target, linear in targets.items():
        if kind == "lora":
            expected.update((target + ".lora_A", target + ".lora_B"))
        else:
            expected.add(target + ".delta")
            if linear.bias is not None:
                expected.add(target + ".delta_bias")
    if set(state) != expected:
        raise ValueError(
            f"Incomplete adapter tensors: missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    for target, linear in targets.items():
        if kind == "lora":
            first, second = target + ".lora_A", target + ".lora_B"
            rank = metadata.get("lora_rank")
            alpha = metadata.get("lora_alpha")
            if (
                type(rank) is not int
                or rank < 1
                or not isinstance(alpha, (int, float))
                or isinstance(alpha, bool)
                or not math.isfinite(alpha)
                or state[first].shape != (linear.in_features, rank)
                or state[second].shape != (rank, linear.out_features)
            ):
                raise ValueError(f"Incompatible LoRA dimensions/scaling: {target}")
            delta = (state[first].float() @ state[second].float()).T * (alpha / rank)
            linear.weight.copy_(_merged_weight(linear.weight, delta, target))
        else:
            name = target + ".delta"
            if state[name].shape != linear.weight.shape:
                raise ValueError(f"Incompatible delta dimensions: {target}")
            linear.weight.copy_(
                _merged_weight(linear.weight, state[name].float(), name)
            )
            bias_name = target + ".delta_bias"
            if linear.bias is not None:
                if state[bias_name].shape != linear.bias.shape:
                    raise ValueError(f"Incompatible delta bias: {target}")
                linear.bias.copy_(
                    _merged_weight(linear.bias, state[bias_name].float(), bias_name)
                )


def export_checkpoint(
    base_path: Path,
    gate_path: Path,
    adapter_path: Path,
    params_path: Path,
    evidence_path: Path,
    output_path: Path,
    *,
    include_gate: bool = True,
) -> None:
    """Write a model export with merged adapter weights and optional TokenGate.

    Loads local files on CPU and saves the model, tokenizer, and export manifest
    in a new directory. Gate and adapter evidence is checked even when the gate
    is omitted from the output.

    Args:
        base_path: Local Hugging Face base-model and tokenizer directory.
        gate_path: Gate checkpoint containing its tensors and training metadata.
        adapter_path: LoRA or delta adapter checkpoint from the same training step.
        params_path: JSON training parameters referenced by the manifest.
        evidence_path: JSON manifest of expected file hashes and training identities.
        output_path: Destination directory, which must not already exist.
        include_gate: Whether to include gate weights and configuration. False
            exports the same merged adapter weights without token filtering.

    Raises:
        ValueError: The destination exists, evidence or weights are inconsistent,
            or the checkpoint uses an unsupported configuration.

    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output_path.exists():
        raise ValueError("Export destination must not already exist")
    gate = torch.load(gate_path, map_location="cpu", weights_only=True)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    evidence = json.loads(evidence_path.read_text())
    provenance = validate_export_evidence(
        gate,
        evidence,
        gate_path=gate_path,
        adapter_path=adapter_path,
        params_path=params_path,
        base_path=base_path,
    )
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    if loading_info["missing_keys"] or loading_info.get("mismatched_keys"):
        raise ValueError(f"Incomplete base language-model weights: {loading_info}")
    if model.config.tie_word_embeddings and not torch.equal(
        model.get_input_embeddings().weight, model.get_output_embeddings().weight
    ):
        raise ValueError("Base checkpoint contradicts its tied-embedding configuration")
    provenance["export_architecture"] = type(model).__name__
    layer, gate_weights = export_gate_weights(gate, model.config)
    merge_fixed_adapter(model, adapter, gate["metadata"])
    state = model.state_dict()
    if include_gate:
        model.config.token_gate = {
            "format_version": TOKEN_GATE_FORMAT_VERSION,
            "attention_semantics": "tokengate_code_v1",
            "gate_layer": layer,
            "attention_layer_indices": list(
                attention_layer_indices(model.config, layer)
            ),
            "selection_mode": "topk",
            "random_topk": False,
            "decode_phase2_form": True,
            "prune_below_gate": False,
            "precision": "bfloat16",
            "provenance": provenance,
        }
        TokenGateConfig.from_hf_config(model.config)
        state.update(gate_weights)
    elif hasattr(model.config, "token_gate"):
        delattr(model.config, "token_gate")
    model.config.architectures = [type(model).__name__]
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model.save_pretrained(
        output_path,
        state_dict=state,
        safe_serialization=True,
        save_original_format=False,
    )
    tokenizer.save_pretrained(output_path)
    (output_path / "token_gate_export_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "gate", "adapter", "params", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--without-token-gate",
        action="store_true",
        help="Export the same merged adapter as an ordinary Qwen3 baseline.",
    )
    args = parser.parse_args()
    export_checkpoint(
        args.base,
        args.gate,
        args.adapter,
        args.params,
        args.evidence,
        args.output,
        include_gate=not args.without_token_gate,
    )


if __name__ == "__main__":
    main()
