# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build tiny saved models so TokenGate tests can run without trained checkpoints.

These helpers create randomly initialized Qwen3 or text-only Qwen3.5 models,
a tokenizer, a token-filtering gate, and adapter updates to model weights.
They write the same files and training-record metadata that the exporter
expects, allowing tests to exercise real model loading and inference APIs.
"""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    TokenizersBackend,
)

from tools.export_token_gate import (
    artifact_sha256,
    export_checkpoint,
    tokenizer_sha256,
)


def create_artifacts(
    root: Path,
    *,
    hybrid: bool = False,
    gate_layer: int = 0,
    head_dim: int = 64,
) -> dict[str, Path]:
    torch.manual_seed(23)
    base = root / "base"
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    if hybrid:
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=head_dim,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention"] * 3
            + ["full_attention"]
            + ["linear_attention"] * 3
            + ["full_attention"],
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
            mamba_ssm_dtype="float32",
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
        )
    model_class = Qwen3_5ForCausalLM if hybrid else Qwen3ForCausalLM
    model = model_class(config).to(torch.bfloat16)
    full_attention_layers = [
        index
        for index, kind in enumerate(config.layer_types)
        if kind == "full_attention"
    ]
    upper_layers = [index for index in full_attention_layers if index > gate_layer]
    adapter_layer = full_attention_layers[0]
    base_name = "tiny-qwen35-test" if hybrid else "tiny-qwen3-test"
    model.save_pretrained(base)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"word{i}": i for i in range(4, 128)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    TokenizersBackend(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    ).save_pretrained(base)
    metadata = {
        "base_model": base_name,
        "gate_layer": gate_layer,
        "step": 7,
        "adapter_type": "lora",
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_layer_ids": [adapter_layer],
        "lora_targets": ["q_proj"],
    }
    gate = {
        "d_model": 128,
        "n_heads": 2,
        "n_layers_bias": len(upper_layers),
        "selection_mode": "topk",
        "random_topk": False,
        "metadata": metadata,
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
        "state_dict": {
            "gate_branch.gate_proj.weight": torch.randn(256, 128) * 0.03,
            "gate_branch.up_proj.weight": torch.randn(256, 128) * 0.03,
            "gate_branch.down_proj.weight": torch.randn(1, 256) * 0.03,
            "gate_scale": torch.tensor([[0.3, 0.9], [0.7, 0.2]])[: len(upper_layers)],
            "attn_scale": torch.tensor([[0.13, 1.77], [0.25, 0.4]])[
                : len(upper_layers)
            ],
            "b_norm.weight": torch.ones(1),
            "agg_weights": torch.full((2,), 0.5),
        },
    }
    gate_path, adapter_path = root / "gate.pt", root / "adapter.pt"
    torch.save(gate, gate_path)
    torch.save(
        {
            f"model.layers.{adapter_layer}.self_attn.q_proj.lora_A": torch.randn(128, 4)
            * 0.01,
            f"model.layers.{adapter_layer}.self_attn.q_proj.lora_B": torch.randn(
                4, model.model.layers[adapter_layer].self_attn.q_proj.out_features
            )
            * 0.01,
        },
        adapter_path,
    )
    params_path = root / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "base_model": base_name,
                "gate_layer": gate_layer,
                "gate_selection": "topk",
                "random_topk": False,
            }
        )
    )
    evidence_path = root / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "selection_mode": "topk",
                "random_topk": False,
                "gate_step": 7,
                "adapter_step": 7,
                "base_model": base_name,
                "reference_revision": "f78ec10aaa77",
                "run_id": "synthetic-test-fixture",
                "gate_sha256": artifact_sha256(gate_path),
                "adapter_sha256": artifact_sha256(adapter_path),
                "params_sha256": artifact_sha256(params_path),
                "tokenizer_sha256": tokenizer_sha256(base),
                "base_config_sha256": artifact_sha256(base / "config.json"),
                "base_weight_sha256": {
                    path.name: artifact_sha256(path)
                    for path in base.glob("*.safetensors")
                },
            }
        )
    )
    return dict(
        base_path=base,
        gate_path=gate_path,
        adapter_path=adapter_path,
        params_path=params_path,
        evidence_path=evidence_path,
    )


def create_export(
    root: Path,
    include_gate: bool = True,
    *,
    hybrid: bool = False,
    gate_layer: int = 0,
    head_dim: int = 64,
) -> Path:
    paths = create_artifacts(
        root, hybrid=hybrid, gate_layer=gate_layer, head_dim=head_dim
    )
    output = root / "export"
    export_checkpoint(**paths, output_path=output, include_gate=include_gate)
    return output
