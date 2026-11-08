# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check TokenGate inference from model loading through answer generation.

Small Qwen3 and text-only Qwen3.5 models run through vLLM's public LLM API with
token filtering enabled. Their next-token probabilities are compared
with an independent calculation for the same input history. Tests also check
that restarting, cancelling, or reusing requests does not corrupt model state,
and that models without TokenGate retain their ordinary behavior.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from tests.token_gate.assets import create_artifacts
from tests.token_gate.reference import reference_logits
from tools.export_token_gate import export_checkpoint
from vllm import SamplingParams
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind


def snapshot_recurrent_state(worker):
    """Read the active request's actual convolution and recurrent cache blocks."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    runner = worker.model_runner
    history = runner.model_state.history
    request = next(iter(history.requests.values()))
    states = {}
    for group_index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if not isinstance(spec, MambaSpec):
            continue
        block = int(runner.block_tables.block_tables[group_index].gpu[request.slot, 0])
        for name in group.layer_names:
            layer = runner.model.get_submodule(name)
            states[name] = tuple(
                state[block].detach().cpu().clone() for state in layer.kv_cache
            )
    return states


@pytest.mark.parametrize("hybrid,gate_layer", [(False, 0), (True, 1), (True, 3)])
@pytest.mark.parametrize("backend", [None, "TRITON_ATTN"])
@pytest.mark.parametrize("prefill_backend", ["triton", "tiled"])
def test_public_generation_matches_dense_reference(
    vllm_runner, tmp_path, monkeypatch, backend, hybrid, gate_layer, prefill_backend
):
    # Advance exactly one engine step at a time so reset/abort precedes completion.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    artifacts = create_artifacts(
        tmp_path, hybrid=hybrid, gate_layer=gate_layer, head_dim=256 if hybrid else 64
    )
    output = tmp_path / "export"
    export_checkpoint(**artifacts, output_path=output)
    # The independent reference loads gate tensors separately from HF.
    reference = (
        AutoModelForCausalLM.from_pretrained(
            output, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    gate = torch.load(artifacts["gate_path"], weights_only=True)
    prompts = [
        list(range(4, 28)),
        list(range(8, 32)),
        list(range(4, 28)),
        list(range(4, 28)),
    ]
    # The final case retains only two rows, shorter than the convolution window.
    spans = [(2, 20), (2, 20), (2, 20), (0, 23)]
    budgets = [1, 10, 18, 1]
    reference.cpu()
    torch.accelerator.empty_cache()
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=3,
            ignore_eos=True,
            logprobs=128,
            extra_args={
                "token_gate_version": 1,
                "token_gate_span_start": start,
                "token_gate_span_end": end,
                "token_gate_keep_tokens": keep,
            },
        )
        for (start, end), keep in zip(spans, budgets)
    ]
    with vllm_runner(
        str(output),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        max_logprobs=128,
        attention_config={
            "backend": backend,
            "token_gate_prefill_backend": prefill_backend,
        },
        kv_cache_memory_bytes=64 * 1024 * 1024,
    ) as runner:
        for invalid in (
            SamplingParams(
                extra_args={**params[0].extra_args, "token_gate_keep_tokens": True}
            ),
            SamplingParams(
                extra_args=params[0].extra_args, prompt_logprob_token_ids=[4]
            ),
        ):
            with pytest.raises(ValueError):
                runner.llm.generate({"prompt_token_ids": prompts[0]}, invalid)
        with pytest.raises(ValueError, match="unchanged"):
            runner.llm.generate(
                {"prompt_token_ids": prompts[0]},
                params[0],
                tokenization_kwargs={"truncate_prompt_tokens": 23},
            )
        results = runner.llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts], params, use_tqdm=False
        )
        # A second call exercises row reuse after all previous requests finish.
        repeated = runner.llm.generate(
            {"prompt_token_ids": prompts[0]}, params[0], use_tqdm=False
        )
        assert repeated[0].outputs[0].token_ids == results[0].outputs[0].token_ids
        # Force replay after generated history exists, through the same public
        # reset API used by vLLM's cache-reset tests (APC remains disabled).
        engine = runner.llm.llm_engine
        replay_params = params[1].clone()
        replay_params.output_kind = RequestOutputKind.CUMULATIVE
        engine.add_request("replay", {"prompt_token_ids": prompts[1]}, replay_params)
        partial = engine.step()
        prefill_states = (
            runner.llm.collective_rpc(snapshot_recurrent_state)[0] if hybrid else None
        )
        partial = engine.step()
        if hybrid:
            assert prefill_states is not None
            decode_states = runner.llm.collective_rpc(snapshot_recurrent_state)[0]
            assert len(decode_states) == 6
            assert decode_states.keys() == prefill_states.keys()
            for name, states in decode_states.items():
                assert len(states) == 2
                assert states[1].dtype == torch.float32
                for before, after in zip(prefill_states[name], states):
                    assert torch.isfinite(after).all()
                    assert before.shape == after.shape
                    assert not torch.equal(before, after), name
        assert len(partial[0].outputs[0].token_ids) == 2
        assert not partial[0].finished
        assert runner.llm.reset_prefix_cache(reset_running_requests=True)
        completed: list[RequestOutput] = []
        while engine.has_unfinished_requests():
            completed.extend(result for result in engine.step() if result.finished)
        assert len(completed) == 1
        replayed = completed[0]
        assert replayed.outputs[0].token_ids == results[1].outputs[0].token_ids
        engine.add_request("abort", {"prompt_token_ids": prompts[1]}, replay_params)
        assert not engine.step()[0].finished
        engine.abort_request(["abort"])
        assert not engine.has_unfinished_requests()
        after_abort = runner.llm.generate(
            {"prompt_token_ids": prompts[0]}, params[0], use_tqdm=False
        )[0]
        assert after_abort.outputs[0].token_ids == results[0].outputs[0].token_ids
        replay_logprobs = replayed.outputs[0].logprobs[-1]
        regular_logprobs = results[1].outputs[0].logprobs[-1]
        torch.testing.assert_close(
            torch.tensor([replay_logprobs[i].logprob for i in range(128)]),
            torch.tensor([regular_logprobs[i].logprob for i in range(128)]),
            atol=2e-2,
            rtol=2e-2,
        )

    # Compare distributions for the same history. BF16 can break an exact HF
    # argmax tie differently, so independent greedy rollouts are not an oracle
    # for later steps. Keep the original logprob tolerance for every vocabulary row.
    reference.cuda()
    for result, prompt, span, keep in zip(results, prompts, spans, budgets):
        assert result.prompt_token_ids == prompt
        history = list(prompt)
        completion = result.outputs[0]
        for token, actual in zip(completion.token_ids, completion.logprobs):
            logits, _, _ = reference_logits(
                reference, gate, history, len(prompt), (*span, keep)
            )
            wanted = logits.log_softmax(-1).cpu()
            values = torch.tensor([actual[index].logprob for index in range(128)])
            torch.testing.assert_close(values, wanted, atol=2e-2, rtol=2e-2)
            assert token == int(values.argmax())
            history.append(token)


@pytest.mark.parametrize("hybrid", [False, True])
def test_ordinary_model_keeps_default_features_and_has_no_gate_state(
    vllm_runner, tmp_path, monkeypatch, hybrid
):
    # This local-only worker inspection sends a callable through collective_rpc.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    artifacts = create_artifacts(tmp_path, hybrid=hybrid, gate_layer=1 if hybrid else 0)
    output = tmp_path / "baseline"
    export_checkpoint(**artifacts, output_path=output, include_gate=False)
    prompt = list(range(4, 28))
    reference = (
        AutoModelForCausalLM.from_pretrained(
            output, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        wanted = (
            reference(torch.tensor([prompt], device="cuda"))
            .logits[0, -1]
            .float()
            .log_softmax(-1)
            .cpu()
        )
    del reference
    torch.accelerator.empty_cache()
    with vllm_runner(
        str(output),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=True,
        max_logprobs=128,
        attention_config={"backend": "TRITON_ATTN"},
        kv_cache_memory_bytes=64 * 1024 * 1024,
    ) as runner:

        def inspect_state(worker):
            state = worker.model_runner.model_state
            model = worker.model_runner.model
            return (
                type(model).__name__,
                type(state).__name__,
                hasattr(state, "prompt_channels"),
            )

        assert runner.llm.collective_rpc(inspect_state) == [
            (
                "Qwen3_5ForCausalLM" if hybrid else "Qwen3ForCausalLM",
                "MambaHybridModelState" if hybrid else "DefaultModelState",
                False,
            )
        ]
        result = runner.llm.generate(
            {"prompt_token_ids": prompt},
            SamplingParams(temperature=0, max_tokens=1, logprobs=128),
            use_tqdm=False,
        )[0]
        actual = result.outputs[0].logprobs[0]
        torch.testing.assert_close(
            torch.tensor([actual[i].logprob for i in range(128)]),
            wanted,
            atol=2e-2,
            rtol=2e-2,
        )
