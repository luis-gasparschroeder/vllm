# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that clients can use TokenGate through the HTTP completion API.

Requests send tokenized prompts, identify the token span eligible for
removal, and specify how many to keep. These CUDA tests check request rejection,
streamed answers, and usage counts based on the original prompt length, using
small Qwen3 and text-only Qwen3.5 models with TokenGate enabled.
"""

import openai
import pytest

from tests.token_gate.assets import create_export
from tests.utils import RemoteOpenAIServer


@pytest.mark.parametrize("hybrid", [False, True])
def test_server_streaming_and_original_prompt_usage(tmp_path, hybrid):
    model = str(create_export(tmp_path, hybrid=hybrid, gate_layer=1 if hybrid else 0))
    prompt = list(range(4, 28))
    gate_args = {
        "token_gate_version": 1,
        "token_gate_span_start": 2,
        "token_gate_span_end": 20,
        "token_gate_keep_tokens": 3,
    }
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "64",
        "--max-num-batched-tokens",
        "64",
        "--max-num-scheduled-tokens",
        "64",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--attention-backend",
        "TRITON_ATTN",
        "--kv-cache-memory-bytes",
        "67108864",
    ]
    with (
        RemoteOpenAIServer(
            model, args, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"}
        ) as server,
        server.get_client() as client,
    ):
        request = dict(model=model, prompt=prompt, max_tokens=3, temperature=0)
        for key in gate_args:
            with pytest.raises(openai.BadRequestError):
                client.completions.create(
                    **request,
                    extra_body={"vllm_xargs": {**gate_args, key: True}},
                )
        for extra in (
            {"truncate_prompt_tokens": 23},
            {"prompt_logprob_token_ids": [4]},
        ):
            with pytest.raises(openai.BadRequestError):
                client.completions.create(
                    **request, extra_body={"vllm_xargs": gate_args, **extra}
                )
        extra = {"vllm_xargs": gate_args, "ignore_eos": True}
        completion = client.completions.create(**request, extra_body=extra)
        assert completion.usage.prompt_tokens == len(prompt)
        assert completion.usage.completion_tokens == 3
        assert completion.choices[0].finish_reason == "length"
        with client.completions.create(
            **request,
            extra_body=extra,
            stream=True,
            stream_options={"include_usage": True},
        ) as stream:
            chunks = list(stream)
        assert (
            "".join(c.choices[0].text for c in chunks if c.choices)
            == completion.choices[0].text
        )
        usage = [chunk.usage for chunk in chunks if chunk.usage]
        assert usage[-1].prompt_tokens == len(prompt)
        assert usage[-1].completion_tokens == 3
        # Stop at the first generated token and verify the next request can reuse state.
        with client.completions.create(
            **request, extra_body=extra, stream=True
        ) as stream:
            next(stream)
        repeated = client.completions.create(**request, extra_body=extra)
        assert repeated.choices[0].text == completion.choices[0].text
