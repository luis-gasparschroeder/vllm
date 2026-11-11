# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that vLLM rejects serving settings this TokenGate implementation cannot use.

Token filtering requires each entire prompt in one execution step.
These tests resolve real engine configurations
for small Qwen3 and text-only Qwen3.5 models on CUDA, checking both valid settings
and rejection of incompatible token limits, chunked prompt processing, prefix
caching, or asynchronous scheduling.
"""

import pytest

from tests.token_gate.assets import create_export
from vllm import EngineArgs


@pytest.fixture(scope="module", params=[False, True])
def checkpoint(tmp_path_factory, request):
    return str(
        create_export(
            tmp_path_factory.mktemp("execution"),
            hybrid=request.param,
            gate_layer=1 if request.param else 0,
        )
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"max_num_batched_tokens": 32}, "max_num_batched_tokens.*smaller"),
        ({"max_num_scheduled_tokens": 32}, "token budgets"),
        ({"long_prefill_token_threshold": 16}, "prefill threshold"),
        ({"enable_prefix_caching": True}, "prefix caching"),
        ({"enable_chunked_prefill": True}, "chunked prefill"),
        ({"async_scheduling": True}, "async scheduling"),
    ],
)
def test_resolved_config_rejects_unsupported_execution(checkpoint, overrides, message):
    options = dict(
        model=checkpoint,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_scheduled_tokens=64,
        max_num_seqs=3,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        long_prefill_token_threshold=0,
    )
    # First prove the unchanged baseline reaches a valid resolved configuration.
    config = EngineArgs(**options).create_engine_config()
    assert config.use_v2_model_runner
    with pytest.raises(ValueError, match=message):
        EngineArgs(**{**options, **overrides}).create_engine_config()
