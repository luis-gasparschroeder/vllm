# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that TokenGate allocates and releases the right amount of cache space.

Layers before token filtering need the full prompt's cache; later layers need
only the retained prompt's cache. Tests exercise the real cache manager as
requests start, generate tokens, restart after cache release, and finish.
They also cover the fixed-size recurrent state used by Qwen3.5.
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_layout import CacheSequenceLayout
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def make_request(request_id="doc", prompt_length=33, retained_length=9):
    return Request(
        request_id,
        list(range(prompt_length)),
        SamplingParams(max_tokens=8),
        pooling_params=None,
        cache_sequence_layout=CacheSequenceLayout(prompt_length, retained_length),
    )


def make_manager(num_blocks=20):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    return KVCacheManager(
        KVCacheConfig(
            num_blocks,
            [],
            [
                KVCacheGroupSpec(["early"], spec),
                KVCacheGroupSpec(["upper"], spec, uses_compact_sequence=True),
            ],
        ),
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )


@pytest.mark.parametrize("full_sequence_must_fit", [False, True])
def test_predicted_and_allocated_occupancy_uses_each_history(full_sequence_must_fit):
    manager = make_manager(num_blocks=5)  # Four usable blocks; one is the null block.
    request = make_request()
    allocated = manager.allocate_slots(
        request, 33, full_sequence_must_fit=full_sequence_must_fit
    )
    assert allocated is not None
    assert [len(group) for group in allocated.blocks] == [3, 1]
    assert all(not block.is_null for group in allocated.blocks for block in group)
    assert manager.block_pool.get_num_free_blocks() == 0
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 4


def test_decode_growth_and_replay_free_both_histories():
    manager = make_manager()
    request = make_request()
    manager.allocate_slots(request, 33)
    request.num_computed_tokens = 33
    request.append_output_token_ids([34])
    allocated = manager.allocate_slots(request, 1)
    assert allocated is not None
    assert [len(group) for group in manager.get_blocks(request.request_id).blocks] == [
        3,
        1,
    ]
    manager.free(request)
    request.num_computed_tokens = 0
    replay = manager.allocate_slots(request, 34, full_sequence_must_fit=True)
    assert replay is not None
    assert [len(group) for group in replay.blocks] == [3, 1]
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 19


def test_partial_prompt_cannot_allocate_a_compact_history():
    with pytest.raises(ValueError, match="complete prompt"):
        make_manager().allocate_slots(make_request(), 16)


def test_free_and_replay_leave_other_requests_blocks_unchanged():
    manager = make_manager()
    first = make_request("first")
    survivor = make_request("survivor", 48, 17)
    assert manager.allocate_slots(first, 33) is not None
    assert manager.allocate_slots(survivor, 48) is not None
    before = [
        [block.block_id for block in group]
        for group in manager.get_blocks("survivor").blocks
    ]
    assert list(map(len, before)) == [3, 2]
    manager.free(first)
    assert manager.allocate_slots(first, 33, full_sequence_must_fit=True) is not None
    after = [
        [block.block_id for block in group]
        for group in manager.get_blocks("survivor").blocks
    ]
    assert before == after
    first_blocks = {
        block.block_id
        for group in manager.get_blocks("first").blocks
        for block in group
    }
    assert not first_blocks.intersection(block for group in after for block in group)
    manager.free(first)
    manager.free(survivor)
    assert manager.block_pool.get_num_free_blocks() == 19


def test_hybrid_allocates_fixed_state_blocks_and_releases_both_histories():
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    recurrent = MambaSpec(
        block_size=64,
        shapes=((4, 8), (2, 8, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="none",
    )
    groups = [
        KVCacheGroupSpec(["lower_attention"], attention),
        KVCacheGroupSpec(["upper_attention"], attention, uses_compact_sequence=True),
        KVCacheGroupSpec(["lower_recurrent"], recurrent),
        KVCacheGroupSpec(["upper_recurrent"], recurrent, uses_compact_sequence=True),
    ]
    manager = KVCacheManager(
        KVCacheConfig(8, [], groups),
        max_model_len=64,
        scheduler_block_size=64,
        hash_block_size=16,
        enable_caching=False,
    )
    request = make_request()
    allocated = manager.allocate_slots(request, 33, full_sequence_must_fit=True)
    assert allocated is not None
    assert [len(group) for group in allocated.blocks] == [3, 1, 1, 1]
    original_states = [group[0].block_id for group in allocated.blocks[2:]]
    request.num_computed_tokens = 33
    request.append_output_token_ids(list(range(8)))
    assert manager.allocate_slots(request, 8) is not None
    blocks = manager.get_blocks(request.request_id).blocks
    assert [len(group) for group in blocks] == [3, 2, 1, 1]
    assert [group[0].block_id for group in blocks[2:]] == original_states
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 7
    request.num_computed_tokens = 0
    replay = manager.allocate_slots(request, 41)
    assert replay is not None
    assert [len(group) for group in replay.blocks] == [3, 2, 1, 1]
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 7


def test_cache_layout_survives_engine_transport_without_feature_arguments():
    source = EngineCoreRequest(
        request_id="layout",
        prompt_token_ids=list(range(33)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        cache_sequence_layout=CacheSequenceLayout(33, 9),
    )
    received = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(source))
    request = Request.from_engine_core_request(received, block_hasher=None)
    assert request.cache_sequence_layout == CacheSequenceLayout(33, 9)
    allocation = make_manager().allocate_slots(request, 33)
    assert [len(group) for group in allocation.blocks] == [3, 1]
