# TokenGate

[TokenGate](https://arxiv.org/abs/2610.26754) adapts pretrained language models
for faster prefill through learned token pruning. A gate scores token hidden
states within a decoder layer and selects which tokens continue through the
model. Subsequent full-attention layers use **token-gated attention**, which
incorporates those scores into the attention computation. Tokens removed by the
gate incur no further layer computation or attention KV-cache entries.

This integration supports dense Qwen3 and the text component of dense Qwen3.5.
It uses their existing model implementations and loads an exported checkpoint
containing a jointly trained gate and adapter. Training happens outside vLLM.
Ordinary checkpoints run without TokenGate; loading an export with `token_gate`
metadata enables it.

## Before you start

Each request identifies one contiguous input span that TokenGate may prune.
The span can contain code, documents, or other text. Its boundaries are token
indices into the complete prompt; TokenGate does not interpret the content's
format. Tokens outside the span are always kept.

| Term | Meaning |
| --- | --- |
| Prunable tokens | Input tokens inside the caller's selected span. |
| Protected tokens | Tokens outside that span, including the final prompt token and all generated tokens. They are always retained. |
| Retained budget, $K$ | Maximum number of prunable tokens to retain. Protected tokens are additional. |
| Full retention | A budget large enough to retain every prunable token. Token-gated attention still applies. |

Use the prompt format and tokenizer used during training. Tokenize the complete
prompt before computing the span; independently tokenizing its pieces can give
different token boundaries.

The current execution requirements are:

| Component | Required setting |
| --- | --- |
| Model | Dense Qwen3 or dense text-only Qwen3.5; no sliding-window attention. |
| Device and precision | One NVIDIA GPU with BF16 support; BF16 backbone and unquantized KV cache. |
| Runner | Model Runner V2, eager execution, synchronous scheduling. |
| Scheduling | One active request, complete prefill, no prefix caching. Requests may queue. |
| Token limits | Both scheduling token budgets cover `max_model_len`; `long_prefill_token_threshold=0`. Prompt and output together fit `max_model_len`. |
| Qwen3.5 cache | Native hybrid cache manager enabled; `mamba_cache_mode="none"`. |

The Python example below sets these options. See [limitations](#limitations)
for combinations that the integration rejects.

## Export trained weights

Prepare a local base-model snapshot with its tokenizer, the matching gate and
adapter checkpoints, and the training `params.json`. An evidence manifest
records their hashes, training step, selection policy, and source revision;
its [required fields](../design/token_gate.md#checkpoint-format) are documented
in the design page.

The exporter is a standalone script in the vLLM source checkout. With vLLM
installed in your environment, run from the repository root using an output
directory that does not exist:

```console
python tools/export_token_gate.py \
    --base /models/qwen3 \
    --gate /run/gate.pt \
    --adapter /run/adapter.pt \
    --params /run/params.json \
    --evidence /run/evidence.json \
    --output /models/qwen3-token-gate
```

The exporter merges the adapter into the BF16 backbone and stores the gate
parameters in FP32. The resulting model can be served without the training
package. Qwen3.5 exports contain only the language model, using
`Qwen3_5ForCausalLM`; adapters on recurrent layers are not supported.

To export the same merged adapter without the gate, add `--without-token-gate`
and choose a different output directory, such as `/models/qwen3-adapter-only`.
This is useful for comparing execution with and without TokenGate. It retains
the jointly trained adapter; a paper-style SFT baseline requires its own
independently trained adapter.

Before relying on an export, compare it with the original research checkpoint.
BF16 adapter merging can change gate scores and the ordering of nearly tied
tokens. Successful export verifies the files and configuration, not model
accuracy or numerical agreement.

## Generate in Python

Set the runner before starting Python:

```console
export VLLM_USE_V2_MODEL_RUNNER=1
```

This example reads `input.txt`, marks its contents as prunable, and protects the
question and answer prefix. Replace the illustrative prompt with the format
used to train your checkpoint.

```python
from pathlib import Path

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.renderers.token_gate import tokenize_span

model_path = "/models/qwen3-token-gate"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
context = Path("input.txt").read_text(encoding="utf-8")
prefix = "Question: What are the main findings?\nDocument:\n"
text = prefix + context + "\nAnswer:"
prompt, gate_args = tokenize_span(
    tokenizer,
    text,
    span_start=len(prefix),
    span_end=len(prefix) + len(context),
    keep_tokens=512,
)

llm = LLM(
    model=model_path,
    dtype="bfloat16",
    enforce_eager=True,
    max_model_len=32768,
    max_num_batched_tokens=32768,
    max_num_scheduled_tokens=32768,
    max_num_seqs=1,
    long_prefill_token_threshold=0,
    async_scheduling=False,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    mamba_cache_mode="none",
)
sampling = SamplingParams(temperature=0, max_tokens=128, extra_args=gate_args)
print(llm.generate(prompt, sampling)[0].outputs[0].text)
```

`tokenize_span()` takes **character offsets** into `text` and returns token
IDs plus aligned **token offsets** for the request. It uses a fast tokenizer
and protects tokens that cross either character boundary. The suffix must
provide at least one token outside the prunable span.

To run **without TokenGate**, start a fresh process with the same example and
replace these two assignments:

```python
model_path = "/models/qwen3-adapter-only"
sampling = SamplingParams(temperature=0, max_tokens=128)
```

Use a checkpoint without `token_gate` metadata and omit `extra_args`. Increasing
$K$ on a TokenGate checkpoint gives full retention, which still uses its learned
attention parameters.

The repository also provides a command-line example. With the runner variable
set, run either command in its own process:

```console
python examples/features/token_gate.py --model /models/qwen3-token-gate \
    --document input.txt --question "What are the main findings?" --keep-tokens 512
python examples/features/token_gate.py --model /models/qwen3-adapter-only \
    --document input.txt --question "What are the main findings?"
```

## Serve tokenized completions

Start the server with the same execution settings:

```console
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /models/qwen3-token-gate \
    --dtype bfloat16 --enforce-eager \
    --max-model-len 32768 --max-num-batched-tokens 32768 \
    --max-num-scheduled-tokens 32768 --max-num-seqs 1 \
    --long-prefill-token-threshold 0 --no-async-scheduling \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --mamba-cache-mode none
```

Send token IDs as `prompt` to `/v1/completions`. Include the aligned request
fields in `vllm_xargs`; for example:

```json
{
  "vllm_xargs": {
    "token_gate_version": 1,
    "token_gate_span_start": 127,
    "token_gate_span_end": 16511,
    "token_gate_keep_tokens": 512
  }
}
```

| Field | Value |
| --- | --- |
| `token_gate_version` | `1`. This is the request format version. |
| `token_gate_span_start` | Zero-based index of the first prunable token. |
| `token_gate_span_end` | Exclusive end of the prunable span. Must be less than the prompt length. |
| `token_gate_keep_tokens` | Positive budget $K$. Values above the span length retain the whole span. |

All four fields must be JSON integers, not booleans. The span must be nonempty.
TokenGate checkpoints require these fields on every request; ordinary
checkpoints reject them. Response usage counts the **original** prompt length.
Output streaming is supported.

## Limitations

The integration is experimental. Unsupported engine settings fail at startup;
invalid request options are rejected before execution.

| Area | Currently unsupported |
| --- | --- |
| Scheduling and execution | Chunked prefill, concurrent active requests, asynchronous scheduling, compilation, CUDA graphs, speculative decoding, and parallel execution. |
| Cache and weights | Prefix caching, KV sharing, cache transfer, cache or weight offload, quantization, and dynamic adapters. |
| Inputs | Text or chat requests, streaming input, multimodal input, and prompt embeddings. Submit pretokenized completions instead. |
| Prompt transformations | Truncation, padding, completion suffixes, or insertion of special tokens after span alignment. |
| Outputs | Multiple completions per request, beam search, prompt logprobs (including `prompt_logprob_token_ids`), pooling, and auxiliary hidden-state outputs. |
| Model variants | MoE, sliding or retrospective sliding-window attention, dual-chunk attention, decoder layer scaling, and HiSparse. |

These limits reflect the current implementation. It selects tokens from a
complete prompt, manages one active request's full and retained histories, and
requires token positions to remain unchanged after alignment. Extending those
assumptions needs additional implementation and testing. Qwen3.5 pruning reduces
recurrent prefill work, but its recurrent-state dimensions remain fixed.

## Check correctness and performance

On a supported CUDA machine, run:

```console
VLLM_USE_V2_MODEL_RUNNER=1 python -m pytest tests/token_gate
```

The tests cover small real checkpoints, independent attention calculations,
the Hugging Face recurrent implementation, and public generation APIs. They do
not establish quality or speed for a trained checkpoint. Initial short Qwen3
and Qwen3.5 TokenGate training pilots failed strict numerical comparisons with
the research implementation; the Qwen3.5 pilots also lost retrieval accuracy.
Those pilots do not establish an accuracy-preserving speedup.

For a trained model, compare the original checkpoint with its merged export,
then compare research execution with vLLM. Check retained tokens, numerical
agreement, and task accuracy before reporting latency or memory improvements.
The [design page](../design/token_gate.md#validation) describes these checks.

For an attention-only benchmark:

```console
python benchmarks/kernels/benchmark_token_gate_attention.py --output timing.json
```

This requires `cupti-python>=13` compatible with the installed CUDA version. It
checks sampled outputs against dense attention, then times ordinary and biased
attention at equal sequence lengths using the production launch heuristic.
Scoring, token selection, metadata preparation, and output restoration are
excluded. Measure end-to-end latency on matched checkpoints and fixed prompts
to determine the benefit of pruning.
