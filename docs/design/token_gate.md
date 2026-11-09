# TokenGate execution and cache management

TokenGate changes how many tokens each part of the model processes. The
integration therefore has two responsibilities: execute the trained gate and
token-gated attention, and keep vLLM's request and cache state consistent with
the retained sequence. The [feature guide](../features/token_gate.md) explains
how to export a checkpoint and submit requests.

## Follow one request through the model

Let $L$ be the zero-based decoder-layer index containing the gate. A request
specifies a prunable span and a retained budget $K$; all other tokens are
protected. Prefill proceeds as follows:

1. **Process the full prompt through layer $L$'s token mixer.** The mixer is
   self-attention in Qwen3, and either full attention or Gated DeltaNet in Qwen3.5.
2. **Score and select tokens.** After the mixer's residual addition and
   post-mixer normalization, the gate scores the prunable tokens. It retains
   the top $K$ in their original order, together with every protected token.
   Both normalized hidden states and residuals are gathered at the same indices.
3. **Continue with the retained sequence.** Layer $L$'s MLP and all later layers
   process only retained tokens. Full-attention layers above $L$ apply
   token-gated attention. Original position IDs remain unchanged for RoPE.
4. **Restore the output layout.** Final hidden states are scattered into the
   original prompt layout, with zeros at pruned rows. The final prompt token is
   protected, so vLLM can use its existing sampling index.

The gate sits **inside** layer $L$, before its MLP. Its token mixer still sees
the full prompt. During decode, every generated token passes through every
layer; the gate does not prune generated tokens.

The existing Qwen3 and NVIDIA Qwen3.5 constructors attach the optional gate and
select the TokenGate backend for full-attention layers above it. The Qwen2 loop
used by Qwen3 and the Qwen3.5 loop receive a sequence-layout object that supplies
positions and restores output rows. No separate TokenGate model class is needed.

## Scores and token-gated attention

The checkpoint declares `attention_semantics="tokengate_code_v1"`. This names
the original TokenGate computation, including its floating-point cast order.

The scoring MLP computes in BF16. Its scalar outputs are converted to FP32 and
RMS-normalized over the prunable span. Normalized scores and ranking remain in
FP32. Selection uses the positive head gains of the first full-attention
layer above the gate, preserving the reference's multiplication and head
reduction before top-K.
Selected indices are sorted back into input order.

### Attention calculation

For a full-attention layer above the gate, define:

| Symbol | Meaning |
| --- | --- |
| $b_j$ | Normalized score of retained token $j$; zero for protected and generated tokens. |
| $p_i$ | Original zero-based position of token $i$. |
| $m_i$ | One-based rank of token $i$ in the retained sequence, including protected tokens. |
| $S_{\ell h},G_{\ell h}$ | Positive learned temperature and score gains for layer $\ell$, query head $h$. Both use softplus. |
| $H,d_h$ | Number of query heads and head dimension. |
| $z_{ij\ell h}$ | Ordinary scaled attention logit, $q_{i\ell h}^{\top}k_{j\ell h}/\sqrt{d_h}$. |

Pruning reduces the number of keys visible to a query. The position ratio
adjusts the attention temperature for that reduction:

$$
r_i=\min\left(1,\frac{\log\max(m_i,2)}{\log\max(p_i+1,2)}\right),
\qquad \tau_{i\ell h}=S_{\ell h}r_i.
$$

The original TokenGate implementation applies query scaling and a key-score bias. Before causal
softmax, its logits are:

$$
a_{ij\ell h}=\tau_{i\ell h}z_{ij\ell h}
+\left(\frac{1}{H}\sum_{h'}\tau_{j\ell h'}\right)G_{\ell h}b_j.
$$

Equation 8 in the [paper](https://arxiv.org/abs/2610.26754) instead applies the
query temperature to both terms. The expression above records the computation
this integration implements. Changing it would change checkpoint behavior.

Floating-point order is also part of that behavior:

| Operation | Cast order |
| --- | --- |
| Prefill query scaling | Cast the gain and position ratio separately to BF16, then multiply. |
| Decode query scaling | Multiply gain and ratio in FP32, then cast the product to BF16. |
| Prompt-key bias | Cast $G_{\ell h}b_j$ to BF16; multiply by the mean over heads of the separately cast gain–ratio products. |

`TokenGateAttentionImpl` scales queries and prepares the prompt-key bias. The
`attention_config.token_gate_prefill_backend` setting selects full-prompt
attention:

- `triton` (default) reads the paged cache through unified Triton attention.
- `tiled` uses a dedicated Triton kernel on the retained Q/K/V tensors already
  produced by the model. It loads each key's bias once per tile and reuses it
  across query rows, with FP32 online softmax and accumulation. Bias values are
  arranged contiguously by head before launch. Normal KV writes still populate
  the paged cache for decoding.

The tiled kernel separates fully visible key tiles from tiles that cross the
causal boundary. Only the latter need causal and sequence-boundary masking.
Both paths retain the same learned bias and online softmax calculation. For
head dimensions above 128, a single pipeline stage keeps wider key tiles within
the shared-memory budget.

The tiled path supports head dimensions up to 256 with ordinary causal attention;
other shapes or attention modifiers use the paged path. Decode and replay
containing generated tokens also use paged attention. Launch geometry depends
only on tensor shape

### Qwen3.5 full-attention layers

Gated DeltaNet has no softmax attention logits. Its layers process the retained
sequence after the gate, but receive no token-gated attention bias. Gain tensors
therefore have one row per full-attention layer above $L$, in this order:

$$
\mathcal{A}=\{\ell>L\mid\text{layer }\ell\text{ uses full attention}\}.
$$

The exported `attention_layer_indices` maps physical layer IDs to those rows.
Using a physical layer offset directly would select the wrong row in a hybrid
model. The gate can be placed after either mixer type, provided at least one
full-attention layer follows it.

## Keep full and retained cache histories

Let $N$ be the original prompt length and $D$ the prunable span length. The
retained prompt length is

$$
M=N-D+\min(K,D).
$$

After the decoder has processed $g$ generated tokens, full-attention layers
through $L$ store $N+g$ KV entries. Full-attention layers above $L$ store $M+g$.
Both use original position $N+g$ for the next token. Scheduler progress,
sampling history, and response usage remain expressed in original tokens.

`CacheSequenceLayout` carries $N$ and $M$ from the input processor through
`EngineCoreRequest`. Core cache code uses these lengths without interpreting
the prunable span. Before predicting or allocating blocks, it maps an original
history endpoint $x$ to the retained history:

$$
f(x)=
\begin{cases}
0 & x=0,\\
M+x-N & x\ge N.
\end{cases}
$$

Endpoints inside a partially processed prompt are rejected. Because $K$ fixes
the retained length before scoring, allocation does not need to copy selected
GPU indices to the CPU.

### Occupied cache versus reserved memory

For homogeneous full attention, let $A$ count layers through $L$, $U$ count
layers above it, $B$ be tokens per block, and $W$ be bytes per block per layer.
The occupied attention-cache bytes are

$$
W\left(
A\left\lceil\frac{N+g}{B}\right\rceil+
U\left\lceil\frac{M+g}{B}\right\rceil
\right).
$$

This is occupied cache, not the size of vLLM's preallocated pool. Reducing
occupied blocks need not reduce reserved GPU memory. Startup capacity is sized
for full retention.

Qwen3.5 uses the native hybrid planner to align physical pages. The resulting
cache groups are partitioned by full or retained history without changing
per-layer cache specifications. Each Gated DeltaNet layer also stores a
convolution state and recurrent state per request. Their shapes do not depend
on $K$: pruning reduces recurrent prefill computation, not state dimensions.

## Own state in the model runner

| Component | Responsibility |
| --- | --- |
| `TokenGate` | Scoring weights, learned gains, selection, and compaction. It owns no mutable request history. |
| `TokenGateHistory` | Persistent request slots containing retained prompt scores and position ratios. |
| `TokenGateBatch` | Views of those buffers and the sequence layout for one execution step. |
| `TokenGateModelState` | Combines TokenGate history with the default model-state lifecycle. |
| `TokenGateHybridModelState` | Combines the same history with the native `MambaHybridModelState` lifecycle. |

The two history channels are FP32. With $R_{\max}$ request slots and configured
maximum sequence length $T_{\max}$, they occupy $8R_{\max}T_{\max}$ bytes.
Removing a request clears its slot before reuse.

Full and retained histories build attention metadata separately so that
mutable native builders cannot alias their sequence lengths. Only
full-attention layers above the gate receive TokenGate metadata. Gated DeltaNet
uses native metadata and state-block indices, without per-token slot mappings.

**Profiling.** Synthetic runs use separate buffers and full retention. They
account for persistent state and temporary workspace before KV-cache sizing.
One-token warmup does not read unallocated caches. This uses the runner's
existing profiling hooks; CUDA graphs remain disabled.

**Preemption.** Replay processes the original prompt and generated history.
It recomputes scores and selection only over the original prunable span and
protects generated tokens. Original prompt rows use prefill arithmetic;
generated rows use decode arithmetic, even during replay. Replay rebuilds both
cache histories and the score/ratio buffers. Hybrid recurrent states are
initialized during fresh prefill or replay, then continued during decode.

## Checkpoint format

The [export command](../features/token_gate.md#export-trained-weights) reads
the TokenGate repository's gate and name-paired adapter files. The gate
contains architecture settings and metadata describing the base model, gate
layer, training step, and adapter targets. Missing selection settings use the
original loader's defaults (`topk`, `random_topk=False`); explicit unsupported
settings are rejected.

The exporter records hashes of the input weights, configuration, and tokenizer
in `config.json`, and writes `token_gate_export_manifest.json` with hashes of
the output files. Benchmark records separately identify pinned Hub revisions
and source code. These records identify files; they do not establish numerical
equivalence or independently verify how the checkpoint was trained.

The exporter checks tensor completeness, dimensions, policies, training steps,
and finite merged values. It rejects unknown tensors, unsupported gate
variants, threshold or random selection, nonfinite scales, and BF16 merge
overflow. It resolves omitted adapter targets to TokenGate loader defaults; delta
adapters cover all four attention and all three MLP projections in each declared
layer. Qwen3.5 adapters must explicitly list full-attention layer IDs and cannot
target recurrent layers.

Serving metadata includes an explicit `attention_layer_indices` list for both
Qwen3 and Qwen3.5. Qwen3.5 exports use
`Qwen3_5ForCausalLM` and preserve partial RoPE, the native attention output gate,
tied embeddings, and the configured recurrent-state dtype.

## Validation

The tests cover request validation, checkpoint pairing, adapter merging, score
casts, selection, residual gathering, output restoration, cache allocation,
paged attention, public generation, request queues, and slot reuse. CUDA tests
use small real Hugging Face checkpoints, independent dense attention, and the
Hugging Face recurrent implementation.

Validate a trained checkpoint in two stages:

1. Compare the original TokenGate model with its merged-weight export to isolate
   adapter-merging effects.
2. Compare that same export in the original TokenGate implementation and vLLM. Check scores on
   identical gate inputs, retained indices, logits or logprobs on a shared
   generated history, and complete answers.

Keep numerical agreement separate from task accuracy: matching answers alone
does not establish equivalent execution. Full retention must be compared with
the same TokenGate model, since an ungated model omits its learned attention.
Then measure accuracy, TTFT, decode latency, occupied cache, and peak memory
across input lengths and budgets. Report serial request throughput separately
from concurrent serving, which this integration does not support.
