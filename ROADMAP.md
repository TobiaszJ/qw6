# qw6 Roadmap

Development phases for the Qwen 3.6-35B-A3B inference engine on AMD BC-250.

---

## Phase 0: Research & Design

- [x] Study ds4 architecture (antirez/ds4)
- [x] Study Qwen 3.6-35B-A3B architecture (config.json, HF transformers docs)
- [x] Study BC-250 / GFX1013 / Vulkan compute landscape
- [x] Write ARCHITECTURE.md
- [x] Write MODEL_CARD.md
- [x] Write BC250_SETUP.md
- [x] Create GitHub repo and project structure

**Deliverable:** complete design documentation and initial source tree.

---

## Phase 1: CPU Reference Path

- [x] GGUF v3 metadata parser and model validator
- [x] `mmap`-backed GGUF tensor indexing
- [x] Unsloth/llama.cpp Qwen3.6 layout support (`qwen35moe`, `blk.*`)
- [x] packed routed-expert tensors split into per-expert views
- [x] tokenizer encode/decode regression (`make test-tokenizer`)
- [x] native dequant: F32, F16, BF16, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_S
- [x] generic native tensor MatVec for dequantized rows
- [x] CPU kernels: RMSNorm, MoE routing (top-8/256), GQA attention, MRoPE,
      Gated DeltaNet (conv1d, update, retrieve), SiLU, softmax, argmax, MTP draft
- [x] full 40-layer forward to logits (CPU smoke path)
- [x] `qw6 --load-only` native GGUF validation/dequant probe
- [x] greedy native token generation smoke test
- [x] `make cpu` build target
- [x] `qw6 --dump-logits` / `--dump-logprobs`
- [ ] correct logit parity vs reference (requires test vectors from RTX 3090 server)

**Note:** Phase 1 was never validated against reference logits. Both CPU and GPU
paths may produce incorrect output. Core inference math needs comparison against
llama.cpp outputs layer-by-layer.

---

## Phase 2: Vulkan Compute Backend (Partial)

Goal: GPU-accelerated inference on BC-250 via Vulkan compute shaders.

### Current audited state

- [x] BC-250 Vulkan device enumeration works in the normal path and reports AMD BC-250 / RADV GFX1013.
- [x] The complete GGUF now loads with `--load-only --vulkan`.
- [x] The Vulkan smoke path can run a real prompt and produce tokens.
- [x] SPIR-V build covers 24 shader sources, not the older "18 + 1" list.
- [x] Pipeline objects are cached again, and there is a reusable descriptor pool, command buffer, and fence.
- [x] IQ2_S has a first GPU shader path.
- [x] IQ3_S GPU matvec shader added.
- [x] Linear-attention Conv1D and Gated DeltaNet have first fused GPU shader paths.
- [x] Q/K RMSNorm per-head moved to GPU (rmsnorm_heads.comp).
- [x] Q/K L2 normalization moved to GPU (l2_norm_heads.comp).
- [ ] Correctness is not proven. CPU, GPU, and llama.cpp logits have not been compared layer-by-layer or token-by-token.
- [ ] Throughput is still far below the stated goal. Current smoke measurements are about 1.57 s/token for `Hi -n 1`, and about 4 tokens in 6.29 s for `Hi -n 4` on the Vulkan path.
- [ ] The current pipeline is not yet a llama.cpp-style backend. It is a Vulkan-assisted forward pass with many host-visible buffers, immediate per-dispatch waits, and several CPU steps inside each layer.

### Audited source scope

- [x] `qw6.c`: GGUF loading, tensor binding, native dequant, CPU forward, session, prefill, generation, CLI, CPU probes.
- [x] `qw6_vk.c`: Vulkan device setup, buffer allocation, host wrappers, self-test, pipeline cache, weight upload, full Vulkan pipeline forward, cleanup.
- [x] `qw6_tok.c`: tokenizer JSON parser, byte-level BPE, pre-tokenization, encode/decode.
- [x] Headers: `qw6.h`, `qw6_vk.h`, `qw6_tok.h`, `qw6_iq_tables.h`.
- [x] Focused shader audit: quant matmuls, attention, Conv1D, Gated DeltaNet, and shader inventory.

### Current GPU shader inventory

- [x] Elementwise and small ops: `add.comp`, `vec_add.comp`, `silu_mul.comp`, `argmax.comp`, `sampling.comp`.
- [x] Norms: `rmsnorm.comp`, `rmsnorm_apply.comp`, `rmsnorm_full.comp`.
- [x] Dense FP32: `matvec_f32.comp`, `mtp_draft.comp`.
- [x] Quant matvec: `matmul_q4k.comp`, `matmul_q5k.comp`, `matmul_q6k.comp`, `matmul_iq2xxs.comp`, `matmul_iq2s.comp`.
- [x] Attention: `rope_mrope.comp`, `attention_gqa.comp`.
- [x] Linear attention: `deltanet_conv1d.comp`, `deltanet_conv1d_f32.comp`, `deltanet_update.comp`, `deltanet_retrieve.comp`, `deltanet_gated.comp`.
- [x] MoE helpers: `moe_route.comp`, `moe_gather.comp`.
- [x] IQ3_S GPU matvec shader added.
- [ ] Missing for performance: fused model-specific kernels that combine the operations currently split across many small dispatches.

### Immediate correctness blockers

- [x] Add a real reference-logit harness: `tests/correctness/compare.py` provides structured tokenizer, logit, and generation comparison against llama.cpp. Runnable on BC-250 with `python3 tests/correctness/compare.py all`.
- [x] Implement `--dump-logits`, `--dump-logprobs`, `--dump-full-logits`, and `--trace-json` for both CPU and Vulkan paths. Full logit vectors can be dumped for cross-reference.
- [x] Add structured trace JSON output (`--trace-json`) for prompt tokens, generated tokens, and top logits after prefill/final token.
- [ ] Establish CPU vs Vulkan parity for one token and for a multi-token prompt. The Vulkan path currently has separate math from the CPU path in Conv1D, Gated DeltaNet, attention, routing, and quant matmuls.
- [ ] Validate the final generated text against llama.cpp. CPU path produces output (verified: "Hello" -> token 1873 on CPU). BC-250 comparison not yet run.
- [x] Make correctness tests fail hard on any CPU fallback in Vulkan performance mode via `--vulkan-strict`. Unsupported GPU quant or long-context attention now abort instead of silently falling back to CPU.
- [x] Add --bench mode: runs 128-token timed generation with tok/s report.
- [x] Add --seed flag for reproducible generation.

### Loader and model binding problems

- [x] Required tensor validation is incomplete. Some binding helpers can silently skip malformed tensors; the loader can still appear to succeed if raw layer tensor counts look plausible.
- [x] `bind_expert_pack` now reports shape, stride, quant, and byte-span errors with descriptive messages. Returns -1 on failure, checked by all callers.
- [x] Unknown GGUF tensor types are fatal. The loader now rejects unsupported tensor types instead of collapsing them toward FP32 semantics.
- [x] Tensor byte sizes validated via `gguf_validate_tensor_ranges()` using `ggml_type_block_elems/bytes`.
- [x] Large-file handling: ftell/fseek → ftello/fseeko with off_t for 64-bit file sizes.
- [x] `type_counts[30]` → `type_counts[36]` (enum goes to GGML_TYPE_TQ2_0 = 35).
- [x] Model metadata validated: architecture, layer count, hidden size, query heads, KV heads, expert count, experts per token, MoE inter size.
- [x] The Vulkan init path mutates `attn_o` tensors to FP32 after pre-dequantization. This workaround should be replaced with a separate GPU-side replacement tensor or fixed Q5_K shader so the model structure remains immutable after load.
- [x] `vk_offset` sentinel fixed: comment now correctly says `(size_t)-1` = not uploaded.

### Tokenizer and prompt-format problems

- [ ] The tokenizer is a reference implementation, not a faithful production Qwen tokenizer. Pre-tokenization is simplified ASCII/basic-Unicode logic and does not implement the full HuggingFace regex behavior.
- [x] Special token detection is handled during encode for added special tokens, so chat and control markers survive as single IDs.
- [ ] The chat template is hardcoded instead of read from tokenizer metadata. This can shift prompt tokens from llama.cpp and invalidate both correctness and performance comparisons.
- [ ] Encoding uses a temporary static hash map and linear merge search. This is not a decode bottleneck, but it is too fragile for repeated prompt benchmarking and server use.
- [ ] Tokenizer regression coverage is tiny. Add fixtures collected from HuggingFace or llama.cpp for chat prompts, special tokens, Unicode, whitespace, code, numbers, and multi-message conversations.

### CPU path problems that affect validation

- [ ] CPU self-tests for toy Q4_K and IQ2 layouts do not validate real GGUF quant layouts. They are useful unit fixtures but not proof that model weights are dequantized correctly.
- [ ] CPU probes cover only small slices or layer 0. They do not prove full-model correctness.
- [ ] DeltaNet has older update/retrieve helpers plus the actual `qw6_gated_delta_net_single` recurrence. Tests must validate the recurrence used by the model path, not only the older helper kernels.
- [ ] The Conv1D probe is not enough to validate the stateful model path. It must cover multi-token state shift, weight layout, SiLU, and in-place buffer reuse.
- [x] CPU fallback inside the Vulkan pipeline uses mapped buffers only in non-strict/debug mode; performance mode now rejects fallback through `--vulkan-strict`.
- [x] Session initialization allocates CPU KV/state buffers even when the Vulkan pipeline is used. That wastes memory and muddies profiling on the BC-250.
- [x] `qw6_forward_token` allocates CPU temporary buffers before it redirects to Vulkan. This creates avoidable allocation overhead in the GPU path.

### Vulkan device and memory architecture problems

- [ ] All buffers are currently allocated as host-visible, host-coherent memory. The 10.7 GB weights should live in device-local memory with staging uploads; host-visible weights are not a high-throughput llama.cpp-style backend.
- [ ] Add a proper allocator for device-local weights, persistent device scratch, upload staging, and small uniform/control buffers. The current one-buffer helper is too limited.
- [x] Add memory-type diagnostics and fail clearly if device-local capacity is insufficient. BC-250 behavior depends on the RADV memory heaps actually selected.
- [x] Device selection picks the first integrated/discrete GPU. Add device listing, BC-250 preference, vendor/device ID checks, queue capability checks, and an environment or CLI override.
- [x] Use dedicated transfer and compute usage flags where appropriate. The current buffer usage is too generic for staging and device-local layout.
- [ ] Stop mapping every performance-critical buffer. Persistent host mapping is acceptable for staging and small readback, not for all weights and activations.
- [x] Rework the +512 MB dequant temporary space in the weight buffer. It is a workaround for `attn_o`, not a real memory plan.
- [x] Avoid copying norm weights into a scratch buffer every layer. Bind norm tensors directly from the weight buffer or store persistent small buffers.
- [ ] KV cache and DeltaNet state are FP32 host-visible buffers. Revisit layout, precision, and device-local placement once correctness is proven.

### Vulkan dispatch architecture problems

- [ ] `qw6_vk_pipe_dispatch` still records one command buffer, submits it, and waits on a fence for every shader call. This destroys throughput.
- [ ] Descriptor sets are still allocated, updated, and the descriptor pool is reset per dispatch. Pipeline caching alone does not remove CPU driver overhead.
- [ ] The cached descriptor layout uses fixed arrays for up to five buffers. That is fragile and blocks richer fused kernels unless expanded or redesigned.
- [ ] The pipeline cache is capped at 32 entries and falls back to the uncached legacy dispatcher if exceeded. This fallback creates pipelines, descriptor pools, command buffers, fences, and shader modules per call.
- [ ] There is no command graph or static execution plan. The layer sequence should be planned once, then replayed with updated push constants and buffer offsets.
- [ ] Replace per-op waits with batched command buffers per token, per layer group, or per decode step. Use pipeline barriers and a small number of queue submissions.
- [x] Use timestamp queries around GPU work. Current profile timing wraps queue submit plus fence wait and mixes CPU driver overhead with GPU execution.
- [x] Add a "no fallback, no readback" decode benchmark mode to prove the GPU is doing the work.
- [x] The current `QW6_PROFILE` path prints useful aggregate shader timing, but it is not a full profiler. Add dispatch counts, CPU-side setup time, GPU timestamps, bytes read, bytes written, and fallback counts.

### Quant matmul problems and todos

- [ ] Fix Q5_K before removing the `attn_o` workaround. The current Q5_K shader is marked wrong and `attn_o` is pre-dequantized to FP32 during init.
- [ ] Validate Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, and IQ3_S against llama.cpp dequant for real rows from the loaded GGUF, not only toy self-test blocks.
- [ ] Optimize and validate the IQ3_S GPU matvec. The shader path exists, but the current implementation is serial per output row and cannot be treated as performance-ready.
- [ ] Decide whether IQ2_M is truly equivalent to the current IQ2_XXS shader path. If not, split it into a separate kernel and validation case.
- [ ] The current quant matmul shaders use one workgroup per output row and do a scalar reduction over columns. That is simple but not enough for BC-250 throughput.
- [ ] Add wave64/subgroup reductions tuned for GFX1013 instead of shared-memory reductions everywhere.
- [ ] Process multiple rows per workgroup or multiple rows per wave where dimensions allow. MoE expert matrices are small enough that launch overhead and row granularity dominate.
- [ ] Cache or broadcast the input activation vector efficiently. Current kernels reread `x` for every output row, which is expensive for large row counts.
- [ ] Add vectorized loads for packed quant data and activation values. The shaders currently reconstruct bytes with scalar helper calls.
- [ ] Specialize kernels for the exact model dimensions: hidden 2048, MoE inter 512, shared inter 512, output vocab 248320, QKV dimensions, and expert top-k 8.
- [ ] Optimize the final output projection separately. It is a huge vocab matvec and should feed GPU argmax/top-k directly instead of copying all logits to CPU every token.

### Linear-attention and DeltaNet problems and todos

- [ ] Validate the fused `deltanet_gated.comp` math against `qw6_gated_delta_net_single` for real model tensors and multi-token state. Naming in the C path passes alpha-like values through a parameter called gate, so tests must verify exact semantics.
- [ ] Move Q/K L2 normalization to GPU. The linear-attention path still normalizes heads on CPU after Conv1D.
- [ ] Move beta sigmoid and alpha softplus/scale to GPU. The current path reads/writes those small vectors through host-mapped memory every linear-attention layer.
- [ ] Move Gated DeltaNet output RMSNorm per value head to GPU.
- [ ] Fuse SiLU(z) with the Gated DeltaNet output norm or the following output projection input preparation.
- [ ] Validate Conv1D state update with in-place input/output buffer reuse. The shader currently reads and writes through different bindings that can point at the same scratch buffer.
- [ ] Fuse the linear-attention sequence where practical: QKV projection, Conv1D, SiLU, Q/K L2 norm, alpha/beta transforms, recurrent update/retrieve, output norm, z gate, and output projection staging.
- [ ] Revisit DeltaNet state layout for coalesced access. The current shader loops over `dim` per output element and may produce poor memory locality.
- [ ] Add long-run state tests. A one-token or tiny random self-test will not catch accumulation drift in recurrent state.

### Full-attention problems and todos

- [ ] Move Q/K per-head RMSNorm to GPU. It is currently CPU work in every full-attention layer.
- [ ] Move attention gate sigmoid/multiply to GPU. It is currently a CPU loop over the attention output.
- [ ] Write K/V cache from GPU, not via host-mapped CPU copies.
- [ ] Remove the `pos < 256` attention limitation. `attention_gqa.comp` has a 256-entry shared score buffer and falls back to CPU for longer contexts.
- [ ] Add an attention kernel that handles the target context lengths with tiling, streaming softmax, and GQA layout tuned to 16 query heads, 2 KV heads, and head dimension 256.
- [ ] Fuse MRoPE, KV append, attention, gate, and output projection preparation where it reduces dispatch and memory traffic.
- [ ] Revisit KV precision and layout. FP32 KV is simple but may waste bandwidth and memory for long contexts.

### MoE problems and todos

- [ ] Keep routing on GPU. The current pipeline computes router logits on GPU, then routes on CPU.
- [ ] Keep selected expert indices and weights on GPU. CPU routing forces host synchronization before every MoE block.
- [ ] Replace the per-expert sequence of gate matmul, up matmul, SiLU multiply, down matmul, and CPU accumulation. With top-k 8 and 40 layers, this creates hundreds of tiny synchronized dispatches per token.
- [ ] Implement a fused or batched selected-expert executor for exactly top-k 8, 256 experts, hidden 2048, and expert inter 512.
- [ ] Accumulate expert outputs on GPU. The current accumulation is a CPU loop over hidden size for each selected expert.
- [ ] Move shared expert routing and accumulation to GPU. The shared-router sigmoid and final shared accumulation are CPU work today.
- [ ] Precompute per-layer expert tensor offsets and quant metadata in a GPU-readable table so kernels do not need CPU-side branching per expert.
- [ ] Add MoE-specific validation against llama.cpp: router logits, selected expert IDs, softmax weights, each selected expert output, shared expert output, and final MoE residual.

### Prefill problems and todos

- [ ] `qw6_prefill` processes prompt tokens one at a time through full decode-style forward. There is no chunked or batched prefill despite `QW6_PREFILL_CHUNK`.
- [ ] Implement chunked prefill for full-attention layers so prompt tokens share projection and attention work.
- [ ] Design a linear-attention prefill strategy for Gated DeltaNet. The recurrence is sequential, but the command scheduling, projections, and MoE work can still be batched or pipelined.
- [ ] Avoid rebuilding or resubmitting the same layer command sequence per prompt token.
- [ ] Add prefill throughput metrics separate from decode throughput. The current smoke timing mixes prompt processing and generated-token timing.

### Sampling and output problems

- [x] `qw6_sample` now respects temperature, top-p, and top-k. Greedy remains the `temp <= 0` path for deterministic validation.
- [ ] Avoid full-vocab logits readback in the Vulkan generation path. For greedy or top-k/top-p, sampling should run on GPU and read back only the chosen token and optional debug values.
- [ ] Finish integrating `argmax.comp` and `sampling.comp` into the decode loop. The shaders are now present and partially wired, but the selected token is not propagated correctly through generation.
- [x] Add a mode to dump logits for correctness and another mode to avoid full-logit readback for performance.

### Benchmark and profiling problems

- [ ] The old 150+ tok/s target is aspirational until correctness and a fair llama.cpp baseline are recorded.
- [ ] Benchmark llama.cpp and qw6 on the same BC-250, same GGUF, same prompt, same context, same token count, same sampling, warm run, and cold run.
- [ ] Measure wall-clock time, GPU timestamp time, CPU submit/setup time, model load time, first-token latency, prefill tok/s, and decode tok/s separately.
- [ ] Use wall-clock timers for user-visible throughput. CPU `clock()` is not enough for GPU queue timing.
- [ ] Report whether weights are host-visible or device-local in each benchmark.
- [ ] Report dispatch counts per generated token and per layer. This is currently a primary bottleneck.
- [ ] Add roofline-style estimates for each dominant kernel: quant bytes read, activation bytes read, FLOPs or effective ops, achieved bandwidth, and occupancy estimate.

### Cleanup and maintainability problems

- [x] `qw6_vk_pipe.h` describes an older separate pipeline structure and API that no longer matches the monolithic implementation in `qw6_vk.c`. Either delete it or bring it back in sync.
- [x] `qw6_vk_int.h` declares extension helpers that are not used by the current monolithic implementation. Remove or reintegrate them.
- [x] File headers still describe Phase 1/Phase 2 as if the project is a clean CPU reference plus planned Vulkan backend. Update once correctness status is clear.
- [ ] Split the Vulkan backend into device/memory, shader dispatch, model upload, kernels, and pipeline orchestration modules once behavior stabilizes. Do not refactor before parity tests exist.
- [ ] Add CI or local scripts for `make cpu`, `make vulkan`, shader compilation, tokenizer tests, CPU self-test, Vulkan self-test, load-only, and a short deterministic generation check.

### Verification currently available

- `make vulkan`
- `./qw6 --self-test`
- `./qw6 --vulkan-self-test`
- `./qw6 --load-only --vulkan -m models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`
- `./qw6 -m models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf -p "Hi" -n 1 --nothink --raw --vulkan`

These checks prove that the program builds, enumerates the GPU, loads the complete GGUF, and executes a smoke prompt. They do not prove correct model output or competitive performance.

---

## Phase 3: Correctness and llama.cpp-Class Optimization

Goal: first match llama.cpp numerically, then surpass its BC-250 throughput for this exact Qwen 3.6-35B-A3B IQ2_XXS GGUF and hardware.

### Milestone 0: Make correctness measurable

- [x] Add deterministic llama.cpp comparison harness: `tests/correctness/compare.py` runs tokenizer, chat template, logit, and generation comparison between qw6 and llama.cpp.
- [x] Tokenizer comparison: 12/15 prompts match. 3 mismatches are pre-tokenization differences with `{`/`}` boundaries (qw6's simplified pre-tokenizer vs HuggingFace regex).
- [x] Add `--dump-full-logits <FILE>` flag: writes full 248k logit vector as raw float32 binary for cross-reference with llama.cpp.
- [x] Fix qw6_dump_tokens to output all tokens (previously truncated at 20 with `...` which broke JSON parsing).
- [x] Add NaN/Inf detection: qw6_check_nan_inf() checks float buffers, integrated into every forward pass (embedding, per-layer attn/FFN, final logits).
- [x] Add `--dump-logits`, `--dump-logprobs`, `--trace-json` flags for inspecting logit/logprob values.
- [ ] Add CPU vs Vulkan tensor-dump comparison for one-token and multi-token prompts.
- [ ] Fix all correctness mismatches before optimizing kernels whose output is not yet proven.

### Milestone 1: Remove unintended CPU work from decode

- [ ] GPU token embedding lookup/dequant.
- [ ] GPU norm weight binding without per-layer CPU copies.
- [x] GPU per-head Q/K RMSNorm (rmsnorm_heads.comp).
- [x] GPU per-head Q/K L2 norm (l2_norm_heads.comp).
- [x] GPU alpha/beta sigmoid/softplus transforms (deltanet_alpha_beta.comp).
- [x] GPU attention gate sigmoid multiplication (sigmoid_mul.comp).
- [x] GPU fused DeltaNet output RMSNorm + SiLU(z) gating (deltanet_norm_gate.comp).
- [x] GPU KV cache writes (buf_copy.comp).
- [ ] Long-context attention: current shader limited to 256-token context.
- [x] GPU MoE routing (moe_route.comp integrated, reads back 8 indices + 8 weights instead of 256 logits).
- [x] GPU expert accumulation (vec_axpy.comp for ffn += w * nrm).
- [ ] Wire GPU argmax/sampling into generation correctly so the no-readback path actually feeds the selected token back into the decode loop; the current shader/pipeline scaffold is not enough by itself.

### Milestone 2: Replace per-dispatch synchronization with a backend graph

- [ ] Build a static per-layer execution plan for the exact model.
- [ ] Allocate persistent descriptors or descriptor buffers for all stable resources.
- [ ] Record batched command buffers for each decode token or layer group.
- [ ] Use barriers between dependent kernels and fence only at the token boundary.
- [ ] Add GPU timestamp queries and fallback counters.
- [x] Keep debug mode separate from performance mode.

### Milestone 3: Move weights and state to the right memory

- [ ] Upload weights to device-local memory through staging.
- [ ] Put scratch, KV cache, DeltaNet state, and Conv1D state in device-local memory.
- [ ] Use small host-visible buffers only for input token IDs, control data, and final token readback.
- [x] Remove the FP32 `attn_o` mutation by fixing Q5_K or storing an explicit replacement tensor outside the model object.
- [ ] Add memory-budget checks for BC-250 before allocation.

### Milestone 4: Model-specific fused kernels

- [ ] Fused linear-attention block for the 30 DeltaNet layers.
- [ ] Fused full-attention block for the 10 GQA layers.
- [ ] Fused selected-expert MoE executor for top-k 8.
- [ ] Optimized output projection plus GPU sampling.
- [ ] Specialized quant matvec kernels for the exact matrix shapes in this GGUF.
- [ ] Wave64/subgroup versions tuned for RADV GFX1013.

### Milestone 5: Batched prefill and server-ready decode

- [ ] Chunked prefill for prompt throughput.
- [ ] Decode loop that keeps all recurrent state on GPU.
- [ ] Prompt cache and session reset semantics.
- [x] Correct greedy, temperature, top-k, and top-p sampling.
- [ ] Benchmark harness that reports prefill tok/s, decode tok/s, first-token latency, and steady-state latency.

---

### Audit 2026-07-06: exact blockers before qw6 can beat llama.cpp

The current implementation is a Vulkan-assisted single-token runner. To surpass llama.cpp, it needs to become a correctness-proven, low-synchronization graph executor with optimized quant kernels, chunked prefill, and trustworthy benchmarking. The items below are the concrete problems found by reading the C code and the Vulkan-facing hot path.

#### 1. Correctness must be proven before performance work is meaningful

- [ ] Build a strict reference harness that compares logits, sampled tokens, and final text against llama.cpp or another trusted Qwen3-Next runner for identical prompts, model files, sampling settings, and context positions.
- [ ] Gate every benchmark on correctness. A faster result is not acceptable until CPU logits, Vulkan logits, tokenizer output, stop conditions, and generated token sequences are within a defined tolerance or exactly match where exact matching is expected.
- [ ] Add layer-by-layer dumps for CPU and Vulkan paths: token embedding, per-layer RMSNorm outputs, attention outputs, DeltaNet outputs, MoE router decisions, selected expert outputs, residual streams, final norm, and logits.
- [ ] Add tensor-level checks for every supported quant format using real GGUF rows, not toy fixtures. The dequantized dot product from qw6 must match the equivalent ggml/llama.cpp path for Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, and IQ3_S.
- [ ] Decide the allowed numeric tolerance per subsystem. Quant matvec, RMSNorm, softmax, DeltaNet recurrence, and MoE accumulation need separate tolerances because they amplify errors differently.
- [ ] Keep the CPU path as a reference path but remove allocation churn from it so CPU validation does not distort profiling results or hide lifetime bugs.

#### 2. Generation correctness blockers

- [ ] Fix the Vulkan greedy/no-readback generation loop. `qw6_forward_token` receives a sampled token from `qw6_vk_pipe_forward_greedy`, but the generated token is not stored in session state and `qw6_generate` still calls `qw6_sample` from `s->logits` before forwarding the next token. In the no-readback path, `s->logits` is stale or absent, so generation can choose the wrong token.
- [ ] Redesign the session API so decode state has an explicit "next token selected on GPU" field, or split prefill/decode so the caller can consume GPU sampling results without a full-vocab readback.
- [ ] Make greedy, temperature sampling, top-k, top-p, and repetition penalties consistent between CPU and Vulkan. The current GPU sampling path only addresses argmax-style selection, while the public generator still assumes CPU-side logits are available.
- [ ] Verify BOS, EOS, IM_START, and IM_END token IDs against tokenizer metadata. The header currently assigns the same value to BOS and EOS while tokenizer constants define separate chat control tokens, which can break stopping behavior and benchmark fairness.
- [ ] Define exact stop-token handling for base prompts, chat template prompts, and generated assistant messages. Do not rely on one hardcoded EOS value until tokenizer metadata is parsed and validated.

#### 3. Tokenizer and prompt parity

- [ ] Replace the simplified pre-tokenizer with a tokenizer path that is proven against the shipped Qwen tokenizer configuration, including byte fallback behavior, special tokens, chat templates, and Unicode edge cases.
- [ ] Remove fixed-size encode limitations from the hot path or turn them into explicit, tested errors. Current temporary buffers cap pre-token bytes and encoded output size, which can silently diverge from reference behavior on long or non-ASCII prompts.
- [ ] Avoid global tokenizer encode-cache state unless it is tied to a tokenizer instance and has an explicit lifetime. A global cache can mix tokenizer data across models and prevents clean reload behavior.
- [ ] Add tokenizer conformance tests with real fixtures from the model tokenizer: plain ASCII, whitespace-heavy text, UTF-8 text, emoji, CJK text, code, chat messages, tool-like markup, and long prompts.
- [ ] Keep hardcoded chat formatting out of the benchmark path unless the llama.cpp comparison uses the identical template and stop rules.

#### 4. Vulkan backend architecture

- [ ] Replace per-dispatch command-buffer recording, queue submission, fence waiting, descriptor-pool reset, and descriptor-set allocation with a graph-style executor that records many operations for a token or batch into one command buffer.
- [ ] Stop waiting on the GPU after every shader. Insert Vulkan memory barriers between dependent kernels inside the same command buffer and wait only at token boundaries, benchmark boundaries, or required readback points.
- [ ] Move descriptor management to reusable descriptor sets, descriptor buffers, push descriptors, or another low-overhead binding model. The current one-set descriptor pool and fixed five-binding layout cannot scale to fused kernels or graph execution.
- [ ] Replace the fixed-size pipeline cache with a real cache keyed by shader, descriptor layout, specialization constants, device properties, and kernel variant. The current cache limit matches the current shader count and leaves no room for optimized variants.
- [ ] Add specialization constants or generated variants for common dimensions: model width, FFN width, head dimension, value dimension, quant block shape, subgroup size, and selected expert count.
- [ ] Add device capability detection for subgroup size, subgroup arithmetic support, shared-memory limits, storage-buffer alignment, preferred workgroup sizes, and vendor/architecture-specific paths.
- [ ] Split init-time upload work from runtime dispatch state. Runtime should not depend on mutable GGUF tensor metadata or ad hoc buffer offset fields that are changed during upload.
- [ ] Introduce a GPU execution plan for the full model: embedding, layer loop, final norm, logits, and sampling. The plan should be created once and replayed with per-token dynamic parameters.

#### 5. GPU memory architecture

- [ ] Stop storing all weights, KV cache, DeltaNet state, scratch buffers, and logits in host-visible coherent memory. This is convenient but leaves bandwidth on the table and prevents performance comparable to llama.cpp on discrete GPUs.
- [ ] Allocate model weights in device-local memory and upload through staging buffers. Keep only small readback buffers host-visible.
- [ ] Allocate activations, MoE scratch, KV cache, and DeltaNet recurrence state in device-local memory unless they are explicitly needed by the CPU.
- [ ] Add a proper allocator with alignment, lifetime classes, transient scratch reuse, and high-watermark reporting. The current buffer-per-purpose approach makes fusion and chunked prefill harder.
- [ ] Keep a compact GPU-readable tensor metadata table with offsets, shapes, strides, quant type, and per-layer/expert identity so kernels can execute selected experts without CPU-side pointer chasing.
- [ ] Correct memory traffic accounting. Current profiling estimates use descriptor ranges that often span from an offset to the end of a buffer, so reported read/write bytes can be far larger than the real tensor access.
- [ ] Add separate accounting for weight bytes, activation bytes, KV bytes, state bytes, CPU-GPU transfers, descriptor overhead, and synchronization time.

#### 6. Quantized matvec and matmul kernels

- [ ] Replace the IQ3_S kernel that performs one output row with one active lane. It serializes the whole row and cannot be competitive.
- [ ] Audit every quant shader for lane utilization. A competitive backend needs subgroup reductions, vectorized loads, coalesced scale/min reads, and multiple output rows or tiles per workgroup where appropriate.
- [ ] Avoid rereading the entire activation vector independently for every output row when several rows can share the same activation tile in shared memory or registers.
- [ ] Implement optimized kernels for the actual dominant GGUF quant formats in the target model, not just generic support for many formats.
- [ ] Resolve the Q5_K handling strategy. The current path replaces at least one Q5_K tensor with an FP32 copy, which increases memory footprint and bandwidth. Either fix native Q5_K performance and correctness or explicitly justify the FP32 replacement with measurements.
- [ ] Clarify IQ2_M support. The enum and Vulkan dispatch path mention it, but GGUF type mapping does not appear to produce it. Unsupported formats should fail clearly; supported formats need real validation.
- [ ] Add kernel variants for output sizes used by q, k, v, o, router, gate, up, down, shared expert, norm projections, and logits instead of routing everything through one generic row kernel.
- [ ] Add a batch/prefill matmul path. Single-vector matvec is the wrong primitive for prompt processing and will not match llama.cpp prompt-processing throughput.
- [ ] Benchmark each quant kernel standalone against llama.cpp-equivalent kernels using identical tensors, dimensions, and GPU clocks before integrating changes into full-model timing.

#### 7. MoE routing and expert execution

- [ ] Keep MoE routing on GPU from router logits through top-k selection, expert weighting, selected expert execution, shared expert gating, and residual accumulation. Current routing still reads selected expert IDs and weights back to the CPU, forcing synchronization and CPU-directed dispatch.
- [ ] Replace the serial single-invocation top-k router with a parallel top-k implementation or a fused router/top-k kernel. With 256 experts and 8 selected experts this is small, but doing it serially in every layer still adds latency and blocks graph execution.
- [ ] Fuse or batch selected expert execution. The current path dispatches gate matmul, up matmul, activation, down matmul, and accumulation per selected expert, per layer. This creates hundreds of tiny dispatches per token.
- [ ] Create a grouped-MoE executor that loads selected expert metadata on GPU, runs all selected experts for a layer in one or a small number of dispatches, and accumulates weighted outputs without CPU intervention.
- [ ] Move shared expert router sigmoid and shared expert accumulation fully onto the GPU. Reading a scalar back to the CPU in every layer is a synchronization point and invalidates the no-readback design.
- [ ] Add support for routing batch tokens during prefill. MoE routing and expert execution need to operate on chunks, not only one token at a time.
- [ ] Track expert weight bandwidth and cache behavior. Selected expert weights dominate decode cost, so expert layout and dispatch order need to be optimized for locality.

#### 8. Attention path

- [ ] Replace the current full-attention kernel with a tiled streaming-softmax attention implementation that supports the full context length, not only the small-context path guarded by a fixed score buffer.
- [ ] Remove the short-context fallback condition that only uses GPU attention below a small position threshold. Long-context decode must stay on GPU and remain correct.
- [ ] Store and access KV cache in a layout optimized for grouped-query attention, coalesced reads, and append-only decode.
- [ ] Add chunked prefill attention. Prompt processing needs batched q/k/v, RoPE, KV writes, and attention over many prompt tokens rather than repeatedly invoking single-token decode.
- [ ] Avoid computing expensive trigonometric RoPE values in every tiny dispatch when values can be precomputed, cached, approximated safely, or fused into the projection path.
- [ ] Validate full-attention gate handling and q/gate tensor splitting against the GGUF tensor layout. A silent layout mismatch here changes model semantics while still producing plausible text.

#### 9. MRoPE and Qwen-specific model semantics

- [ ] Implement exact Qwen3-Next MRoPE semantics. The current CPU and shader paths rotate contiguous pairs with a simplified position model; they need to match the model's rotary dimensions, sectioning, and interleaving rules.
- [ ] Add tests that compare MRoPE output vectors against a trusted reference for multiple positions, dimensions, and heads.
- [ ] Confirm every tensor name and shape assumption in the loader against the target GGUF. Hardcoded names and dimensions must fail loudly when a model variant differs.
- [ ] Validate RMSNorm epsilon, attention scaling, gated attention, residual ordering, and final norm/logit projection against the reference architecture.

#### 10. DeltaNet and linear-attention path

- [ ] Prove the DeltaNet recurrence against a reference implementation. State shape, key/value head mapping, alpha/beta usage, decay/gate semantics, and normalization order need layer-level validation.
- [ ] Revisit DeltaNet state layout for GPU memory coalescing. The state is large and updated every token, so layout has direct decode-throughput impact.
- [ ] Fuse q/k/v projection post-processing, short convolution, L2 normalization, DeltaNet recurrence, norm/gate, and output projection preparation where practical. The current path uses many small kernels around one recurrence.
- [ ] Keep convolution state, DeltaNet state, and intermediate q/k/v/gate vectors entirely on GPU during decode.
- [ ] Add chunked prefill support for linear-attention layers. Replaying the single-token recurrence for every prompt token will not reach competitive prompt throughput.
- [ ] Validate mixed full-attention and linear-attention layer scheduling so state updates, KV updates, and residuals occur in exactly the reference order.

#### 11. Prefill architecture

- [ ] Replace token-by-token prefill with chunked prompt processing. The code currently runs the full decode path once per prompt token, which cannot compete with llama.cpp prompt processing.
- [ ] Add separate prefill kernels and scheduling for batched embeddings, batched projections, batched MoE routing, batched expert execution, batched attention, and batched state updates.
- [ ] Expose and tune prefill chunk size. The existing chunk-size constant is not enough; runtime scheduling must choose chunk sizes based on memory, context length, and GPU capability.
- [ ] Keep prefill and decode benchmark numbers separate. A backend can improve token generation while still being far behind on prompt processing, and the roadmap needs to track both.

#### 12. Final logits and sampling

- [ ] Avoid full-vocabulary logits readback during normal generation. Logits should remain on GPU through final projection, penalties, filtering, and sampling unless the user explicitly asks for logits.
- [ ] Implement GPU-side repetition/presence/frequency penalties or define that the first competitive target is greedy-only. If non-greedy sampling remains CPU-side, readback will dominate generation.
- [ ] Add GPU top-k/top-p/min-p/temperature sampling with deterministic seeding and parity tests against CPU sampling rules where deterministic behavior is required.
- [ ] Add a small readback path for only the selected token, probability, and optional diagnostics.
- [ ] Keep a debug mode that can read full logits for validation, but ensure benchmarks do not accidentally use it.

#### 13. Benchmarking and performance targets

- [ ] Replace `clock()` timing with monotonic wall-clock timing and Vulkan timestamp queries. CPU process time is not a valid measure for GPU throughput.
- [ ] Add warmup runs, repeated trials, median/p95 reporting, tokens/sec split by prompt-processing and token-generation phases, and fixed random seeds.
- [ ] Benchmark against llama.cpp with the same model file, quantization, prompt, context length, thread count, GPU backend, batch size, sampling settings, and power/performance mode.
- [ ] Record hardware details: GPU name, driver version, Vulkan version, subgroup size, memory type used for weights, CPU model, RAM, OS, compiler, shader compiler, and build flags.
- [ ] Add regression thresholds for correctness and performance. Each optimization should show whether it improves decode, prefill, memory traffic, dispatch count, or CPU overhead.
- [ ] Count dispatches per generated token and per prompt token. This should become a tracked metric because the current design issues too many tiny dispatches.
- [ ] Count CPU-GPU synchronization points per token. The target for normal decode should be one final synchronization or less per token, not one per kernel.

#### 14. Build, documentation, and maintainability

- [ ] Update README and architecture documentation to match the actual shader set, pipeline-cache behavior, current Vulkan path, and known limitations. The docs still describe older shader counts and older pipeline-cache status.
- [ ] Separate experimental scaffolding from supported paths. If a shader or quant format is present but not correctness-proven, mark it experimental and exclude it from performance claims.
- [ ] Add build modes for debug validation, profiling, release benchmarking, and shader development. These modes need explicit compiler flags, validation-layer settings, and profiling output.
- [ ] Keep generated or table-heavy files documented with their source and regeneration process. Large quant lookup tables should not become unverifiable magic constants.
- [ ] Add a portability matrix for integrated GPUs, discrete GPUs, and CPU fallback. The best memory and dispatch strategy differs significantly between UMA and discrete systems.

#### 15. Recommended execution order

- [ ] Stage A: fix generation correctness, tokenizer/stop-token parity, MRoPE semantics, DeltaNet semantics, and quant row validation. Do not optimize aggressively before this stage is green.
- [ ] Stage B: eliminate accidental readbacks and per-kernel waits in decode. Keep all layer internals on GPU and return only the selected token for greedy generation.
- [ ] Stage C: move weights and recurrent state to device-local memory with staging uploads and a real allocator.
- [ ] Stage D: replace generic quant matvec kernels with optimized per-format kernels, starting with the quant formats and tensor shapes that dominate the target GGUF.
- [ ] Stage E: implement fused or grouped MoE execution so each layer does not launch separate kernels for every selected expert operation.
- [ ] Stage F: implement long-context tiled attention and exact MRoPE in GPU kernels.
- [ ] Stage G: add chunked prefill. Only after this stage can prompt-processing numbers be compared fairly with llama.cpp.
- [ ] Stage H: implement GPU-side non-greedy sampling and penalties if the goal includes matching llama.cpp sampling features, not only greedy throughput.
- [ ] Stage I: run a controlled llama.cpp comparison matrix and update performance targets based on measured gaps rather than assumptions.

---

## Phase 4: Server + Tooling

- [ ] OpenAI-compatible `qw6-server`
- [ ] `/v1/chat/completions` and `/v1/completions`
- [ ] `/v1/models`
- [ ] Streaming and non-streaming responses
- [ ] Thinking/non-thinking modes
- [ ] Tool calling support
- [ ] Session management and prompt caching
- [ ] GGUF conversion and quantization tools
- [ ] Model download helper
- [ ] Quality testing suite

---

## Phase 5: MTP Speculative Decoding

- [ ] MTP draft head loading and forward pass
- [ ] Speculative decoding loop
- [ ] Identical greedy-output validation
- [ ] Speedup measurement

---

## Phase 6: Agent (Post-V1)

- [ ] `qw6-agent.c`
- [ ] Tool calling integration
- [ ] Long-session context management
- [ ] OpenAI API compatibility for coding-agent clients

---

## Phase 7: Multimodal (Future)

- [ ] Vision encoder
- [ ] Image token processing
- [ ] 3D MRoPE for vision/video
- [ ] Video token processing

---

## Summary

| Phase | Focus | Status | Key Deliverable |
|---|---|---|---|
| 0 | Research & design | Done | Design docs |
| 1 | CPU reference | In progress | Correct native logits (needs ref validation) |
| 2 | Vulkan backend | Partial / active | Runs on BC-250, not yet proven correct or fast |
| 3 | Correctness + optimization | Planned | llama.cpp parity, then tuned Vulkan backend |
| 4 | Server + tooling | Not started | OpenAI API server |
| 5 | MTP | Not started | Speculative decoding |
| 6 | Agent | Future | Coding agent |
| 7 | Multimodal | Future | Vision/video support |
