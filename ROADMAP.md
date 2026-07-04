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

## Phase 1: CPU Reference Path (Current)

Goal: correct text inference on CPU first. Speed is secondary until the local
forward pass can be checked against reference logits.

- [x] GGUF v3 metadata parser and model validator
- [x] `mmap`-backed GGUF tensor indexing
- [x] Unsloth/llama.cpp Qwen3.6 layout support (`qwen35moe`, `blk.*`)
- [x] packed routed-expert tensors split into per-expert views
- [x] tokenizer encode/decode regression (`make test-tokenizer`)
- [x] native dequant: F32, F16, BF16, Q4_K, Q5_K, Q6_K
- [x] native dequant: IQ2_XXS, IQ2_S, IQ3_S routed-expert formats
- [x] generic native tensor MatVec for dequantized rows
- [x] CPU kernels/probes:
  - [x] RMSNorm
  - [x] MatMul / MatVec scaffolding
  - [x] causal Conv1d reference kernel
  - [x] Gated DeltaNet update/retrieval scaffolding
  - [x] MoE routing top-8 from 256
  - [x] shared expert FFN probe (gate/up/down)
  - [x] argmax / greedy sampling primitive
  - [x] MRoPE partial rotary reference kernel
  - [x] Gated Attention / GQA reference kernel
  - [x] attention QKV projection probe
  - [x] routed expert FFN gather
  - [x] full Gated DeltaNet layer-0 forward probe
  - [x] full 40-layer forward to logits (CPU smoke path)
  - [x] MTP draft layer reference kernel
- [x] `qw6 --dump-tokens`
- [x] `qw6 --load-only` native GGUF validation/dequant probe
- [x] greedy native token generation smoke test (`./qw6 -m ... -p "Hello" -n 2`)
- [ ] `qw6 --dump-logits` / `--dump-logprobs`
- [ ] official Qwen3.6-35B-A3B reference vectors
- [ ] token-by-token logit comparison regression
- [x] `make cpu` build target

**Current native probe path:** `./qw6 --load-only -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`
validates the GGUF, dequantizes real tensors, runs output/shared-expert MatVec,
routes layer 0 top-8 experts, runs routed/shared FFN probes, and executes a
single-token layer-0 Gated DeltaNet forward probe through `ssm_out`.

**Current native generation smoke path:** `./qw6 -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf -p "Hello" -n 2 --nothink`
runs the 40-layer CPU path with KV/DeltaNet/Conv state and greedy sampling.
Reference-logit parity is still pending.

**Deliverable:** `./qw6 -p "Hello" --cpu` produces correct output matching Qwen
reference logits.

---

## Phase 2: Vulkan Compute Backend

Goal: GPU-accelerated inference on BC-250 via Vulkan compute shaders. Match or
exceed llama.cpp Vulkan performance as baseline.

- [x] Vulkan device initialization
- [x] host-visible buffer allocation and compute dispatch self-test
- [x] SPIR-V shader build via `glslc`
- [x] `matvec_f32.comp` + layer-0 router probe against real Qwen weights
- [ ] `matmul_iq2.comp` for IQ2 routed experts
- [ ] `matmul_q4k.comp`, `matmul_q5k.comp`, `matmul_q6k.comp`
- [x] `rmsnorm_full.comp` single-vector RMSNorm
- [ ] `rmsnorm.comp` multi-workgroup/chunked RMSNorm
- [ ] `rope_mrope.comp`
- [ ] `attention_gqa.comp`
- [ ] `deltanet_conv1d.comp`
- [ ] `deltanet_update.comp`
- [ ] `deltanet_retrieve.comp`
- [ ] `moe_route.comp`
- [ ] `moe_gather.comp`
- [ ] `mtp_draft.comp`
- [ ] `argmax.comp` / `sampling.comp` (partial: single-dispatch scaffold)
- [ ] pipeline barriers and dispatch orchestration (partial: backend self-test only)
- [x] `make vulkan` build target
- [ ] performance benchmark vs llama.cpp Vulkan baseline

**Current Vulkan smoke path:** `make vulkan && ./qw6 --vulkan-self-test`
compiles SPIR-V shaders, selects the BC-250 RADV device, dispatches GPU
`vec_add`, `rmsnorm_full`, and `matvec_f32`, and validates results on the host.
`./qw6 --load-only --vulkan -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` additionally
runs layer-0 router MatVec on the GPU against real Qwen weights and compares it
to CPU (`max_diff ~= 7.6e-6`). Full model inference still falls back to CPU
kernels.

**Deliverable:** `./qw6 -p "Hello" --vulkan` with all 40 layers on GPU at or
above the current BC-250 llama.cpp baseline.

---

## Phase 3: Optimization

Goal: exceed the llama.cpp baseline. Target 150+ tok/s generation.

- [ ] Wave64-optimized tiling for GFX1013
- [ ] fused kernels for RMSNorm, residual, routing, and activation hot paths
- [ ] MRoPE fused into attention
- [ ] MoE expert locality and hot-expert caching
- [ ] IQ2 dequant fused into MatMul
- [ ] persistent kernel design
- [ ] chunked prefill optimization
- [ ] memory layout tuning
- [ ] `qw6-bench`
- [ ] roofline measurement and stall analysis

---

## Phase 4: Server + Tooling

- [ ] OpenAI-compatible `qw6-server`
- [ ] `/v1/chat/completions`
- [ ] `/v1/completions`
- [ ] `/v1/models`
- [ ] streaming and non-streaming responses
- [ ] thinking/non-thinking modes
- [ ] tool calling support
- [ ] session management and prompt caching
- [ ] GGUF conversion and quantization tools
- [ ] model download helper
- [ ] quality testing suite

---

## Phase 5: MTP Speculative Decoding

- [ ] MTP draft head loading and forward pass
- [ ] speculative decoding loop
- [ ] identical greedy-output validation
- [ ] speedup measurement

---

## Phase 6: Agent (Post-V1)

- [ ] `qw6-agent.c`
- [ ] tool calling integration
- [ ] long-session context management
- [ ] OpenAI API compatibility for coding-agent clients

---

## Phase 7: Multimodal (Future)

- [ ] vision encoder
- [ ] image token processing
- [ ] 3D MRoPE for vision/video
- [ ] video token processing

---

## Summary

| Phase | Focus | Status | Key Deliverable |
|---|---|---|---|
| 0 | Research & design | Done | Design docs |
| 1 | CPU reference | In progress | Correct native logits |
| 2 | Vulkan backend | Not started | GPU inference |
| 3 | Optimization | Not started | 150+ tok/s |
| 4 | Server + tooling | Not started | OpenAI API server |
| 5 | MTP | Not started | Speculative decoding |
| 6 | Agent | Future | Coding agent |
| 7 | Multimodal | Future | Vision/video support |
