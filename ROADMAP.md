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
- [ ] `qw6 --dump-logits` / `--dump-logprobs`
- [ ] correct logit parity vs reference (requires test vectors from RTX 3090 server)

**Note:** Phase 1 was never validated against reference logits. Both CPU and GPU
paths may produce incorrect output. Core inference math needs comparison against
llama.cpp outputs layer-by-layer.

---

## Phase 2: Vulkan Compute Backend ✅

Goal: GPU-accelerated inference on BC-250 via Vulkan compute shaders.

### GPU Shaders (18 + 1)
- [x] `matvec_f32.comp` — FP32 matrix-vector multiply
- [x] `matmul_q4k.comp` — Q4_K block dequant + matmul
- [x] `matmul_q5k.comp` — Q5_K block dequant + matmul (known issue: produces wrong output, pre-dequantize to FP32 workaround)
- [x] `matmul_q6k.comp` — Q6_K block dequant + matmul
- [x] `matmul_iq2xxs.comp` — IQ2_XXS block dequant + matmul (verified 3.5e-6 vs CPU)
- [x] `add.comp` — element-wise vector add
- [x] `silu_mul.comp` — SiLU activation + multiply
- [x] `rmsnorm.comp` / `rmsnorm_full.comp` / `rmsnorm_apply.comp` — RMSNorm
- [x] `rope_mrope.comp` — MRoPE rotary position encoding
- [x] `attention_gqa.comp` — GQA attention (short-context)
- [x] `deltanet_conv1d.comp` / `deltanet_update.comp` / `deltanet_retrieve.comp` — Gated DeltaNet
- [x] `moe_route.comp` — top-k MoE routing
- [x] `moe_gather.comp` — expert output gather
- [x] `mtp_draft.comp` — MTP speculative decoding draft
- [x] `argmax.comp` / `sampling.comp` — token sampling

### Pipeline Infrastructure
- [x] Vulkan device init, buffer allocation, SPIR-V build via `glslc`
- [x] Big weight buffer: 10.7 GB upload of all model weights at init
- [x] Persistent scratch buffers for intermediate activations
- [x] Per-layer KV cache (full-attn) and DeltaNet state (linear-attn) on GPU
- [x] Full 40-layer dispatch orchestration in `qw6_vk_pipe_forward()`
- [x] CPU ops in pipeline: RMSNorm per head, Q/K norms, attention gate, small matmuls
- [x] Chat template: automatic CHATML wrapping for Qwen conversation format
- [x] Pipeline caching infrastructure (VkPipeline reuse; disabled — GPU hang)
- [x] attn_o workaround: pre-dequantize Q5_K→FP32 on CPU during init

### Known Issues
- [ ] `matmul_q5k.spv` produces 100-500x wrong output (byte-identical data)
- [ ] IQ2_S / IQ3_S expert weights lack GPU shaders (CPU fallback)
- [ ] Pipeline caching GPU hang (descriptor/push constant mismatch)
- [ ] Multi-token prefill processes one token at a time (~28s/token)
- [ ] Core inference math not validated against reference output

### Verification
- `./qw6 --vulkan-self-test` — all individual shaders verified on BC-250
- `./qw6 --load-only --vulkan -m model.gguf` — IQ2 probe matches CPU within 3.5e-6
- `./qw6 -m model.gguf -p "Hi" -n 1 --nothink --raw --vulkan` — full pipeline test

---

## Phase 3: Optimization

Goal: exceed the llama.cpp baseline. Target 150+ tok/s generation.

- [ ] Fix pipeline caching (VkPipeline reuse, descriptor pooling)
- [ ] Wire up GPU attention, conv1d, DeltaNet, MoE gather dispatches
- [ ] Fix Q5_K matmul shader (port dequant logic from llama.cpp)
- [ ] Add IQ2_S / IQ3_S GPU shaders (port from llama.cpp)
- [ ] Chunked prefill: batch N tokens through forward pass
- [ ] Wave64-optimized tiling for GFX1013
- [ ] Fused kernels (RMSNorm+attention, MRoPE+attention, IQ2 dequant+matmul)
- [ ] Persistent kernel design (reduce per-dispatch overhead)
- [ ] Memory layout tuning for GPU cache hierarchy
- [ ] `qw6-bench` and roofline measurement

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
| 2 | Vulkan backend | **Active** | GPU dispatch pipeline on BC-250 |
| 3 | Optimization | Not started | 150+ tok/s |
| 4 | Server + tooling | Not started | OpenAI API server |
| 5 | MTP | Not started | Speculative decoding |
| 6 | Agent | Future | Coding agent |
| 7 | Multimodal | Future | Vision/video support |
