# qw6 Roadmap

Development phases for the Qwen 3.6-35B-A3B inference engine on AMD BC-250.

---

## Phase 0: Research & Design ✅ (Current)

- [x] Study ds4 architecture (antirez/ds4)
- [x] Study Qwen 3.6-35B-A3B architecture (config.json, HF transformers docs)
- [x] Study BC-250 / GFX1013 / Vulkan compute landscape (akandr/bc250, elektricm docs)
- [x] Write ARCHITECTURE.md
- [x] Write MODEL_CARD.md
- [x] Write BC250_SETUP.md
- [x] Create GitHub repo, set up project structure

**Deliverable:** This repository with complete design documentation.

---

## Phase 1: CPU Reference Path

Goal: Correct inference on CPU (no GPU). Validate model loading, tokenizer,
forward pass, and logits against official Qwen API test vectors.

- [ ] `qw6.c` — GGUF loader for qw6 tensor layout
- [ ] Tokenizer (BPE, vocab 248,320, ChatML template)
- [ ] CPU reference kernels:
  - [ ] MatMul (generic, for correctness checking only)
  - [ ] RMSNorm
  - [ ] MRoPE (partial rotary, interleaved, [11,11,10] sections)
  - [ ] Gated Attention (GQA, 16 Q heads, 2 KV heads, gating)
  - [ ] Causal Conv1d (kernel_size=4)
  - [ ] Gated DeltaNet (delta-rule state update + retrieval + gating)
  - [ ] MoE routing (top-8 from 256 + shared expert)
  - [ ] MoE expert gather
  - [ ] MTP draft layer
  - [ ] Argmax / sampling
- [ ] `qw6 --dump-logits` / `--dump-logprobs` / `--dump-tokens`
- [ ] Test vectors from official Qwen 3.6-35B-A3B API (greedy, thinking disabled)
- [ ] Token-by-token logit comparison regression test
- [ ] `make cpu` build target

**Deliverable:** `./qw6 -p "Hello" --cpu` produces correct output matching Qwen API.

**Estimated duration:** 4–6 weeks

---

## Phase 2: Vulkan Compute Backend

Goal: GPU-accelerated inference on BC-250 via Vulkan compute shaders. Match
or exceed llama.cpp Vulkan performance as baseline.

- [ ] Vulkan device initialisation (instance, physical device, queue family)
- [ ] Memory management (GTT-backed device-local buffers, unified memory)
- [ ] `matmul_iq2m.comp` — IQ2_M on-the-fly dequant + tiled MatMul
- [ ] `matmul_q4k.comp` — Q4_K dequant + MatMul (shared expert, attention)
- [ ] `matmul_q8.comp` — Q8_0 MatMul (embeddings, output, router)
- [ ] `rmsnorm.comp`
- [ ] `rope_mrope.comp` — MRoPE with partial rotary + interleaving
- [ ] `attention_gqa.comp` — GQA attention with KV cache
- [ ] `deltanet_conv1d.comp` — Causal short convolution
- [ ] `deltanet_update.comp` — Delta-rule state update
- [ ] `deltanet_retrieve.comp` — State → output retrieval
- [ ] `moe_route.comp` — Top-8 routing from 256 experts
- [ ] `moe_gather.comp` — Expert weight gather + SiLU FFN
- [ ] `mtp_draft.comp` — MTP speculative draft
- [ ] `argmax.comp` / `sampling.comp`
- [ ] Pipeline barriers and dispatch orchestration
- [ ] `make vulkan` build target
- [ ] Shader cache (RADV handles this, but verify)
- [ ] Performance benchmark vs llama.cpp Vulkan baseline (78 tok/s target match)

**Deliverable:** `./qw6 -p "Hello" --vulkan` at ≥78 tok/s (matching llama.cpp).

**Estimated duration:** 8–12 weeks

---

## Phase 3: Optimisation

Goal: Exceed llama.cpp baseline significantly. Target 150+ tok/s generation.

- [ ] Wave64-optimised tiling for GFX1013 (2 SE × 2 SH × 10 CU)
- [ ] Fused kernels (RMSNorm + attention + residual in one dispatch)
- [ ] MRoPE fused into attention kernel
- [ ] MoE expert locality — cache hot experts in CU shared memory (LDS)
- [ ] IQ2_M dequant fused into MatMul (no separate dequant pass)
- [ ] Persistent kernel design (reduce dispatch overhead)
- [ ] Chunked prefill optimisation (pipelined chunks across CUs)
- [ ] Memory layout tuning (tile alignment, bank-conflict avoidance)
- [ ] `qw6-bench` — frontier-based benchmarking (prefill + gen at context frontiers)
- [ ] Roofline measurement and stall analysis

**Deliverable:** `./qw6 --bench` showing ≥150 tok/s generation at 4K context.

**Estimated duration:** 4–8 weeks

---

## Phase 4: Server + Tooling

Goal: Production-ready server and model tooling.

- [ ] `qw6-server.c` — OpenAI-compatible HTTP server
  - [ ] `/v1/chat/completions` (streaming + non-streaming)
  - [ ] `/v1/completions`
  - [ ] `/v1/models`
  - [ ] Thinking/non-thinking modes
  - [ ] Tool calling (XML-based Qwen tool format)
- [ ] Session management (prompt caching, context continuation)
- [ ] `gguf-tools/convert.py` — HF safetensors → qw6 GGUF
- [ ] `gguf-tools/quantize.c` — IQ2_M, Q4_K, Q8_0, asymmetric mix
- [ ] `gguf-tools/imatrix/` — Calibration + imatrix for routed experts
- [ ] `download_model.sh` — fetch pre-built GGUFs from HuggingFace
- [ ] Quality testing suite (official logits comparison, needle-in-haystack)

**Deliverable:** `./qw6-server` serving OpenAI API with correct, fast inference.

**Estimated duration:** 4–6 weeks

---

## Phase 5: MTP Speculative Decoding

Goal: Multi-token-prediction speculative decoding for accelerated generation.

- [ ] MTP draft head loading and forward pass
- [ ] Speculative decoding loop (draft → verify → accept/reject)
- [ ] Correctness validation (must produce identical output to greedy)
- [ ] Speedup measurement (target: 1.3–2.0× over Phase 3 generation speed)

**Deliverable:** `./qw6 --mtp` at ≥200 tok/s with identical greedy output.

**Estimated duration:** 2–4 weeks

---

## Phase 6: Agent (Post-V1)

- [ ] `qw6-agent.c` — coding agent (ds4-agent analogue)
- [ ] Tool calling integration
- [ ] Context management for long agent sessions
- [ ] Claude Code / Aider compatibility (OpenAI API mode)

**Estimated duration:** 4–8 weeks

---

## Phase 7: Multimodal (Future)

- [ ] Vision encoder (Qwen3-VL based)
- [ ] Image token processing
- [ ] MRoPE 3D position encoding for vision tokens
- [ ] Video token processing

**Estimated duration:** Unknown, post-V1 research

---

## Summary

| Phase | Focus | Duration | Key Deliverable |
|---|---|---|---|
| 0 ✅ | Research & Design | Done | This repo |
| 1 | CPU Reference | 4–6 weeks | Correct inference, logit-matched |
| 2 | Vulkan Backend | 8–12 weeks | GPU inference ≥78 tok/s |
| 3 | Optimisation | 4–8 weeks | ≥150 tok/s generation |
| 4 | Server + Tooling | 4–6 weeks | OpenAI API server + GGUF tools |
| 5 | MTP | 2–4 weeks | Speculative decoding, ≥200 tok/s |
| 6 | Agent | 4–8 weeks | Coding agent |
| 7 | Multimodal | TBD | Vision support |

**Total to V1 (Phases 0–4):** ~20–32 weeks (5–8 months) for one developer with AI assistance.