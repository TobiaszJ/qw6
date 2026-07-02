# qw6 Architecture

This document describes the design of the `qw6` inference engine: a self-contained
C program that runs Qwen 3.6-35B-A3B on an AMD BC-250 using Vulkan compute.

---

## 1. Model Architecture: Qwen 3.6-35B-A3B

### 1.1 Overview

Qwen 3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`) is a natively multimodal
MoE model with a **hybrid attention** stack:

| Property | Value |
|---|---|
| Architecture | `qwen3_5_moe` |
| Total parameters | 35B |
| Active parameters per token | 3B (8 routed experts + 1 shared expert) |
| Hidden size | 2048 |
| Layers | 40 |
| Full attention interval | Every 4th layer (10 full-attn, 30 linear-attn) |
| Attention heads | 16 (GQA), KV heads: 2 |
| Head dimension | 256 |
| Experts | 256 routed, 8 active per token |
| Shared expert | 1 (always active) |
| MoE intermediate size | 512 (per expert) |
| Shared expert intermediate size | 512 |
| Vocabulary | 248,320 |
| Max position embeddings | 262,144 (256K) |
| MTP layers | 1 (speculative decoding) |

### 1.2 Layer Stack (40 layers)

```
Layer 0:  linear_attention  (Gated DeltaNet)
Layer 1:  linear_attention  (Gated DeltaNet)
Layer 2:  linear_attention  (Gated DeltaNet)
Layer 3:  full_attention    (Gated Attention / GQA)
Layer 4:  linear_attention  (Gated DeltaNet)
...
Layer 39: full_attention    (Gated Attention / GQA)
```

Pattern: 3× linear attention, 1× full attention, repeat. 10 full-attention layers
total (every 4th layer, 0-indexed: layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39).

### 1.3 Gated DeltaNet (Linear Attention)

Used in 30 of 40 layers. This is the key innovation that makes qw6 viable on a
16 GB board:

- **O(1) recurrent state** — no KV cache growth with context length
- **Gated DeltaNet** = delta-rule update + gating mechanism
- Per-layer state: [16 key heads × 128 key_dim, 32 value heads × 128 val_dim]
  = fixed-size tensor, independent of sequence length
- Requires `causal_conv1d` (short convolution, kernel_size=4) before the delta update
- Memory bandwidth per token: read state + state delta, write new state → small

**Implication for qw6:** No KV cache disk persistence, no SSD streaming, no compressed
KV. The 30 linear-attention layers contribute zero KV cache growth. The entire
"KV cache problem" that dominates ds4's complexity simply does not exist here.

### 1.4 Gated Attention (Full Attention / GQA)

Used in 10 of 40 layers:

- **GQA** with 16 query heads, 2 KV heads, head_dim=256
- **Gated** — attention output multiplied by a gate before residual
- **MRoPE** — multimodal rotary positional embedding:
  - Head dimension split into 3 components: [11, 11, 10] (temporal, height, width)
  - `partial_rotary_factor = 0.25` (only 25% of each head dimension gets RoPE)
  - `mrope_interleaved = true`
- **KV cache**: only 10 layers × 2 KV heads × 256 head_dim × 2 (K+V) × seq_len tokens

KV cache size at 256K context, Q4_0 quantised:
```
10 layers × 2 heads × 256 dim × 2 (KV) × 262144 tokens × 0.5 bytes (Q4_0)
= 10 × 2 × 256 × 2 × 262144 × 0.5
= 1.34 GB
```

At 64K context (practical ceiling for interactive use):
```
= 0.34 GB
```

This is **negligible** compared to the 10 GB model weights.

### 1.5 MoE Block

Each layer's FFN is replaced by a Mixture-of-Experts block:

- **256 routed experts** (only 8 activated per token via top-k router)
- **1 shared expert** (always active, processes every token)
- Per-expert intermediate size: 512 (3 linear layers: gate, up, down — SiLU activation)
- Router: linear projection hidden_size → 256, then top-8 selection + softmax
- Auxiliary load-balancing loss (coef=0.001, training-only)

**Per-token compute:**
- 8 routed experts × 3 × (2048 × 512) = 8 × 3,145,728 = 25.2M FLOPs (gate+up)
- 8 routed experts × 3 × (512 × 2048) = 25.2M FLOPs (down)
- 1 shared expert × same shape = 6.3M FLOPs
- Total MoE: ~56.7M FLOPs per layer, × 40 layers = ~2.27 GFLOPS per token

**Per-token weight read:**
- 8 active experts × (2048×512 + 512×512 + 512×2048) × 2.5 bits (IQ2_M)
  = 8 × (1,048,576 + 262,144 + 1,048,576) × 0.3125 bytes
  = 8 × 2,359,296 × 0.3125 = 5.9 MB
- Shared expert: 0.74 MB
- Attention weights (per layer): ~0.5 MB (IQ2_M)
- Total per token: ~6.6 MB × 40 layers = ~264 MB ≈ **0.26 GB per token of weight reads**

At 357 GB/s bandwidth (measured on BC-250):
```
Theoretical max generation: 357 / 0.26 ≈ 1,373 tok/s
```

Realistic with overhead, routing, activation: **150–300 tok/s** target.

### 1.6 Multi-Token Prediction (MTP)

1 MTP layer for speculative decoding:
- Predicts multiple future tokens in one forward pass
- `mtp_num_hidden_layers = 1`
- `mtp_use_dedicated_embeddings = false` (reuses main embeddings)
- Compatible with ds4-style speculative decoding path

### 1.7 Tokenizer

- BPE-based, vocab 248,320
- Special tokens for chat template (ChatML-style with Qwen extensions):
  - `system`, `user`, `assistant`, `tool` roles
  - Thinking/non-thinking modes (preserve_thinking)
  - Tool calling (XML-based tool call format)
- Image and video tokens (multimodal — text-only path can ignore these)

### 1.8 Multimodal (Future)

- Vision encoder (Qwen3-VL based)
- MRoPE for spatial/temporal position encoding of vision tokens
- **Text-only V1** — vision support is a post-V1 goal

---

## 2. Hardware Architecture: AMD BC-250

### 2.1 Specs

| Component | Details |
|---|---|
| CPU | Zen 2, 6c/12t |
| GPU | Cyan Skillfish, GFX1013 ("RDNA 1.5") |
| Compute Units | 24 stock / 40 with community unlock |
| Shader Processors | 1536 stock / 2560 unlocked |
| Memory | 16 GB GDDR6, unified (CPU+GPU), 256-bit bus |
| Measured bandwidth | 357 GB/s (streaming) |
| Measured FP32 | 3,901 GFLOPS (40-CU) |
| Roofline ridge point | 10.9 FLOP/byte |
| Storage | NVMe (475 GB typical) |
| TDP | 220 W nameplate, ~52–116 W during inference |

### 2.2 Compute Backend: Vulkan (RADV)

- **Only viable GPU compute path on GFX1013**
  - ROCm: `rocblas_abort()` — no GFX1013 solution libraries
  - OpenCL/rusticl: not functional
  - AMDVLK: does not support GFX1013
  - AMDGPU-PRO: does not support BC-250
- **RADV** (Mesa 25.1+) works. Version 25.3.4 confirmed (Vulkan 1.4.328)
- Vulkan 1.4 with full compute shader support
- Shaders: SPIR-V (compiled from GLSL or hand-written)
- Shader cache: RADV caches compiled SPIR-V to disk (fast subsequent launches)

### 2.3 Memory Architecture

16 GB unified GDDR6 pool. After tuning:

```
VRAM carveout: 512 MB (BIOS-reserved)
GTT:           16 GiB (ttm.pages_limit=4194304)
Vulkan heaps:  16.5 GiB (two overlapping views of same pool)
```

**Memory budget for qw6 (14 GB available, headless OS):**

```
IQ2_M experts (34 layers):       10.47 GB
IQ3_XXS experts (6 tail layers):  1.86 GB
Shared expert (Q4_K_M):           0.08 GB
Full attention (Q4_K_M):          0.11 GB
Linear attention (Q4_K_M):        0.53 GB
Embeddings (Q4_K_M):              0.31 GB
Output projection (Q8_0):         0.54 GB
Router (Q8_0):                    0.02 GB
MTP (Q4_K_M):                     0.50 GB
Norms + conv1d (FP32/FP16):      <0.01 GB
─────────────────────────────────────────────
Total weights:                  14.42 GB (primary: 12.56 GB with IQ2_M only)

DeltaNet state (FP16):            0.50 GB  (fixed, 30 layers)
KV cache Q4_0 @ 64K:              0.34 GB  (10 full-attn layers only)
Compute scratch:                  0.50 GB
─────────────────────────────────────────────
Total overhead:                   1.34 GB
Total (primary):                 13.90 GB
Headroom:                         +0.10 GB
```

Fits in 14 GB. No SSD streaming, no disk KV cache. Tight but viable.

### 2.4 40-CU Unlock

- Factory firmware masks 16 of 40 physical CUs (24 active)
- Community patch by S. Duggan re-enables all 40
- Measured speedup: **1.32× median generation, 1.50× prefill** (n=3, 11 models)
- Qwen3.6-35B-A3B specifically: 59.6 → 78.0 tok/s (+31%)
- Required for qw6's performance targets

---

## 3. Engine Design

### 3.1 Overview

```
┌──────────────────────────────────────────────────────────┐
│                      qw6 (qw6.c)                          │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Tokenizer  │  │ Model Loader │  │  Server (HTTP)   │ │
│  │  (BPE)      │  │  (GGUF)      │  │  (OpenAI API)    │ │
│  └──────┬──────┘  └──────┬───────┘  └────┬─────────────┘ │
│         │                │                │                │
│         │    ┌───────────▼────────────┐   │               │
│         │    │   Inference Loop        │   │               │
│         │    │  ┌─────────────────┐    │   │               │
│         ├───►│  │ Prefill (chunk) │    │◄──┤               │
│         │    │  └───────┬─────────┘    │   │               │
│         │    │  ┌───────▼─────────┐    │   │               │
│         │    │  │ Generate (tok)  │    │   │               │
│         │    │  └───────┬─────────┘    │   │               │
│         │    │  ┌───────▼─────────┐    │   │               │
│         │    │  │ MTP (speculative)│   │   │               │
│         │    │  └──────────────────┘   │   │               │
│         │    └─────────────────────────┘   │               │
│         │                │                │                │
│  ┌──────▼──────────────▼────────────────▼──────────────┐  │
│  │            Vulkan Compute Backend                   │  │
│  │  ┌────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐   │  │
│  │  │ MatMul │ │ Attention│ │DeltaNet│ │ MoE Route │   │  │
│  │  │(IQ2_M) │ │ (GQA)    │ │(conv1d)│ │ (256→8)   │   │  │
│  │  └────────┘ └──────────┘ └────────┘ └───────────┘   │  │
│  │  ┌────────┐ ┌──────────┐ ┌────────┐                 │  │
│  │  │ RMSNorm│ │  MRoPE   │ │  MTP   │                 │  │
│  │  └────────┘ └──────────┘ └────────┘                 │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Module Breakdown

| Module | File | Description |
|---|---|---|
| Core engine | `qw6.c` | Single-file inference engine (ds4-style) |
| Server | `qw6-server.c` | OpenAI-compatible HTTP server |
| Agent | `qw6-agent.c` | Coding agent (post-V1) |
| Eval | `qw6-eval.c` | Test-vector extractor |
| Bench | `qw6-bench.c` | Frontier-based benchmarking |
| Test runner | `qw6_test.c` | Regression tests |
| Vulkan kernels | `vulkan/*.comp` | SPIR-V compute shaders |
| GGUF tools | `gguf-tools/` | Conversion, quantisation, imatrix |
| Test vectors | `tests/test-vectors/` | Official Qwen API logits |

### 3.3 Vulkan Compute Pipeline

Unlike Metal (ds4's primary backend), Vulkan requires more boilerplate but offers
full control. The pipeline per compute dispatch:

```
1. Create / reuse VkPipeline (cached — RADV caches SPIR-V to disk)
2. Bind pipeline
3. Push constants / write descriptor sets (tensor addrs, dims)
4. Dispatch (workgroups × CU topology)
5. Memory barrier (ensure write-completion before next dispatch)
6. Read result buffer
```

**Key design decisions for GFX1013:**

- **Wavefront size: 64** (RDNA architecture). All kernels designed for wave64.
- **CU topology: 2 SE × 2 SH × 10 CU = 40 CUs** (after unlock). Dispatch workgroups
  to saturate all 40 CUs.
- **Shared memory (LDS):** 64 KB per CU. Use for tile-based MatMul.
- **Memory layout:** Tensors stored in row-major with tile-friendly alignment.
  IQ2_M dequantisation happens on-the-fly inside the MatMul kernel (no separate
  dequant pass).
- **Unified memory:** No CPU↔GPU copy needed. Tensors in GTT pages visible to both.
  Vulkan memory flags: `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` on the device-local heap.

### 3.4 Vulkan Kernels

| Kernel | Input | Output | Notes |
|---|---|---|---|
| `matmul_iq2m.comp` | IQ2_M weights, FP16/INT8 activations | FP16 output | On-the-fly dequant, tiled MatMul |
| `matmul_q4k.comp` | Q4_K weights (shared expert, attention) | FP16 output | Q4_K dequant + MatMul |
| `matmul_q8.comp` | Q8_0 weights (critical layers) | FP16 output | For attention/output if quality-sensitive |
| `attention_gqa.comp` | Q, K, V tensors | Attention output | GQA: 16 Q heads, 2 KV heads, MRoPE, gating |
| `deltanet_conv1d.comp` | Input, conv kernel (size 4) | Conv output | Causal short conv before delta update |
| `deltanet_update.comp` | State, key, value, delta | New state | Delta-rule recurrent update |
| `deltanet_retrieve.comp` | State, query | Retrieved value | State→output read |
| `moe_route.comp` | Router logits | Expert indices + weights | Top-8 selection from 256 experts |
| `moe_gather.comp` | Expert weights, activations | Expert outputs | Gather 8 routed + 1 shared |
| `rmsnorm.comp` | Input, weight, eps | Normalised output | RMSNorm with eps=1e-6 |
| `rope_mrope.comp` | Q/K tensors, positions | Rotated Q/K | MRoPE: split [11,11,10], interleave, partial 0.25 |
| `mtp_draft.comp` | Hidden state, embeddings | Draft tokens | 1-layer MTP for speculative decoding |
| `argmax.comp` | Logits | Token ID | Greedy / top-k / top-p sampling |
| `softmax.comp` | Logits / router logits | Probabilities | For sampling and MoE routing |

### 3.5 Inference Loop

#### Prefill (chunked)

```
for each chunk of N tokens (default N=2048):
    for each layer (0..39):
        if layer % 4 == 3:  # full attention
            # GatedDeltaNet not present — only Gated Attention
            Q = matmul(input, wq)       # [N, 16, 256]
            K = matmul(input, wk)       # [N, 2, 256] → repeat to 16
            V = matmul(input, wv)       # [N, 2, 256] → repeat to 16
            Q, K = mrope(Q, K, positions)
            attn_out = gqa_attention(Q, K, V, kv_cache_append)
            attn_out = attn_out * gate
            input = input + matmul(attn_out, wo)
        else:  # linear attention (Gated DeltaNet)
            conv_out = causal_conv1d(input, conv_kernel)
            key = matmul(conv_out, wkey)
            value = matmul(conv_out, wvalue)
            query = matmul(conv_out, wquery)
            key = layernorm(key)  # delta rule normalisation
            # Chunked parallel update: process all N tokens
            state = deltanet_chunk_update(state, key, value, query)
            delta_out = deltanet_retrieve(state, query)
            delta_out = delta_out * gate
            input = input + matmul(delta_out, wo)

        # MoE / FFN
        if input_layer_is_moe:
            router_logits = matmul(input, router_weight)
            expert_idx, expert_weights = moe_route(router_logits, top_k=8)
            moe_out = moe_gather(experts[expert_idx], input) * expert_weights
            shared_out = shared_expert(input)
            input = input + moe_out + shared_out

    logits = matmul(input, output_embedding)
```

#### Generation (1 token at a time)

Same as prefill but N=1. For linear-attention layers, the DeltaNet update is a
single-step recurrence (O(1) per token). For full-attention layers, standard
KV-cache append + attention.

#### MTP Speculative Decoding

```
1. Main model generates token T0, logits L0, hidden state H0
2. MTP layer: draft_token = MTP(H0, embeddings[T0]) → T1_draft
3. Verify T1_draft by running main forward with T1_draft:
   - If confirmed: accept, extend draft to T2 with MTP(H1, embed[T1])
   - If rejected: fall back to L0 argmax for T1
4. Repeat up to N speculative steps (default: 3-4)
```

### 3.6 Server API

OpenAI-compatible, same as ds4-server:

```
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
```

Chat template: Qwen ChatML format:
```xml
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

Thinking modes:
- `thinking: {"type": "disabled"}` → non-thinking
- `think: true` → thinking mode (default)
- `reasoning_effort: "max"` → think max (requires large context)

### 3.7 Quantisation Strategy (ds4-style asymmetric)

Following ds4's approach — different quantisation for different tensor classes.
Routed experts are 91% of all parameters but only 8/256 are active per token.
The critical components (shared expert, attention, embeddings, output, router)
process every token and are quality-sensitive, but they're tiny — keeping them
at Q4_K_M or Q8_0 costs almost nothing.

| Component | Quant | Bytes/param | Size | Rationale |
|---|---|---|---|---|
| Routed experts (main, 34 layers) | **IQ2_M** | 0.325 | 10.47 GB | 91% of model; only 8 active per token |
| Routed experts (tail, 6 layers) | **IQ3_XXS** | 0.383 | 1.86 GB | Last 6 layers are output-critical |
| Shared expert | **Q4_K_M** | 0.606 | 0.08 GB | Always active — quality-critical, tiny |
| Full attention (Q,K,V,O) | **Q4_K_M** | 0.606 | 0.11 GB | Quality-sensitive, tiny |
| Linear attention (DeltaNet) | **Q4_K_M** | 0.606 | 0.53 GB | State quality matters, tiny |
| Embeddings (input) | **Q4_K_M** | 0.606 | 0.31 GB | Large vocab, quality matters |
| Output projection (lm_head) | **Q8_0** | 1.063 | 0.54 GB | Token selection critical, untied |
| Router weights | **Q8_0** | 1.063 | 0.02 GB | Routing accuracy essential, tiny |
| MTP layer | **Q4_K_M** | 0.606 | 0.50 GB | Speculative decoding needs accuracy |
| RMSNorm weights | FP32 | 4.0 | <0.01 GB | No benefit to quantise, tiny |
| Conv1d kernels | FP16 | 2.0 | <0.01 GB | 4 elements per layer, tiny |
| DeltaNet state | FP16 | 2.0 | 0.50 GB | Fixed [16×128×32×128] per layer, capacity trade-off |

**Primary scenario: 12.56 GB weights + 1.34 GB overhead = 13.90 GB total.**
Fits in 14 GB (headless OS, ~2 GB system overhead) with 0.10 GB headroom.

IQ3 or Q4 for all experts does NOT fit — 256 experts × 40 layers = 32.2B params.
Even IQ3_XXS produces 12.34 GB for experts alone. The ds4-style asymmetric mix
is the only way to maximise quality while staying in memory.

### 3.8 GGUF Format

qw6 uses its own GGUF metadata layout, not compatible with generic loaders:

```
GGUF Header
├── metadata:
│   ├── qw6.version = 1
│   ├── qw6.architecture = "qwen3_5_moe"
│   ├── qw6.num_layers = 40
│   ├── qw6.full_attention_interval = 4
│   ├── qw6.num_experts = 256
│   ├── qw6.experts_per_tok = 8
│   ├── qw6.shared_expert = true
│   ├── qw6.mtp_layers = 1
│   └── quant metadata per tensor
├── tensors:
│   ├── tok_embeddings.weight          [Q8_0]
│   ├── layers.0.norm.weight           [FP32]
│   ├── layers.0.attn_q.weight         [Q4_K]
│   ├── layers.0.attn_k.weight         [Q4_K]
│   ├── layers.0.attn_v.weight         [Q4_K]
│   ├── layers.0.attn_o.weight         [Q4_K]
│   ├── layers.0.attn_gate.weight      [FP16]
│   ├── layers.0.conv1d.weight          [FP16]
│   ├── layers.0.deltanet_key.weight   [Q4_K]
│   ├── layers.0.deltanet_value.weight [Q4_K]
│   ├── layers.0.deltanet_query.weight [Q4_K]
│   ├── layers.0.deltanet_gate.weight  [FP16]
│   ├── layers.0.moe.router.weight     [Q8_0]
│   ├── layers.0.moe.experts.0..255.*  [IQ2_M]
│   ├── layers.0.moe.shared.*          [Q4_K]
│   ├── ... (layers 1..39)
│   └── mtp.layers.0.*                 [Q4_K]
```

---

## 4. What qw6 Does NOT Need (vs ds4)

| ds4 feature | Needed in qw6? | Why |
|---|---|---|
| SSD streaming | ❌ | Model fits in 16 GB |
| Disk KV cache | ❌ | 30/40 layers are linear attention (O(1) state); 10 attention layers' KV is tiny |
| Compressed MLA + ratio-4 indexer | ❌ | GQA is simple standard attention |
| Distributed inference | ❌ | Single board |
| Metal backend | ❌ | Not Apple hardware |
| CUDA backend | ❌ | Not NVIDIA hardware |
| ROCm backend | ❌ | Not supported on GFX1013 |
| Thinking-modes disk persistence | Later | Possible but low priority |

---

## 5. Performance Targets

### 5.1 Roofline Analysis

From bc250 community measurements:
- Peak streaming bandwidth: **357 GB/s**
- Peak FP32: 3,901 GFLOPS
- Ridge point: 10.9 FLOP/byte (all LLM decode is below → bandwidth-bound)

### 5.2 Generation Speed (estimates)

| Configuration | Weight reads/token | Roofline max | Target (40% eff) | Baseline (llama.cpp) |
|---|---|---|---|---|
| IQ2_XXS experts + Q4_K_M rest, 40-CU | 1.29 GB | 278 tok/s | ~111 tok/s | — |
| IQ2_M experts + Q4_K_M rest, 40-CU | 1.35 GB | 264 tok/s | ~106 tok/s | 78 tok/s |
| IQ2_XXS + IQ3_XXS tail, 40-CU | 1.33 GB | 268 tok/s | ~107 tok/s | — |

The gap between llama.cpp baseline (78 tok/s) and roofline (~264 tok/s) is the
optimisation headroom a model-specific engine can capture:

1. **On-the-fly dequant in MatMul kernel** — avoids separate dequant pass
2. **MoE expert locality** — keep hot experts in CU shared memory
3. **Fused layers** — RMSNorm + attention + residual in one dispatch
4. **MRoPE fused into attention** — no separate RoPE kernel
5. **DeltaNet single-step recurrence** — no attention matrix materialisation
6. **Wave64-optimised tiling** — match GFX1013 CU topology exactly

### 5.3 Prefill Speed

Prefill is more compute-bound (higher arithmetic intensity). Expected 200–400 tok/s
at 4K context with 40-CU.

### 5.4 Context Capacity

| Context | KV cache (Q4_0) | Total memory | Fits 16 GB? |
|---|---|---|---|
| 4K | 0.02 GB | ~10.5 GB | ✅ |
| 16K | 0.08 GB | ~10.6 GB | ✅ |
| 64K | 0.34 GB | ~10.9 GB | ✅ |
| 256K | 1.34 GB | ~11.9 GB | ✅ (tight with OS) |

All contexts fit in 16 GB — no SSD streaming, no disk KV, no context truncation.