# Model Card: Qwen 3.6-35B-A3B

This document specifies the exact model architecture, tensor layout, and
quantisation plan for the Qwen 3.6-35B-A3B weights used by qw6.

---

## 1. Source Model

| Property | Value |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` |
| HuggingFace | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| Architecture | `Qwen3_5MoeForConditionalGeneration` |
| Model type | `qwen3_5_moe` |
| License | Apache 2.0 |
| Release date | April 2026 |
| Training data | Interleaved text, image, and video tokens (natively multimodal) |

---

## 2. Architecture (from `config.json`)

### 2.1 Text Config

```json
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ],
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "shared_expert_intermediate_size": 512,
    "moe_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "max_position_embeddings": 262144,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
      "rope_theta": 10000000,
      "rope_type": "default",
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]
    }
  }
}
```

### 2.2 Layer Type Summary

- **40 layers total**
- **30 × `linear_attention`** (Gated DeltaNet) — layers 0,1,2, 4,5,6, 8,9,10, ...
- **10 × `full_attention`** (Gated Attention / GQA) — layers 3,7,11,15,19,23,27,31,35,39
- Pattern: 3:1 ratio, repeating every 4 layers
- Full attention layers are always at index `i % 4 == 3`

### 2.3 Linear Attention Parameters (Gated DeltaNet)

| Property | Value | Description |
|---|---|---|
| `linear_num_key_heads` | 16 | Key/State heads |
| `linear_key_head_dim` | 128 | Per-head key dimension |
| `linear_num_value_heads` | 32 | Value heads |
| `linear_value_head_dim` | 128 | Per-head value dimension |
| `linear_conv_kernel_dim` | 4 | Causal short-conv kernel size |
| `mamba_ssm_dtype` | float32 | State precision |

**State shape per layer:** [16, 128, 32, 128] — fixed, independent of sequence length.

**DeltaNet mechanism:**
1. **Causal conv1d** (kernel_size=4) on input → short-range mixing
2. Project to key, value, query, gate
3. **Delta-rule update:** `state += beta(key) ⊗ (value - state ⊗ key)` (per-head)
4. **Retrieve:** `output = state ⊗ query`
5. **Gate:** `output = output * sigmoid(gate)`
6. Normalisation: key is normalised (layernorm) before state update

### 2.4 Full Attention Parameters (Gated Attention / GQA)

| Property | Value | Description |
|---|---|---|
| `num_attention_heads` | 16 | Query heads |
| `num_key_value_heads` | 2 | KV heads (GQA ratio 8:1) |
| `head_dim` | 256 | Per-head dimension |
| `attn_output_gate` | true | Attention output is gated |
| `partial_rotary_factor` | 0.25 | Only first 25% of head dim gets RoPE |

**KV cache shape per layer:** [seq_len, 2, 256] — grows with sequence.

### 2.5 MoE Parameters

| Property | Value | Description |
|---|---|---|
| `num_experts` | 256 | Total routed experts |
| `num_experts_per_tok` | 8 | Active experts per token |
| `moe_intermediate_size` | 512 | Per-expert FFN intermediate size |
| `shared_expert_intermediate_size` | 512 | Shared expert intermediate size |
| `router_aux_loss_coef` | 0.001 | Load balancing (training only) |
| `output_router_logits` | false | Don't return router logits during inference |

**Expert FFN structure (per expert):**
1. **Gate projection:** [2048, 512] — SiLU activation
2. **Up projection:** [2048, 512] — element-wise multiply with gate
3. **Down projection:** [512, 2048] — output

**Shared expert:** same shape as one routed expert, always active.

### 2.6 MTP Parameters

| Property | Value | Description |
|---|---|---|
| `mtp_num_hidden_layers` | 1 | One MTP layer |
| `mtp_use_dedicated_embeddings` | false | Reuses main token embeddings |

### 2.7 MRoPE (Multimodal Rotary Position Embedding)

Rotary embeddings split the head dimension (256) into 3 components:
- **Temporal:** 11 dims (×2 for sin/cos = 22 of 256)
- **Height:** 11 dims (×2 = 22)
- **Width:** 10 dims (×2 = 20)
- **Total rotary:** 22 + 22 + 20 = 64 dims = 25% of 256 (matches `partial_rotary_factor`)
- **Non-rotary:** 256 - 64 = 192 dims (75%)
- **Interleaved:** sin/cos pairs are interleaved (not concatenated)
- `rope_theta = 10,000,000`

For text-only inference: all 3 components use the same 1D position (text positions
are scalar). Vision tokens would have 3D positions (frame × height × width).

### 2.8 Tokenizer

| Property | Value |
|---|---|
| Type | BPE |
| Vocab size | 248,320 |
| `bos_token_id` | 248,044 |
| `eos_token_id` | 248,044 |
| `image_token_id` | 248,056 |
| `video_token_id` | 248,057 |
| `pad_token_id` | null (no padding) |

Chat template: ChatML-style with Qwen extensions for thinking and tool calling.

---

## 3. Tensor Layout (qw6 GGUF)

### 3.0 Current Supported GGUF Layout

The current native loader supports the Unsloth/llama.cpp GGUF naming used by
`unsloth/Qwen3.6-35B-A3B-GGUF`:

```
general.architecture = "qwen35moe"

token_embd.weight
output.weight
output_norm.weight

blk.{i}.attn_norm.weight
blk.{i}.attn_qkv.weight
blk.{i}.attn_output.weight
blk.{i}.attn_gate.weight

blk.{i}.ssm_*.weight / blk.{i}.ssm_* parameters

blk.{i}.ffn_gate_inp.weight
blk.{i}.ffn_gate_exps.weight       # packed [hidden, inter, 256]
blk.{i}.ffn_up_exps.weight         # packed [hidden, inter, 256]
blk.{i}.ffn_down_exps.weight       # packed [inter, hidden, 256]
blk.{i}.ffn_gate_shexp.weight
blk.{i}.ffn_up_shexp.weight
blk.{i}.ffn_down_shexp.weight
```

The loader maps packed routed-expert tensors into 256 per-expert views without
copying. Native dequantization currently supports F32, F16, BF16, Q4_K, Q5_K,
and Q6_K. Routed expert IQ2/IQ3 formats are the next required step for full
native MoE inference.

### 3.1 Naming Convention

```
tok_embeddings.weight                    [vocab, hidden]

# Per-layer (0..39):
layers.{i}.norm.weight                   [hidden]  FP32

# Full attention layers (i % 4 == 3):
layers.{i}.attn_q.weight                 [num_heads * head_dim, hidden]
layers.{i}.attn_k.weight                 [kv_heads * head_dim, hidden]
layers.{i}.attn_v.weight                 [kv_heads * head_dim, hidden]
layers.{i}.attn_o.weight                 [num_heads * head_dim, hidden]
layers.{i}.attn_gate.weight              [hidden]  FP16 or FP32
layers.{i}.attn_norm.weight              [hidden]  FP32 (QK-norm if present)

# Linear attention layers (i % 4 != 3):
layers.{i}.conv1d.weight                 [conv_dim, hidden]  FP16
layers.{i}.deltanet_key.weight           [num_key_heads * key_head_dim, hidden]
layers.{i}.deltanet_value.weight         [num_value_heads * value_head_dim, hidden]
layers.{i}.deltanet_query.weight         [num_value_heads * value_head_dim, hidden]
layers.{i}.deltanet_gate.weight           [hidden]  FP16 or FP32
layers.{i}.deltanet_out.weight           [num_value_heads * value_head_dim, hidden]
layers.{i}.deltanet_norm.weight           [hidden]  FP32 (key normalisation)

# MoE (all layers):
layers.{i}.moe.router.weight              [num_experts, hidden]  Q8_0 or FP16
layers.{i}.moe.experts.{j}.gate_proj.weight  [moe_intermediate, hidden]  IQ2_M
layers.{i}.moe.experts.{j}.up_proj.weight    [moe_intermediate, hidden]  IQ2_M
layers.{i}.moe.experts.{j}.down_proj.weight  [hidden, moe_intermediate]  IQ2_M
layers.{i}.moe.shared.gate_proj.weight       [moe_intermediate, hidden]  Q4_K
layers.{i}.moe.shared.up_proj.weight         [moe_intermediate, hidden]  Q4_K
layers.{i}.moe.shared.down_proj.weight       [hidden, moe_intermediate]  Q4_K

# MTP:
mtp.layers.0.norm.weight                 [hidden]  FP32
mtp.layers.0.embed.weight                [vocab, hidden]  Q8_0 (or shared with main)
mtp.layers.0.{attn/deltanet/moe}.*       (same as main layers)

# Output:
output.weight                            [vocab, hidden]  Q8_0
```

### 3.2 Tensor Shapes (concrete)

| Tensor | Shape | Parameters |
|---|---|---|
| `tok_embeddings.weight` | [248320, 2048] | 509M |
| `layers.{i}.attn_q.weight` | [4096, 2048] | 8.4M (full-attn layers only) |
| `layers.{i}.attn_k.weight` | [512, 2048] | 1.05M |
| `layers.{i}.attn_v.weight` | [512, 2048] | 1.05M |
| `layers.{i}.attn_o.weight` | [4096, 2048] | 8.4M |
| `layers.{i}.deltanet_key.weight` | [2048, 2048] | 4.2M (line-attn layers only) |
| `layers.{i}.deltanet_value.weight` | [4096, 2048] | 8.4M |
| `layers.{i}.deltanet_query.weight` | [4096, 2048] | 8.4M |
| `layers.{i}.deltanet_out.weight` | [2048, 2048] | 4.2M |
| `layers.{i}.conv1d.weight` | [2048, 4] | 8K |
| `layers.{i}.moe.router.weight` | [256, 2048] | 524K |
| `layers.{i}.moe.experts.{j}.*` | 3× [512, 2048] + [2048, 512] | 2.36M per expert |
| `layers.{i}.moe.shared.*` | 3× [512, 2048] + [2048, 512] | 2.36M |
| `output.weight` | [248320, 2048] | 509M |
| `mtp.layers.0.*` | (one full layer) | ~varies |

## 3. Parameter Breakdown

| Component | Parameters | % of total |
|---|---|---|
| Routed experts (256 × 40 layers) | 32.21B | 91.4% |
| Linear attention (30 layers) | 881M | 2.5% |
| Embeddings + output (untied) | 1,018M | 2.9% |
| MTP layer | 827M | 2.3% |
| Full attention (10 layers) | 189M | 0.5% |
| Shared expert (40 layers) | 126M | 0.4% |
| Router (40 layers) | 21M | 0.06% |
| **Total** | **35.27B** | **100%** |

Routed experts dominate at 91%. This is where aggressive quantisation saves the
most memory. The critical components (router, shared expert, attention, embeddings,
MTP) are tiny by comparison — keeping them at higher precision costs almost nothing.

---

## 4. Quantisation Plan (ds4-style Asymmetric Mix)

### 4.1 Design Philosophy

Following ds4's approach: **quantise the big routed experts aggressively, leave
critical components at higher precision.** The routed experts are 91% of the model
but only 8 are active per token. The critical components (shared expert, attention,
embeddings, output, router) process every token and are quality-sensitive — but
they're tiny, so keeping them at Q4_K_M or Q8_0 costs almost nothing.

### 4.2 Recommended Quant Mixes (14 GB budget, headless OS)

With headless CachyOS using ~2 GB, ~14 GB are available for qw6.

| Scenario | Routed experts | Rest | Critical | Weights | Total | Headroom | Fit |
|---|---|---|---|---|---|---|---|
| **★ Primary** | IQ2_M | Q4_K_M | Q8_0 (emb/out/rtr) | 12.56 GB | 13.90 GB | +0.10 GB | ✅ |
| **★ Quality** | IQ2_M | Q4_K_S | Q8_0 | 12.40 GB | 13.74 GB | +0.26 GB | ✅ |
| **Balanced** | IQ2_XXS + IQ3_XXS tail(6) | Q4_K_M | Q8_0 | 10.74 GB | 12.09 GB | +1.91 GB | ✅ |
| **Compact** | IQ2_XXS | Q4_K_M | Q8_0 | 10.37 GB | 11.71 GB | +2.29 GB | ✅ |
| **Failed** | IQ3_XXS all | Q4_K_M | Q8_0 | ~13.7 GB | ~15.0 GB | -1.0 GB | ❌ |

**IQ3 / Q4 for all experts does NOT fit** in 14 GB. The 256 experts × 40 layers are
too many tensors — even IQ3_XXS (0.383 bytes/param) produces 12.34 GB for experts
alone, leaving no room for the rest.

### 4.3 Per-Component Quantisation

| Component | Quant | Bytes/param | Size | Rationale |
|---|---|---|---|---|
| Routed experts (all 40 layers) | **IQ2_M** | 0.325 | 10.47 GB | 91% of model; only 8 active per token |
| Shared expert | **Q6_K** | 0.729 | 0.09 GB | Always active — near-lossless quality, tiny |
| Full attention (Q,K,V,O) | **Q6_K** | 0.729 | 0.13 GB | Quality-sensitive, tiny |
| Linear attention (DeltaNet) | **Q6_K** | 0.729 | 0.64 GB | State quality matters, tiny |
| Embeddings (input) | **Q6_K** | 0.729 | 0.37 GB | Large vocab, near-lossless quality |
| Output projection (lm_head) | **Q6_K** | 0.729 | 0.37 GB | Token selection critical, untied |
| Router weights | **Q8_0** | 1.063 | 0.02 GB | Routing accuracy essential, tiny |
| MTP layer | **Q4_K_M** | 0.606 | 0.50 GB | Speculative decoding — good quality, smaller |
| RMSNorm weights | FP32 | 4.0 | <0.01 GB | No benefit to quantise, tiny |
| Conv1d kernels | FP16 | 2.0 | <0.01 GB | 4 elements per layer, tiny |
| DeltaNet state | FP16 | 2.0 | 0.50 GB | Fixed [16×128×32×128] per layer |

**Primary scenario: 12.60 GB weights + 1.34 GB overhead = 13.94 GB.**
Fits in 14 GB with +0.06 GB headroom. Q6_K for critical components costs only
+0.37 GB vs Q4_K_M but provides near-lossless quality for every component that
processes every token.

Alternative with more headroom: IQ2_XXS for experts (8.28 GB) + Q6_K for all
critical components = 11.85 GB total, +2.15 GB headroom. Trades expert quality
for more breathing room and higher-precision critical path.

IQ3/Q4 for all experts does NOT fit — 256 experts × 40 layers = 32.2B params.
Even IQ3_XXS produces 12.34 GB for experts alone. The ds4-style asymmetric mix
(aggressive expert quant + high-precision critical path) is the only way to
maximise quality while staying in memory.

### 4.4 Why IQ3/Q4 for All Experts Doesn't Fit

| Expert quant | Expert size | Total weights | 14 GB fit? |
|---|---|---|---|
| IQ2_XXS (0.257 B/param) | 8.28 GB | 10.37 GB | ✅ (+2.29) |
| IQ2_M (0.325 B/param) | 10.47 GB | 12.56 GB | ✅ (+0.10) |
| IQ3_XXS (0.383 B/param) | 12.34 GB | ~14.4 GB | ❌ (-0.4) |
| IQ3_M (0.465 B/param) | 14.98 GB | ~17.0 GB | ❌ |
| Q4_K_M (0.606 B/param) | 19.53 GB | ~21.5 GB | ❌ |

The 256 experts × 40 layers = 32.2B parameters dominate. Even IQ3_XXS at 2.5 bits
average produces 12.34 GB for experts alone, plus 2+ GB for everything else. There
is no room for IQ3 or higher on all experts in 14 GB.

**The ds4-style asymmetric approach is the only way to get expert quantisation
above IQ2 while fitting in memory:** keep 256 experts at IQ2_M, and use Q4_K_M for
the small critical components that process every token.

---

## 4. Quality Validation

### 4.1 Test Vectors

Official logits obtained from the Qwen 3.6-35B-A3B API (or HuggingFace
Transformers reference implementation) for:

- Short prompt continuation (greedy decoding, thinking disabled)
- Medium context (4K tokens)
- Long context (32K tokens) — tests linear-attention state management
- Tool calling format (XML-based tool calls)
- Thinking vs non-thinking mode

### 4.2 Regression Checks

```
./qw6 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
```

Compare local greedy continuation token-by-token against API test vectors.
Tokeniser, template, attention, and DeltaNet state regressions all surface
before they become long-generation failures.

### 4.3 Needle-in-Haystack

Fill context to 80% with real tokens (not padding), plant needles at different
depths. Verify retrieval at 4K, 16K, 32K, 64K. This tests long-context quality
of the linear-attention state.

---

## 5. Comparison with ds4's Model (DeepSeek V4 Flash)

| | DeepSeek V4 Flash | Qwen 3.6-35B-A3B |
|---|---|---|
| Total params | ~685B | 35B |
| Active params | ~21B | 3B |
| Layers | ~61 | 40 |
| Attention | MLA (compressed KV + ratio-4 indexer) | GQA (simple) + Gated DeltaNet (linear) |
| KV cache growth | Linear with context (mitigated by compression) | **Zero** for 30/40 layers; linear for 10/40 |
| Disk KV cache needed | Yes (large KV even compressed) | **No** (KV from 10 GQA layers is tiny) |
| SSD streaming needed | Yes (model > RAM) | **No** (IQ2_M fits in 16 GB) |
| MTP | Yes (multiple layers) | Yes (1 layer) |
| Multimodal | No (text-only) | Yes (vision + video, but text-only V1) |
| Experts | 256+ shared, very large experts | 256 + 1 shared, small experts |
