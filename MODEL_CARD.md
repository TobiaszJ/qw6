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

### 3.3 Size Estimates by Quantisation

| Quant mix | Routed experts | Shared/attn | Embed/output | Total |
|---|---|---|---|---|
| IQ2_M + Q4_K + Q8_0 | ~6.1 GB (256×40) | ~1.2 GB | ~1.0 GB | ~9.5 GB |
| IQ2_XXS + Q4_K + Q8_0 | ~4.9 GB | ~1.2 GB | ~1.0 GB | ~8.3 GB |
| Q4_K_M (all) | ~12.1 GB | ~1.2 GB | ~1.0 GB | ~14.3 GB |

**IQ2_M is the primary target** — best quality/capacity trade-off for 16 GB.

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