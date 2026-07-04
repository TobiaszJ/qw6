# qw6

**A model-specific native inference engine for Qwen 3.6-35B-A3B on the AMD BC-250.**

`qw6` is a self-contained C inference engine — not a generic GGUF runner, not a
wrapper around another runtime. It is purpose-built for one model (Qwen 3.6-35B-A3B)
on one hardware platform (AMD BC-250 / GFX1013 / Vulkan), in the same spirit as
[antirez/ds4](https://github.com/antirez/ds4) is built for DeepSeek V4 on Apple Silicon.

---

## Why?

| | ds4 (DeepSeek V4) | qw6 (Qwen 3.6-35B-A3B) |
|---|---|---|
| **Model size (IQ2)** | ~45 GB | ~10 GB |
| **Active params** | ~21B | 3B |
| **Target RAM** | 96–512 GB | 16 GB unified |
| **Target hardware** | MacBook / DGX Spark | AMD BC-250 ($140 board) |
| **KV cache problem** | Complex (MLA + compressed + disk) | Trivial (30/40 layers are linear attention = O(1) state) |
| **SSD streaming** | Core feature (model > RAM) | Not needed (model fits in 16 GB) |
| **Backend** | Metal / CUDA / ROCm | Vulkan (RADV / GFX1013) |
| **Current baseline perf** | ~27 tok/s (M3 Max 128GB) | ~78 tok/s (BC250 40-CU, llama.cpp) |
| **Optimisation goal** | — | 150–250 tok/s (roofline: 357 GB/s ÷ ~1.4 GB/token) |

Qwen 3.6-35B-A3B is a **better fit** for a model-specific engine than DeepSeek V4:

- **35B total, 3B active** — minimal compute per token
- **Hybrid attention** — 30 of 40 layers use Gated DeltaNet (linear attention, O(1) state); only 10 layers use Gated Attention (GQA, 2 KV heads). KV cache stays tiny even at 256K context.
- **Fits in 16 GB** at IQ2_M (~10 GB weights + KV + compute ≈ 13 GB) — no SSD streaming needed
- **MTP** — 1 multi-token-prediction layer for speculative decoding

The BC-250 community has solved the hard low-level problems: BIOS unlock, kernel
patches, TTM memory tuning, 40-CU unlock, oberon governor, Vulkan compute. This
project builds on top of that.

---

## Status

**Pre-alpha. Native CPU reference path in progress.**

`qw6` now performs real native work against Qwen 3.6 GGUF weights:

- GGUF v3 metadata parsing and model validation for Unsloth/llama.cpp Qwen3.6 layouts (`qwen35moe`)
- `mmap`-backed weight access with tensor offsets, shapes, quant types, and byte spans
- packed routed-expert tensors split into per-expert views
- tokenizer loading and token dump path
- CPU kernels and probes for RMSNorm, top-k MoE routing, routed/shared-expert FFN, and native matvec
- native dequantization for F32, F16, BF16, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, and IQ3_S
- layer-0 Gated DeltaNet single-token forward probe through SSM conv, GDN update, gated norm, and `ssm_out`
- 40-layer CPU greedy generation smoke path
- Phase 2 Vulkan runtime smoke path with SPIR-V shader build, GPU `rmsnorm_full`,
  GPU `matvec_f32`, and a layer-0 router probe against real Qwen weights

The repository currently contains CPU engine code, tokenizer implementation,
native GGUF loader/indexing, kernel scaffolding, and design documentation:

- [ARCHITECTURE.md](ARCHITECTURE.md) — Engine design (Vulkan compute pipeline, MoE routing, Gated DeltaNet, Gated Attention, MTP, server API)
- [MODEL_CARD.md](MODEL_CARD.md) — Qwen 3.6-35B-A3B specifications, tensor layout, quantisation plan
- [BC250_SETUP.md](BC250_SETUP.md) — Hardware setup guide (Vulkan, TTM, 40-CU unlock, oberon, memory)
- [ROADMAP.md](ROADMAP.md) — Development phases and milestones
- [CONTRIBUTING.md](CONTRIBUTING.md) — Build, test, regression protocol

The CPU scaffolding can be checked without BC-250 hardware, model weights, or
tokenizer test vectors:

```bash
make test        # builds qw6 and runs ./qw6 --self-test
./qw6 --inspect-gguf model.gguf  # inspect a GGUF header without loading tensors
```

With real Qwen3.6 GGUF weights available, the native loader/dequant probes can
be run with:

```bash
./qw6 --load-only -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
./qw6 -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf -p "Hello" -n 2 --nothink
make vulkan && ./qw6 --vulkan-self-test
./qw6 --load-only --vulkan -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
```

Expected current behavior: model metadata validation passes and native probes
print tensor checksums, router top-k experts, routed/shared-FFN summaries, and a
layer-0 DeltaNet forward checksum. The generation command runs a native
40-layer CPU smoke path and prints greedy token IDs plus decoded text; reference
logit parity and performance work are still pending. The Vulkan self-test
selects the BC-250 RADV device and validates real compute dispatches; the
`--load-only --vulkan` path also compares GPU layer-0 router MatVec against CPU.
Full model inference still uses CPU fallback kernels when `--vulkan` is
requested.

## Acknowledgements

- **[antirez/ds4](https://github.com/antirez/ds4)** — the architectural blueprint for a model-specific C inference engine
- **[akandr/bc250](https://github.com/akandr/bc250)** — BC-250 LLM setup, benchmarks, roofline analysis, 40-CU A/B data
- **[elektricm/amd-bc250-docs](https://elektricm.github.io/amd-bc250-docs/)** — comprehensive BC-250 documentation
- **[duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock)** — community kernel patch re-enabling all 40 CUs
- **[QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6)** — the model
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF format, quantisation kernels, Vulkan backend reference

## License

MIT — see [LICENSE](LICENSE).
