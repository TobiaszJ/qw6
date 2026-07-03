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

**Pre-alpha. CPU reference scaffolding, tokenizer work, design, and documentation phase.**

Full inference is not functional yet. The repository currently contains early CPU engine code, a tokenizer implementation, CLI scaffolding, kernel stubs, and design documentation:

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

## Acknowledgements

- **[antirez/ds4](https://github.com/antirez/ds4)** — the architectural blueprint for a model-specific C inference engine
- **[akandr/bc250](https://github.com/akandr/bc250)** — BC-250 LLM setup, benchmarks, roofline analysis, 40-CU A/B data
- **[elektricm/amd-bc250-docs](https://elektricm.github.io/amd-bc250-docs/)** — comprehensive BC-250 documentation
- **[duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock)** — community kernel patch re-enabling all 40 CUs
- **[QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6)** — the model
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF format, quantisation kernels, Vulkan backend reference

## License

MIT — see [LICENSE](LICENSE).
