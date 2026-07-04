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

**Phase 2: GPU dispatch pipeline — active on BC-250.**

qw6 now performs real GPU-accelerated inference against Qwen 3.6 GGUF weights on
the BC-250 via Vulkan compute:

**GPU pipeline (Phase 2):**
- Full 40-layer inference orchestration with on-GPU matmul, RMSNorm, SiLU, and
  element-wise add operations
- 10.7 GB model weight upload to GPU-visible buffer (once at init)
- GPU matmul dispatches for FP32, Q4_K, Q5_K, Q6_K, IQ2_XXS quant formats
- Persistent KV cache and DeltaNet state buffers for all layers
- Pipeline caching infrastructure (VkPipeline reuse across dispatches)
- Attention output projection: pre-dequantized Q5_K→FP32 on CPU during init
  as workaround for a GPU Q5_K shader issue
- Chat template: automatic CHATML wrapping for Qwen conversation format

**CPU reference (Phase 1):**
- GGUF v3 metadata parsing and model validation for Unsloth/llama.cpp Qwen3.6 layouts
- `mmap`-backed weight access with tensor offsets, shapes, quant types, and byte spans
- Packed routed-expert tensors split into per-expert views
- Tokenizer (BPE, 248k vocab) with encode/decode and dump support
- Native dequantization for F32, F16, BF16, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_S
- CPU kernels: RMSNorm, MoE routing (top-8 of 256), GQA attention, MRoPE,
  Gated DeltaNet (conv1d, update, retrieve), SiLU, softmax, argmax
- 40-layer greedy generation smoke path

**18 Vulkan compute shaders:** `rmsnorm`, `rmsnorm_apply`, `rmsnorm_full`,
`matvec_f32`, `matmul_q4k`, `matmul_q5k`, `matmul_q6k`, `matmul_iq2xxs`,
`add`, `silu_mul`, `argmax`, `sampling`, `rope_mrope`, `attention_gqa`,
`moe_route`, `moe_gather`, `deltanet_conv1d`, `deltanet_update`,
`deltanet_retrieve`, `mtp_draft`.

The repository contains:
- [ARCHITECTURE.md](ARCHITECTURE.md) — Engine design and implementation details
- [MODEL_CARD.md](MODEL_CARD.md) — Qwen 3.6-35B-A3B specifications, tensor layout, quantisation plan
- [BC250_SETUP.md](BC250_SETUP.md) — Hardware setup guide (Vulkan, TTM, 40-CU unlock, oberon, memory)
- [ROADMAP.md](ROADMAP.md) — Development phases and milestones
- [CONTRIBUTING.md](CONTRIBUTING.md) — Build, test, regression protocol

---

## Quick Start

### CPU self-test (no hardware required)
```bash
make test        # builds qw6 and runs ./qw6 --self-test
./qw6 --inspect-gguf model.gguf  # inspect GGUF header
```

### GPU pipeline (BC-250 required)
```bash
make vulkan                    # build with Vulkan support
./qw6 --vulkan-self-test       # verify all GPU shaders
./qw6 --load-only --vulkan -m model.gguf  # probe weights vs CPU
./qw6 -m model.gguf -p "Hello" -n 5 --nothink --vulkan  # generate
./qw6 -m model.gguf -p "Hello" -n 5 --vulkan  # with chat template
./qw6 -m model.gguf -p "Hi" -n 1 --nothink --raw --vulkan  # raw prompt
```

### Usage flags
| Flag | Description |
|---|---|
| `-m model.gguf` | Model file (required for inference) |
| `-p "prompt"` | Prompt text |
| `-n N` | Number of tokens to generate |
| `--ctx N` | Context window size (default: 65536) |
| `--temp T` | Sampling temperature (0 = greedy) |
| `--seed N` | Random seed for deterministic generation |
| `--vulkan` | Use GPU pipeline (Phase 2) |
| `--cpu` | Use CPU reference path (default) |
| `--nothink` | Disable thinking mode |
| `--raw` | Skip chat template (raw prompt only) |
| `--vulkan-strict` | Fail on any GPU fallback |
| `--vulkan-self-test` | Run Vulkan compute shader self-test |
| `--load-only` | Load model and run probes only |
| `--self-test` | Run CPU self-test |
| `--inspect-gguf` | Print GGUF metadata and tensor table |
| `--dump-tokens` | Tokenize prompt and print all token IDs |
| `--dump-logits` | Dump top-10 logits after prefill and each generated token |
| `--dump-logprobs` | Dump top-10 log-probabilities with token probabilities |
| `--dump-full-logits <FILE>` | Write full logit vector as raw float32 binary |
| `--trace-json <FILE>` | Write structured trace JSON after generation |
| `--bench` | Run 128-token timed benchmark |

### Output example
```
$ ./qw6 -m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf -p "Hi" -n 1 --nothink --raw --vulkan
...
qw6: generated text: "!"
qw6: 1 tokens in 28.61s
```

---

## Project Structure

| File | Purpose |
|---|---|
| `qw6.c` (~115K) | Main inference engine (GGUF loader, CPU kernels, session) |
| `qw6.h` (~13K) | Architecture constants, quant types, tensor structs, API |
| `qw6_tok.c` (~41K) | BPE tokenizer (248k vocab) |
| `qw6_tok.h` (~2K) | Tokenizer API |
| `qw6_vk.c` (~2250 lines) | Vulkan compute backend (shaders, dispatch, pipeline) |
| `qw6_vk.h` (~3K) | Vulkan backend API |
| `qw6_vk_pipe.h` (~3K) | GPU pipeline context (opaque handle) |
| `vulkan/*.comp` | 18 GLSL compute shaders + `add.comp` |
| `Makefile` | Build targets: `cpu`, `vulkan`, `test`, `bench` |

## Acknowledgements

- **[antirez/ds4](https://github.com/antirez/ds4)** — the architectural blueprint for a model-specific C inference engine
- **[akandr/bc250](https://github.com/akandr/bc250)** — BC-250 LLM setup, benchmarks, roofline analysis, 40-CU A/B data
- **[elektricm/amd-bc250-docs](https://elektricm.github.io/amd-bc250-docs/)** — comprehensive BC-250 documentation
- **[duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock)** — community kernel patch re-enabling all 40 CUs
- **[QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6)** — the model
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF format, quantisation kernels, Vulkan backend reference

## License

MIT — see [LICENSE](LICENSE).
