# Contributing to qw6

This project is in pre-alpha. Phase 2 (Vulkan GPU pipeline) is the active
development focus. A correctness baseline against llama.cpp is being established
in `tests/correctness/compare.py`. Both CPU and GPU paths produce output but
have not been validated against reference logits layer-by-layer.

---

## Build Targets

```bash
make cpu        # CPU-only reference/debug build (no Vulkan needed)
make vulkan     # Vulkan build for BC-250 (requires Mesa 25.1+, Vulkan SDK, glslc)
make test       # Run regression tests (CPU, no model needed)
make test-tokenizer # Run tokenizer regression tests (requires tokenizer data)
```

### GPU Pipeline Test
```bash
make vulkan
./qw6 --vulkan-self-test                    # Verify all GPU shaders
./qw6 --load-only --vulkan -m model.gguf    # Model probe vs CPU reference
./qw6 -m model.gguf -p "Hi" -n 1 --nothink --raw --vulkan  # Full pipeline test
./qw6 -m model.gguf -p "Hello" -n 3 --nothink --vulkan      # With chat template
```

**CPU build** works on any Linux machine. A correctness baseline against
llama.cpp is being established (see **Correctness Baseline** below).

**Vulkan build** requires:
- Mesa 25.1.0+ (25.3.4+ recommended)
- Vulkan headers and `glslc` (Vulkan SDK or `shaderc`)
- GFX1013 GPU (BC-250) — see [BC250_SETUP.md](BC250_SETUP.md) for hardware setup
- TTM tuning for 16 GB unified memory

---

## Correctness Baseline

A comparison harness lives at `tests/correctness/compare.py`:

```bash
# Tokenizer comparison (quick, 15 test cases)
python3 tests/correctness/compare.py tok -m model.gguf

# Chat template parity
python3 tests/correctness/compare.py chat -m model.gguf

# Full logit dump (requires model load)
python3 tests/correctness/compare.py logits -m model.gguf

# Greedy generation comparison (requires BC-250 for llama.cpp speed)
python3 tests/correctness/compare.py gen -m model.gguf

# Everything
python3 tests/correctness/compare.py all -m model.gguf
```

**Current status (CPU, no GPU):**
- Tokenizer: 12/15 prompts match llama.cpp. 3 mismatches are pre-tokenization
  differences with `{`/`}` boundaries (qw6's simplified pre-tokenizer vs
  HuggingFace regex).
- CPU inference: verified functional (Hello → token 1873, 7.59s/token).
- Full logit dump: `--dump-full-logits <FILE>` writes 248k float32 values.
- llama.cpp comparison on BC-250: not yet run.

### Debugging flags

```bash
./qw6 --dump-tokens -p "..."                  # All token IDs (no truncation)
./qw6 -m model.gguf --dump-logits -p "..."     # Top-10 logits after prefill
./qw6 -m model.gguf --dump-logprobs -p "..."   # Top-10 logprobs
./qw6 -m model.gguf --dump-full-logits /tmp/l.f32 -p "..."  # Full logit vector
./qw6 -m model.gguf --trace-json /tmp/t.json -p "..."       # Structured trace
```

---

### GPU Pipeline
- `matmul_q5k.spv` produces 100-500x wrong output despite byte-identical weight
  data — workaround: pre-dequantize Q5_K→FP32 during pipeline init
- IQ2_S / IQ3_S expert weights lack GPU shaders — CPU fallback
- Pipeline caching disabled (GPU hang when reusing VkPipeline objects)
- No chunked prefill — multi-token prompts are processed one at a time (~28s/token)
- Core inference math not validated against reference outputs

### CPU Reference
- Not yet validated layer-by-layer against llama.cpp logits
- Both CPU and GPU paths may produce incorrect tokens
- Tokenizer: pre-tokenization is simplified (3/15 test prompts differ from
  llama.cpp on `{`/`}` boundary splitting)
- `qw6_cpu_matmul_iq2m` uses a flat test format, NOT the real IQ2_XXS GGUF blocks
  (but `qw6_tensor_matvec` → `qw6_tensor_dequantize_row` uses the correct format)

---

## Debugging

```bash
./qw6 --dump-tokens -p "..."        # Tokenise and print token IDs
./qw6 --inspect-gguf model.gguf     # Print GGUF metadata and tensor table
./qw6 -m model.gguf --load-only     # Run native probe functions

# With --raw flag to skip chat template (shorter prefill):
./qw6 -m model.gguf -p "Hi" -n 1 --nothink --raw --vulkan
```

### Using llama.cpp as Reference

A working llama.cpp build with Vulkan support is available at `/home/teejay/llama.cpp/`.

```bash
# Compare outputs:
/home/teejay/llama.cpp/build/bin/llama-completion -m model.gguf \
  --prompt "Hello" -n 10 --temp 0 --no-mmap -ngl 999 --no-warmup
```

llama.cpp processes ~17ms/token on the BC-250 with Vulkan (`-ngl 999`).

---

## Quality Standards

1. **No uncontrolled control flow** — no gotos, no recursion
2. **All loops must terminate** — fixed upper bound or clear exit path
3. **No dynamic memory after init** — pre-allocate
4. **Max 60 lines per function** (soft limit)
5. **At least 2 guards per function** — input validation, postconditions
6. **Minimal dereferencing** — guard before access
7. **Check every return value** — never silent-fail
8. **Linter warnings = errors** — strict mode (`-Werror`)

---

## Pull Request Checklist

- [ ] `make cpu` succeeds with no warnings
- [ ] `make vulkan` succeeds with no warnings
- [ ] `./qw6 --self-test` passes
- [ ] `./qw6 --vulkan-self-test` passes on BC-250
- [ ] `./qw6 --load-only --vulkan -m <model.gguf>` passes
- [ ] Tokenizer comparison: `python3 tests/correctness/compare.py tok -m <model.gguf>` — no regressions vs baseline
- [ ] No new `valgrind` issues in CPU path
- [ ] Style: function length, assertions, strict linter
- [ ] Commit message: descriptive, present tense

---

## Areas Needing Help

- **Fix `matmul_q5k.spv`** — produce correct output despite correct weight data
- **Port IQ2_S / IQ3_S shaders** — from llama.cpp's `mul_mat_vec_iq2_s.comp`
- **Fix pipeline caching** — GPU hang when reusing VkPipeline objects
- **Chunked prefill** — batch N tokens through forward pass
- **Model validation** — compare layer-by-layer logits against llama.cpp
- **DeltaNet / Attention GPU kernels** — wire up existing shaders into pipeline
- **Performance optimisation** — wave64 tiling, fused kernels, memory layout
- **Chat template** — read from GGUF metadata instead of hardcoded CHATML
