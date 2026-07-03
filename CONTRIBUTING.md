# Contributing to qw6

This project is in pre-alpha. The priority is Phase 1 (CPU reference path) —
correctness before speed.

---

## Build Targets

```bash
make cpu        # CPU-only reference/debug build (no Vulkan needed)
make vulkan     # Vulkan build for BC-250 (requires Mesa 25.1+, Vulkan SDK)
make test       # Run regression tests
make test-tokenizer # Run tokenizer regression tests when tokenizer data is available
make bench      # Run benchmarks
./qw6 --self-test # Run CPU self-tests without model/tokenizer data
```

**CPU build** works on any Linux machine. It is for correctness checking only —
do not treat it as the production target.

**Vulkan build** requires:
- Mesa 25.1.0+ (25.3.4+ recommended)
- Vulkan headers (`vulkan-dev` or equivalent)
- GFX1013 GPU (BC-250) or compatible RADV-supported AMD APU
- TTM tuning (see [BC250_SETUP.md](BC250_SETUP.md))

---

## Testing

### Test Vectors

`tests/test-vectors/` contains official Qwen 3.6-35B-A3B API continuations:

```
tests/test-vectors/
├── short_prompt.txt          # ~100 token prompt, greedy decoding, thinking disabled
├── short_expected.json       # Expected top-k logprobs at each step
├── medium_4k.txt             # 4K token prompt
├── medium_4k_expected.json
├── long_32k.txt              # 32K token prompt (tests DeltaNet state)
├── long_32k_expected.json
├── tool_call.txt             # Tool calling format test
└── tool_call_expected.json
```

### Regression Test

```bash
make test
# or manually:
./qw6 --self-test

# Optional tokenizer regression test (requires tokenizer/tokenizer.json):
make test-tokenizer

# Planned full regression runner:
./qw6_test --logprob-vectors      # Compare local logprobs vs API test vectors
./qw6_test --server               # Server API smoke test
```

A regression test failure means tokeniser, template, attention, or DeltaNet
state has diverged from the reference. Fix before submitting a PR.

### Debugging

```bash
./qw6 --dump-tokens -p "..."     # Tokenise and exit (check tokeniser)
./qw6 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./qw6 --dump-logits /tmp/logits.json --vulkan --nothink --prompt-file prompt.txt
./qw6-server --trace /tmp/trace.txt ...   # Full session trace
```

---

## Quality Standards

Following the project's coding standards:

1. **No uncontrolled control flow** — no goto, no recursion, no callback hell
2. **All loops must terminate** — fixed upper bound or clear exit path
3. **No dynamic memory after init** — allocate upfront, no uncontrolled object creation
4. **Max 60 lines per function**
5. **At least 2 guards/assertions per function** (input validation, postconditions)
6. **Minimal dereferencing** — no deep nested property access, no non-null without guard
7. **Metaprogramming only for conditionals** — no eval, exec, complex macros
8. **Minimal scope** — no global mutable state, no mutable default arguments
9. **Check every return value, validate every input** — never silent-fail
10. **Linter warnings = errors** — strict mode, warnings are errors

---

## Pull Request Checklist

- [ ] `make cpu` succeeds
- [ ] `make test` passes (tokenizer-free CPU self-test)
- [ ] `make test-tokenizer` passes when `tokenizer/tokenizer.json` is available
- [ ] Logprob vector tests pass once Phase 1 inference is complete
- [ ] Style: 60-line function limit, 2+ assertions per function, strict linter
- [ ] No new memory leaks (`valgrind --leak-check=full ./qw6 --cpu -p "test"`)
- [ ] Vulkan build: no validation layer errors (`vulkaninfo --validate`)
- [ ] Commit message: descriptive, present tense ("Add Gated DeltaNet conv1d kernel" not "added")

---

## Areas Needing Help

- **Vulkan compute shader optimisation** — especially Gfx1013 wave64 tiling
- **Gated DeltaNet kernel** — reference implementation from `fla` / `causal_conv1d`
- **imatrix collection** — calibration corpus for routed expert quantisation
- **Test vectors** — additional prompts at different context lengths
- **Documentation** — additional BC-250 hardware quirks, GFX1013 ISA notes
