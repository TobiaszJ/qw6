# Test Vectors

Reference logprob vectors from the official Qwen 3.6-35B-A3B API for qw6
regression testing. Each vector contains the prompt, generated text, and
top-k logprobs at each token position.

## Collecting Vectors

```bash
# Set your API key (OpenRouter or any OpenAI-compatible endpoint)
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run the collector
python3 tests/test-vectors/collect_vectors.py
```

Alternative providers:
```bash
# DashScope (Alibaba's official API)
export QW6_API_KEY=***";  export QW6_API_BASE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export QW6_MODEL="qwen3.6-35b-a3b"
python3 tests/test-vectors/collect_vectors.py
```

## Vector Format

Each `.json` file contains:

```json
{
  "name": "short_prompt",
  "prompt": "The capital of France is",
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 5,
  "temperature": 0,
  "thinking": false,
  "generated_text": "Paris, the capital...",
  "finish_reason": "stop",
  "token_logprobs": [
    [
      {"token": 12345, "logprob": -0.123, "bytes": "Par"},
      {"token": 67890, "logprob": -3.456, "bytes": " The"}
    ],
    ...
  ],
  "model": "qwen/qwen3.6-35b-a3b",
  "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
}
```

## Regression Test

```bash
# After qw6 Phase 1 is complete:
./qw6_test --logprob-vectors
```

This compares local greedy logprobs token-by-token against the API vectors.
Any divergence means tokeniser, template, attention, or DeltaNet state has a bug.