#!/usr/bin/env python3
"""Collect test vectors from Qwen 3.6-35B-A3B locally via HuggingFace Transformers.

This is the GOLD STANDARD for qw6 regression testing — it produces exact
BF16 logits directly from the model weights, not filtered through an API.

Requirements:
  pip install torch transformers accelerate

System:
  - 128 GB RAM (model is ~70 GB in BF16)
  - GPU optional but helps (RTX 3090 24 GB → auto-offload)
  - ~70 GB disk for model cache (download once)

Usage:
  python3 collect_vectors_local.py

Output:
  tests/test-vectors/*.json — same format as collect_vectors.py
  But with FULL logits (top-50 per token, not just top-20 from API)
"""

import json
import os
import sys
import time
import traceback

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
TOP_K = 50  # Save top-50 logprobs per token (more than API's 20)


def load_model():
    """Load Qwen 3.6-35B-A3B in BF16 with auto device mapping."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model: {MODEL_ID} (BF16, auto device map)")
    print("  This will download ~70 GB on first run and take a few minutes...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # GPU + CPU offload
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"  Model loaded in {elapsed:.1f}s")

    # Print device map summary
    if hasattr(model, "hf_device_map"):
        devices = {}
        for name, dev in model.hf_device_map.items():
            devices[str(dev)] = devices.get(str(dev), 0) + 1
        print(f"  Device map: {devices}")

    return model, tokenizer, torch


def generate_with_logprobs(model, tokenizer, torch, messages, max_tokens=5,
                           temperature=0.0):
    """Generate tokens greedily and capture top-k logprobs at each step.

    Returns dict with generated text, token IDs, and per-token top-k logprobs.
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generated_ids = []
    all_logprobs = []
    input_len = inputs["input_ids"].shape[1]

    print(f"  Input: {input_len} tokens")

    with torch.no_grad():
        for step in range(max_tokens):
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Greedy: pick argmax
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Get top-k logprobs
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_vals, topk_indices = torch.topk(log_probs, TOP_K, dim=-1)

            top_k_list = []
            for i in range(min(TOP_K, topk_indices.shape[-1])):
                tok_id = topk_indices[0, i].item()
                tok_str = tokenizer.decode([tok_id])
                top_k_list.append({
                    "token_id": tok_id,
                    "token": tok_str,
                    "logprob": topk_vals[0, i].item(),
                })
            all_logprobs.append(top_k_list)

            next_id = next_token[0].item()
            generated_ids.append(next_id)

            # Check EOS
            if next_id == tokenizer.eos_token_id:
                break

            # Append to input for next step
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"],
                next_token
            ], dim=-1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat([
                    inputs["attention_mask"],
                    torch.ones_like(next_token)
                ], dim=-1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "token_logprobs": all_logprobs,
        "input_tokens": input_len,
    }


def save_vector(name, data):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {name}.json ({os.path.getsize(path)} bytes)")


# --- Test Cases ---

TESTS = [
    {
        "name": "short_prompt",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 5,
        "thinking": False,
    },
    {
        "name": "short_prompt_think",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 20,
        "thinking": True,
    },
    {
        "name": "coding_prompt",
        "messages": [{"role": "user", "content": "Write a Python function that sorts a list using quicksort."}],
        "max_tokens": 30,
        "thinking": False,
    },
    {
        "name": "system_prompt",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses very short."},
            {"role": "user", "content": "Say hello in one word."}
        ],
        "max_tokens": 5,
        "thinking": False,
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name? Reply in one word."}
        ],
        "max_tokens": 3,
        "thinking": False,
    },
    {
        "name": "reasoning",
        "messages": [{"role": "user", "content": "What is 15 * 37? Think step by step."}],
        "max_tokens": 50,
        "thinking": True,
    },
]


def main():
    print("=" * 60)
    print("qw6 test vector collector (local — HuggingFace Transformers)")
    print("=" * 60)

    model, tokenizer, torch = load_model()

    print(f"\nModel: {MODEL_ID}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token_id}")
    print(f"Tests: {len(TESTS)}")
    print()

    for i, test in enumerate(TESTS):
        name = test["name"]
        print(f"=== Test {i+1}/{len(TESTS)}: {name} ===")
        print(f"  Prompt: {test['messages'][-1]['content'][:60]}...")
        print(f"  Max tokens: {test['max_tokens']}, thinking: {test['thinking']}")

        t0 = time.time()
        try:
            result = generate_with_logprobs(
                model, tokenizer, torch,
                messages=test["messages"],
                max_tokens=test["max_tokens"],
                temperature=0.0,
            )
            elapsed = time.time() - t0

            vector = {
                "name": name,
                "messages": test["messages"],
                "max_tokens": test["max_tokens"],
                "temperature": 0.0,
                "thinking": test.get("thinking", False),
                "model": MODEL_ID,
                "precision": "bf16",
                "generated_text": result["generated_text"],
                "generated_token_ids": result["generated_token_ids"],
                "token_logprobs": result["token_logprobs"],
                "input_tokens": result["input_tokens"],
                "elapsed_seconds": round(elapsed, 3),
            }
            save_vector(name, vector)
            print(f"  Generated: '{result['generated_text'][:80]}'")
            print(f"  Tokens: {len(result['generated_token_ids'])}")
            print(f"  Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

        print()
        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 60)
    print(f"Done. Vectors saved to: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".json"):
            print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))} bytes)")


if __name__ == "__main__":
    main()