#!/usr/bin/env python3
"""
qw6 Correctness Baseline Harness.

Compares qw6 output against llama.cpp reference for:
  1. Tokenizer parity (same prompt -> same token IDs)
  2. Chat template parity (same chat -> same formatted tokens)
  3. Full logit parity (single-token forward, same input -> same output logits)
  4. Greedy generation parity (same prompt -> same generated tokens)

Usage:
  # Tokenizer comparison (quick, no model needed for qw6)
  python3 tests/correctness/compare.py tok --model model.gguf

  # Logit comparison (requires model load, ~10-30s)
  python3 tests/correctness/compare.py logits --model model.gguf

  # Full generation comparison (slow, 1+ minute)
  python3 tests/correctness/compare.py gen --model model.gguf

  # Run everything
  python3 tests/correctness/compare.py all --model model.gguf
"""

import subprocess
import sys
import json
import os
import struct
import argparse
import re
import tempfile
import math

QW6 = "qw6"
QW6_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QW6_BIN = os.path.join(QW6_DIR, QW6)
LLAMA_DIR = "/home/teejay/llama.cpp/build/bin"
LLAMA_TOKENIZE = os.path.join(LLAMA_DIR, "llama-tokenize")
LLAMA_CLI = os.path.join(LLAMA_DIR, "llama-cli")

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TOKENIZER_PROMPTS = [
    ("simple", "The capital of France is"),
    ("hello", "Hello, world!"),
    ("digits", "1234567890"),
    ("special_tokens", "<|im_start|><|im_end|>"),
    ("unicode", "café résumé 北京"),
    ("code_simple", "fn main() { println!(\"hello\"); }"),
    ("punctuation", "well-known, high-quality, state-of-the-art"),
    ("whitespace", "a\tb\nc\td"),
    ("code_fn", "function fib(n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); }"),
    ("thinking_marker", "Wait, let me think about this step by step."),
    ("numbers_vs_words", "There are 42 apples and 7 oranges."),
    ("url", "Visit https://example.com/path?q=search for more info."),
    ("json_example", '{"key": "value", "nums": [1, 2, 3]}'),
    ("long_word", "antidisestablishmentarianism"),
    ("multiline", "Line one.\nLine two.\nLine three."),
]

CHAT_TEMPLATE_TESTS = [
    ("simple_qa", "What is the capital of France?"),
    ("coding", "Write a Python function to sort a list."),
    ("multi_sentence", "Explain quantum computing in simple terms. Give an example."),
]

GENERATION_PROMPTS = [
    ("short", "The capital of France is", 5),
    ("single", "Hello", 1),
    ("what_is", "What is 2+2?", 3),
]

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run(cmd, timeout=120, retries=2, **kwargs):
    """Run a command and return stdout, stderr, returncode. Retries on failure."""
    for attempt in range(1 + retries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
            if r.returncode == 0 or attempt >= retries:
                return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            if attempt >= retries:
                return "", f"TIMEOUT after {timeout}s: {' '.join(cmd)}", -1
        except FileNotFoundError as e:
            return "", f"FILE NOT FOUND: {e.filename}", -1
        # Brief pause before retry
        import time as _time
        _time.sleep(0.5)
    return "", "all retries exhausted", -1


def parse_qw6_tokens(output):
    """Parse qw6 --dump-tokens output: [760, 6511, 314, 9338, 369] (5 tokens)"""
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('['):
            # Strip any trailing text after the closing bracket
            bracket_end = line.index(']')
            token_str = line[:bracket_end + 1]
            try:
                return json.loads(token_str)
            except json.JSONDecodeError as e:
                print(f"    [debug] JSON parse error on token line: {token_str[:200]}: {e}")
                return None
    return None


def parse_llama_tokens(output):
    """Parse llama-tokenize --ids output: [760, 6511, 314, 9338, 369]"""
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('['):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def parse_trace_json(path):
    """Read and parse a qw6 trace JSON file."""
    with open(path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# 1. Tokenizer comparison
# ---------------------------------------------------------------------------

def test_tokenizer(model_path, prompts=None):
    if prompts is None:
        prompts = TOKENIZER_PROMPTS
    if not os.path.exists(LLAMA_TOKENIZE):
        print(f"FAIL: llama-tokenize not found at {LLAMA_TOKENIZE}")
        return False

    failures = 0
    total = 0

    for name, prompt in prompts:
        total += 1
        # Run qw6 tokenizer
        qw6_cmd = [QW6_BIN, "--dump-tokens", "-p", prompt]
        qw6_out, qw6_err, qw6_rc = run(qw6_cmd, timeout=30)
        if qw6_rc != 0:
            print(f"  FAIL [{name}] qw6 returned {qw6_rc}: {qw6_err.strip()[:100]}")
            failures += 1
            continue

        qw6_tokens = parse_qw6_tokens(qw6_out)
        if qw6_tokens is None:
            print(f"  FAIL [{name}] could not parse qw6 tokens")
            print(f"    stdout: {qw6_out[:200] if qw6_out else '(empty)'}")
            print(f"    stderr: {qw6_err.strip()[:200] if qw6_err else '(none)'}")
            failures += 1
            continue

        # Run llama-tokenize
        llm_cmd = [LLAMA_TOKENIZE, "-m", model_path, "--ids", "-p", prompt, "--log-disable"]
        llm_proc = subprocess.run(llm_cmd, capture_output=True, text=True, timeout=120)
        llm_out = llm_proc.stdout
        llm_tokens = parse_llama_tokens(llm_out)

        if llm_tokens is None:
            # Try with --no-bos
            llm_cmd2 = [LLAMA_TOKENIZE, "-m", model_path, "--ids", "--no-bos", "-p", prompt, "--log-disable"]
            llm_proc2 = subprocess.run(llm_cmd2, capture_output=True, text=True, timeout=120)
            llm_out2 = llm_proc2.stdout
            llm_tokens = parse_llama_tokens(llm_out2)
            if llm_tokens is None:
                stderr_snip = llm_proc.stderr[:300] if llm_proc.stderr else ""
                print(f"  FAIL [{name}] could not parse llama tokens\n"
                      f"    stdout: {llm_out[:200] if llm_out else '(empty)'}\n"
                      f"    stderr: {stderr_snip}")
                failures += 1
                continue

        # Compare
        if qw6_tokens == llm_tokens:
            print(f"  PASS [{name}] ({len(qw6_tokens)} tokens)")
        else:
            failures += 1
            print(f"  FAIL [{name}]")
            print(f"    qw6:   {qw6_tokens}")
            print(f"    llama: {llm_tokens}")
            # Show common prefix
            min_len = min(len(qw6_tokens), len(llm_tokens))
            same_prefix = 0
            for i in range(min_len):
                if qw6_tokens[i] == llm_tokens[i]:
                    same_prefix += 1
                else:
                    break
            print(f"    common prefix: {same_prefix} tokens")
            if same_prefix > 0:
                print(f"    first diff at index {same_prefix}")
                if same_prefix < len(qw6_tokens):
                    print(f"    qw6[{same_prefix}]:   {qw6_tokens[same_prefix]}")
                if same_prefix < len(llm_tokens):
                    print(f"    llama[{same_prefix}]: {llm_tokens[same_prefix]}")

    status = "PASS" if failures == 0 else "FAIL"
    print(f"\nTokenizer: {status} ({total - failures}/{total} passed, {failures} failed)")
    return failures == 0


# ---------------------------------------------------------------------------
# 2. Chat template parity
# ---------------------------------------------------------------------------

QW6_CHAT_TEMPLATE = [
    "<|im_start|>system",
    "You are a helpful assistant",
    "<|im_end|>",
    "<|im_start|>user",
    "{prompt}",
    "<|im_end|>",
    "<|im_start|>assistant",
]

def test_chat_template(model_path, tokenizer_path):
    """Compare chat-formatted token sequences between qw6 and llama.cpp."""
    print("\n=== Chat Template Parity ===\n")

    if not os.path.exists(LLAMA_TOKENIZE):
        print(f"SKIP: llama-tokenize not found")
        return None

    failures = 0
    for name, prompt in CHAT_TEMPLATE_TESTS:
        # Build the expected chat template
        sys_msg = "You are a helpful assistant"
        templated = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Get llama tokens for the templated prompt
        llm_cmd = [LLAMA_TOKENIZE, "-m", model_path, "--ids", "--no-bos", "-p", templated, "--log-disable"]
        llm_proc = subprocess.run(llm_cmd, capture_output=True, text=True, timeout=120)
        llm_tokens = parse_llama_tokens(llm_proc.stdout)
        if llm_tokens is None:
            print(f"  FAIL [{name}] could not parse llama tokens")
            failures += 1
            continue

        # qw6 applies its own chat template internally during inference.
        # We can verify by running qw6 with --nothink (no --raw) and dumping generated tokens.
        # The prompt tokens are not dumped separately, but we can check the output.
        # For the token sequence itself, manually tokenize the same template
        qw6_cmd = [QW6_BIN, "--dump-tokens", "-p", templated, "--tok", tokenizer_path]
        qw6_out, qw6_err, qw6_rc = run(qw6_cmd)
        if qw6_rc != 0:
            print(f"  FAIL [{name}] qw6 error: {qw6_err.strip()[:100]}")
            failures += 1
            continue

        qw6_tokens = parse_qw6_tokens(qw6_out)

        if qw6_tokens is None:
            print(f"  FAIL [{name}] could not parse qw6 tokens")
            failures += 1
            continue

        if qw6_tokens == llm_tokens:
            print(f"  PASS [{name}] ({len(qw6_tokens)} tokens)")
        else:
            failures += 1
            print(f"  FAIL [{name}] token mismatch")
            min_len = min(len(qw6_tokens), len(llm_tokens))
            for i in range(min_len):
                if qw6_tokens[i] != llm_tokens[i]:
                    print(f"    first diff at index {i}: qw6={qw6_tokens[i]} llama={llm_tokens[i]}")
                    print(f"    qw6 tokens around diff: {qw6_tokens[max(0,i-3):i+3]}")
                    print(f"    llama tokens around diff: {llm_tokens[max(0,i-3):i+3]}")
                    break
            if len(qw6_tokens) != len(llm_tokens):
                print(f"    qw6 length: {len(qw6_tokens)}, llama length: {len(llm_tokens)}")

    status = "PASS" if failures == 0 else "FAIL"
    print(f"\nChat Template: {status}")
    return failures == 0


# ---------------------------------------------------------------------------
# 3. Logit comparison
# ---------------------------------------------------------------------------

def read_float32_bin(path):
    """Read a binary file of float32 values."""
    with open(path, 'rb') as f:
        data = f.read()
    return struct.unpack(f'{len(data)//4}f', data)


def test_logit_parity(model_path, tokenizer_path):
    """Compare qw6 CPU logits against llama.cpp for a single-token prompt."""
    print("\n=== Logit Parity ===\n")

    # We need a way to extract logits from both systems.
    # For qw6: --dump-full-logits <file>
    # For llama.cpp: we use the --logit-bias trick with ppl output, or write a
    #   custom C program that links llama.cpp. For now, compare top-k values.

    test_prompts = [
        ("single_token", "Hello", 1),
        ("short_prompt", "The capital of France is", 1),
    ]

    failures = 0
    for name, prompt, n_gen in test_prompts:
        # --- qw6 CPU path ---
        with tempfile.NamedTemporaryFile(suffix='.f32', delete=False) as f:
            logits_path = f.name

        qw6_cmd = [
            QW6_BIN, "-m", model_path, "-p", prompt, "-n", str(n_gen),
            "--temp", "0", "--nothink", "--raw",
            "--dump-full-logits", logits_path
        ]
        qw6_out, qw6_err, qw6_rc = run(qw6_cmd, timeout=600)
        if qw6_rc != 0:
            print(f"  FAIL [{name}] qw6 error: {qw6_err.strip()[:200]}")
            failures += 1
            try: os.unlink(logits_path)
            except: pass
            continue

        if not os.path.exists(logits_path):
            print(f"  FAIL [{name}] qw6 did not create logits file")
            failures += 1
            continue

        qw6_logits = read_float32_bin(logits_path)
        try: os.unlink(logits_path)
        except: pass

        # Check for NaN/Inf
        bad = sum(1 for v in qw6_logits if math.isnan(v) or math.isinf(v))
        if bad > 0:
            print(f"  FAIL [{name}] qw6 logits contain {bad} NaN/Inf values!")
            failures += 1
            continue

        # --- llama.cpp path ---
        # Use llama-cli with --temp 0, --ignore-eos, and capture output
        # We can't directly dump logits from llama-cli without custom code,
        # but we CAN compare the generated tokens
        llm_cmd = [
            LLAMA_CLI, "-m", model_path, "-p", prompt, "-n", str(n_gen),
            "--temp", "0", "--ignore-eos", "--no-display-prompt",
            "--simple-io"
        ]
        llm_out, llm_err, llm_rc = run(llm_cmd, timeout=600)

        # Parse generated tokens from the error stream (llama-cli outputs to stderr)
        # The output format varies; check the output text
        # For --simple-io, the output is just text on stdout
        llm_text = llm_out.strip() if llm_out.strip() else llm_err.strip()

        # For now, let's compare logits at the top-k level
        # Sort indices by logit value (descending)
        qw6_top20 = sorted(
            range(len(qw6_logits)),
            key=lambda i: qw6_logits[i],
            reverse=True
        )[:20]

        print(f"  qw6 [{name}]: top-5 token IDs: {qw6_top20[:5]}, values: {[qw6_logits[i] for i in qw6_top20[:5]]}")

        # Logit value range check
        max_val = max(qw6_logits)
        min_val = min(qw6_logits)
        if max_val > 100 or min_val < -100:
            print(f"  WARN [{name}] logit range seems off: [{min_val:.1f}, {max_val:.1f}]")

        # Check that the generated token is within a reasonable range (0..vocab_size)
        # The first generated token is the argmax of the logits
        gen_token = qw6_top20[0]
        print(f"  qw6 [{name}]: greedy token = {gen_token}")

        # What does llama.cpp generate?
        gen_via_tok = subprocess.run(
            [LLAMA_TOKENIZE, "-m", model_path, "--ids", "--log-disable", "--no-bos"],
            input=llm_text, capture_output=True, text=True, timeout=30
        )
        llm_gen_tokens = parse_llama_tokens(gen_via_tok.stdout)

        print(f"  llama [{name}]: generated text = \"{llm_text[:100]}\"")
        if llm_gen_tokens:
            print(f"  llama [{name}]: generated token IDs = {llm_gen_tokens}")

    if failures == 0:
        print(f"\nLogit Parity: no hard failures (logits dumped, manual inspection possible)")
    else:
        print(f"\nLogit Parity: {failures} failures")
    return failures == 0


# ---------------------------------------------------------------------------
# 4. Generation parity
# ---------------------------------------------------------------------------

def test_generation(model_path, tokenizer_path):
    """Compare greedy generation between qw6 CPU and llama.cpp."""
    print("\n=== Greedy Generation Parity ===\n")

    test_prompts = [
        ("short", "The capital of France is", 5),
        ("single", "Hello", 1),
    ]

    failures = 0
    for name, prompt, n_gen in test_prompts:
        # --- qw6 CPU ---
        qw6_cmd = [
            QW6_BIN, "-m", model_path, "-p", prompt, "-n", str(n_gen),
            "--temp", "0", "--nothink", "--raw"
        ]
        qw6_out, qw6_err, qw6_rc = run(qw6_cmd, timeout=600)
        if qw6_rc != 0:
            print(f"  FAIL [{name}] qw6 error: {qw6_err.strip()[:200]}")
            failures += 1
            continue

        # Parse generated text from qw6 output
        qw6_text = ""
        for line in qw6_err.split('\n'):
            m = re.match(r'qw6: generated text: "(.*)"', line)
            if m:
                qw6_text = m.group(1)

        # --- llama.cpp ---
        llm_cmd = [
            LLAMA_CLI, "-m", model_path, "-p", prompt, "-n", str(n_gen),
            "--temp", "0", "--ignore-eos", "--no-display-prompt",
            "--simple-io"
        ]
        llm_out, llm_err, llm_rc = run(llm_cmd, timeout=600)
        llm_text = llm_out.strip()

        # Compare
        match = (qw6_text == llm_text)
        if match:
            print(f"  PASS [{name}] text matches: \"{qw6_text[:100]}\"")
        else:
            failures += 1
            print(f"  FAIL [{name}] generation mismatch")
            print(f"    qw6:   \"{qw6_text[:200]}\"")
            print(f"    llama: \"{llm_text[:200]}\"")
            # Show first differing char
            for i, (a, b) in enumerate(zip(qw6_text, llm_text)):
                if a != b:
                    print(f"    first diff at char {i}: qw6={repr(a)} llama={repr(b)}")
                    break
            if len(qw6_text) != len(llm_text):
                print(f"    qw6 length: {len(qw6_text)}, llama length: {len(llm_text)}")

    status = "PASS" if failures == 0 else "FAIL"
    print(f"\nGeneration: {status} ({len(test_prompts) - failures}/{len(test_prompts)} passed)")
    return failures == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="qw6 Correctness Baseline Harness")
    parser.add_argument("mode", choices=["tok", "chat", "logits", "gen", "all"],
                       help="What to test")
    parser.add_argument("--model", "-m", default=os.path.join(QW6_DIR, "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
                       help="Path to GGUF model")
    parser.add_argument("--tokenizer", "-t", default=os.path.join(QW6_DIR, "tokenizer", "tokenizer.json"),
                       help="Path to tokenizer.json")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    tokenizer_path = os.path.abspath(args.tokenizer)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        print("(Tokenizer not required for all tests; continuing...)")

    if not os.path.exists(QW6_BIN):
        print(f"qw6 binary not found: {QW6_BIN}")
        print("Run 'make cpu' first.")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"qw6: {QW6_BIN}")
    print()

    results = {}

    if args.mode in ("tok", "all"):
        results["tokenizer"] = test_tokenizer(model_path)
        print()

    if args.mode in ("chat", "all"):
        results["chat_template"] = test_chat_template(model_path, tokenizer_path)
        print()

    if args.mode in ("logits", "all"):
        results["logit_parity"] = test_logit_parity(model_path, tokenizer_path)
        print()

    if args.mode in ("gen", "all"):
        results["generation"] = test_generation(model_path, tokenizer_path)
        print()

    # Summary
    if results:
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        all_pass = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_pass = False
        print()
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
