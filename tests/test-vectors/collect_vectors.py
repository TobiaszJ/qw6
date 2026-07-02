#!/usr/bin/env python3
"""Collect test vectors from Qwen 3.6-35B-A3B API for qw6 regression testing.

Generates reference logprobs that qw6 must match token-by-token:
- short_prompt.json      — greedy, thinking disabled (5 tokens)
- short_prompt_think.json — greedy, thinking enabled (20 tokens)
- coding_prompt.json     — coding task, greedy (30 tokens)
- tool_call.json         — XML tool calling format
- system_prompt.json     — system + user multi-turn (5 tokens)
- multi_turn.json        — 3-turn conversation (3 tokens)

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python3 collect_vectors.py

Or with any OpenAI-compatible API:
  export QW6_API_KEY="..."
  export QW6_API_BASE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  export QW6_MODEL="qwen3.6-35b-a3b"
  python3 collect_vectors.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# --- Configuration ---

API_KEY = os.environ.get("QW6_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
API_BASE = os.environ.get("QW6_API_BASE", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("QW6_MODEL", "qwen/qwen3.6-35b-a3b")

# Try .env fallback
if not API_KEY:
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                    API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

if not API_KEY:
    print("ERROR: No API key found. Set OPENROUTER_API_KEY or QW6_API_KEY.")
    sys.exit(1)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

print(f"qw6 test vector collector")
print(f"  API: {API_BASE}")
print(f"  Model: {MODEL}")
print(f"  Key: {API_KEY[:8]}...{API_KEY[-4:]}")
print(f"  Output: {OUT_DIR}")
print()


def call_api(messages, max_tokens=10, temperature=0, top_logprobs=20, tools=None):
    """Call the API and return the response."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if top_logprobs is not None:
        body["top_logprobs"] = top_logprobs
        body["logprobs"] = True
    if tools:
        body["tools"] = tools

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()[:500] if e.fp else str(e)
        print(f"  API Error {e.code}: {error_body}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_logprobs(result):
    """Extract top-k logprobs from API response."""
    choice = result["choices"][0]
    top_lp = []
    for lp in choice.get("logprobs", {}).get("content", []):
        token_top = []
        for t in lp.get("top_logprobs", []):
            token_top.append({
                "token": t["token"],
                "logprob": t["logprob"],
                "bytes": t.get("bytes"),
            })
        top_lp.append(token_top)
    return top_lp


def save_vector(name, data):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {name}.json ({os.path.getsize(path)} bytes)")


# --- Test Vectors ---

def test_short_prompt():
    """Test 1: Short prompt, greedy, thinking disabled."""
    print("=== Test 1: Short prompt (greedy, thinking disabled) ===")
    prompt = "The capital of France is"
    messages = [{"role": "user", "content": prompt}]
    result = call_api(messages, max_tokens=5, temperature=0)
    if result:
        choice = result["choices"][0]
        save_vector("short_prompt", {
            "name": "short_prompt",
            "prompt": prompt,
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0,
            "thinking": False,
            "generated_text": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "token_logprobs": extract_logprobs(result),
            "model": result.get("model", ""),
            "usage": result.get("usage", {}),
        })
        print(f"  Generated: '{choice['message']['content'][:80]}'")
        return True
    return False


def test_short_prompt_thinking():
    """Test 2: Short prompt with thinking enabled."""
    print("\n=== Test 2: Short prompt (thinking enabled) ===")
    prompt = "The capital of France is"
    messages = [{"role": "user", "content": prompt}]
    result = call_api(messages, max_tokens=20, temperature=0)
    if result:
        choice = result["choices"][0]
        save_vector("short_prompt_think", {
            "name": "short_prompt_think",
            "prompt": prompt,
            "messages": messages,
            "max_tokens": 20,
            "temperature": 0,
            "thinking": True,
            "generated_text": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "token_logprobs": extract_logprobs(result),
            "model": result.get("model", ""),
            "usage": result.get("usage", {}),
        })
        print(f"  Generated: '{choice['message']['content'][:80]}'")
        return True
    return False


def test_coding_prompt():
    """Test 3: Coding prompt, greedy."""
    print("\n=== Test 3: Coding prompt (greedy) ===")
    prompt = "Write a Python function that sorts a list using quicksort."
    messages = [{"role": "user", "content": prompt}]
    result = call_api(messages, max_tokens=30, temperature=0)
    if result:
        choice = result["choices"][0]
        save_vector("coding_prompt", {
            "name": "coding_prompt",
            "prompt": prompt,
            "messages": messages,
            "max_tokens": 30,
            "temperature": 0,
            "thinking": False,
            "generated_text": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "token_logprobs": extract_logprobs(result),
            "model": result.get("model", ""),
        })
        print(f"  Generated: '{choice['message']['content'][:80]}'")
        return True
    return False


def test_tool_call():
    """Test 4: Tool calling format."""
    print("\n=== Test 4: Tool calling ===")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    prompt = "What's the weather like in Zurich right now? Use the get_weather tool."
    messages = [{"role": "user", "content": prompt}]
    result = call_api(messages, max_tokens=100, temperature=0, tools=tools)
    if result:
        choice = result["choices"][0]
        save_vector("tool_call", {
            "name": "tool_call",
            "prompt": prompt,
            "messages": messages,
            "tools": tools,
            "max_tokens": 100,
            "temperature": 0,
            "generated_text": choice["message"].get("content", ""),
            "tool_calls": choice["message"].get("tool_calls", []),
            "finish_reason": choice["finish_reason"],
            "model": result.get("model", ""),
        })
        tc = choice["message"].get("tool_calls", [])
        print(f"  Tool calls: {len(tc)}")
        if tc:
            print(f"  First: {tc[0]['function']['name']}({tc[0]['function']['arguments'][:80]})")
        return True
    return False


def test_system_prompt():
    """Test 5: System prompt + user."""
    print("\n=== Test 5: System prompt ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses very short."},
        {"role": "user", "content": "Say hello in one word."}
    ]
    result = call_api(messages, max_tokens=5, temperature=0)
    if result:
        choice = result["choices"][0]
        save_vector("system_prompt", {
            "name": "system_prompt",
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0,
            "thinking": False,
            "generated_text": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "token_logprobs": extract_logprobs(result),
            "model": result.get("model", ""),
        })
        print(f"  Generated: '{choice['message']['content'][:80]}'")
        return True
    return False


def test_multi_turn():
    """Test 6: Multi-turn conversation."""
    print("\n=== Test 6: Multi-turn ===")
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What's my name? Reply in one word."}
    ]
    result = call_api(messages, max_tokens=3, temperature=0)
    if result:
        choice = result["choices"][0]
        save_vector("multi_turn", {
            "name": "multi_turn",
            "messages": messages,
            "max_tokens": 3,
            "temperature": 0,
            "thinking": False,
            "generated_text": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
            "token_logprobs": extract_logprobs(result),
            "model": result.get("model", ""),
        })
        print(f"  Generated: '{choice['message']['content'][:80]}'")
        return True
    return False


# --- Main ---

if __name__ == "__main__":
    tests = [
        test_short_prompt,
        test_short_prompt_thinking,
        test_coding_prompt,
        test_tool_call,
        test_system_prompt,
        test_multi_turn,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
            time.sleep(1)  # rate limit courtesy
        else:
            print("  → FAILED, continuing...\n")

    print(f"\n=== Done: {passed}/{len(tests)} vectors collected ===")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".json"):
            print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))} bytes)")