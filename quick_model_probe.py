"""
Quickly probe which OpenRouter models respond without long waits.

Uses short timeouts and single retry to fail fast.
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model_clients.openrouter_client import OpenRouterClient  # noqa: E402


def probe_model(model_id: str, label: str) -> dict:
    """Ping a model with a tiny prompt and short timeout."""
    config = {
        "model_name": model_id,
        "temperature": 0.0,
        "max_tokens": 32,
        "timeout": 10,       # seconds per request
        "max_retries": 1,    # fail fast
    }
    client = OpenRouterClient(config)

    start = time.time()
    try:
        resp = client.generate(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            system_prompt="You are a probe endpoint. Answer exactly 'pong'.",
            temperature=0.0,
            max_tokens=8,
        )
        elapsed = time.time() - start
        return {
            "model": label,
            "status": "ok",
            "time_s": round(elapsed, 2),
            "tokens": resp.get("usage", {}).get("total_tokens", 0),
            "response": resp.get("response", "")[:120],
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": label,
            "status": "error",
            "time_s": round(elapsed, 2),
            "error": str(e),
        }


def main():
    load_dotenv()

    models = [
        ("deepseek/deepseek-chat", "DeepSeek-v3"),
        ("openai/gpt-4o-mini", "GPT-4o-mini"),
        ("anthropic/claude-3.5-sonnet", "Claude-3.5-Sonnet"),
    ]

    print("\n=== Quick OpenRouter Probe (10s timeout, 1 retry) ===")
    results = []
    for mid, label in models:
        print(f"Probing {label}...", end="", flush=True)
        result = probe_model(mid, label)
        results.append(result)
        if result["status"] == "ok":
            print(f" OK ({result['time_s']}s, tokens={result['tokens']})")
        else:
            print(f" ERROR ({result['time_s']}s): {result['error']}")

    print("\nSummary:")
    for r in results:
        if r["status"] == "ok":
            print(f"- {r['model']}: OK in {r['time_s']}s (tokens={r['tokens']}) response={r['response']}")
        else:
            print(f"- {r['model']}: ERROR in {r['time_s']}s -> {r['error']}")


if __name__ == "__main__":
    main()
