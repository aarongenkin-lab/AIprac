"""
Run Baseline vs Enhanced agent on a local WikiTableQuestions mini-set (JSONL).
Requires: benchmarks/wtq_mini/wtq_mini.jsonl (built via prepare_wtq_mini.py)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from agents.enhanced_react_agent import EnhancedReActAgent  # noqa: E402
from src.model_clients.openrouter_client import OpenRouterClient  # noqa: E402


def extract_answer(text: str) -> str:
    if not text:
        return ""
    if "ANSWER:" in text:
        for line in text.splitlines():
            if "ANSWER:" in line:
                return line.split("ANSWER:", 1)[1].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def check_answer(model_answer: str, correct_answer: Any) -> Tuple[bool, bool]:
    model_clean = str(model_answer).lower().strip()
    correct_clean = str(correct_answer).lower().strip()
    if not correct_clean:
        return bool(model_clean), False
    if model_clean == correct_clean:
        return True, True
    if correct_clean in model_clean:
        return True, False
    return False, False


@dataclass
class RunResult:
    qid: str
    framework: str
    model: str
    success: bool
    exact: bool
    tokens: int
    time_s: float
    steps: int
    tools: List[str]
    answer: str


def load_mini(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(items) >= max_samples:
                break
            items.append(json.loads(line))
    return items


def run_baseline(model_name: str, ex: Dict[str, Any], max_tokens: int = 900) -> RunResult:
    config = {"model_name": model_name, "temperature": 0.0, "max_tokens": max_tokens}
    client = OpenRouterClient(config)
    prompt = f"""Table:
{ex['table_text']}

Question: {ex['question']}

Provide the answer. End with: ANSWER: <answer>"""
    start = time.time()
    resp = client.generate(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="You answer table questions exactly. Use the table above.",
        temperature=0.0,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start
    answer = extract_answer(resp["response"])
    success, exact = check_answer(answer, ex["answer"])
    return RunResult(
        qid=ex["id"],
        framework="Baseline",
        model=model_name,
        success=success,
        exact=exact,
        tokens=resp.get("usage", {}).get("total_tokens", 0),
        time_s=round(elapsed, 2),
        steps=1,
        tools=[],
        answer=answer,
    )


def run_enhanced(model_name: str, ex: Dict[str, Any], max_tokens: int = 900, max_steps: int = 6) -> RunResult:
    agent = EnhancedReActAgent(
        model_name=model_name,
        max_steps=max_steps,
        temperature=0.0,
        max_tokens=max_tokens,
        verbose=False,
    )
    prompt = f"""You are given a table and a question. Use python for any lookups or aggregations. Show a concise chain, then final answer.

Table:
{ex['table_text']}

Question: {ex['question']}

Provide final answer as: ANSWER: <answer>"""
    start = time.time()
    resp = agent.reason(prompt)
    elapsed = time.time() - start
    answer = resp.final_answer or ""
    success, exact = check_answer(answer, ex["answer"])
    tools = list({step.action.value for step in resp.steps})
    return RunResult(
        qid=ex["id"],
        framework="Enhanced ReAct",
        model=model_name,
        success=success,
        exact=exact,
        tokens=resp.total_tokens,
        time_s=round(elapsed, 2),
        steps=len(resp.steps),
        tools=tools,
        answer=answer,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/wtq_mini/wtq_mini.jsonl", help="Path to mini JSONL")
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["deepseek/deepseek-chat", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
    )
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    load_dotenv()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Mini set not found: {data_path}. Build with prepare_wtq_mini.py")
    items = load_mini(data_path, args.max_samples)
    print(f"Loaded {len(items)} WTQ mini examples from {data_path}")

    results: List[RunResult] = []
    for model in args.models:
        print(f"\n=== Model: {model} ===")
        for i, ex in enumerate(items, 1):
            print(f"\nSample {i}/{len(items)}: {ex['id']}")
            # Baseline
            try:
                r = run_baseline(model, ex, max_tokens=args.max_tokens)
                results.append(r)
                print(f"  Baseline: {'PASS' if r.success else 'FAIL'} (tokens={r.tokens}, time={r.time_s}s)")
            except Exception as e:
                print(f"  Baseline ERROR: {e}")
            # Enhanced
            try:
                r = run_enhanced(model, ex, max_tokens=args.max_tokens, max_steps=args.max_steps)
                results.append(r)
                print(f"  Enhanced: {'PASS' if r.success else 'FAIL'} (tokens={r.tokens}, time={r.time_s}s, tools={r.tools})")
            except Exception as e:
                print(f"  Enhanced ERROR: {e}")
            time.sleep(1.0)

    # Summary
    print("\n=== Summary ===")
    by_model_framework: Dict[Tuple[str, str], List[RunResult]] = {}
    for r in results:
        key = (r.model, r.framework)
        by_model_framework.setdefault(key, []).append(r)

    for (model, fw), rs in by_model_framework.items():
        total = len(rs)
        passed = sum(1 for x in rs if x.success)
        exact = sum(1 for x in rs if x.exact)
        avg_tokens = sum(x.tokens for x in rs) / total if total else 0
        avg_time = sum(x.time_s for x in rs) / total if total else 0
        print(f"{model} | {fw}: {passed}/{total} pass ({passed/total*100:.1f}%), exact {exact}/{total}, avg tokens {avg_tokens:.0f}, avg time {avg_time:.2f}s")


if __name__ == "__main__":
    main()
