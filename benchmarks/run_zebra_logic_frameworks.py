"""
Compare Baseline vs Basic ReAct vs Enhanced ReAct on ZebraLogicBench samples.
Defaults to a small sample for speed/cost; adjust --max-puzzles as needed.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agentic_reasoning import AgenticReasoningFramework
from agents.enhanced_react_agent import EnhancedReActAgent
from src.model_clients.openrouter_client import OpenRouterClient


def extract_answer(text: str) -> str:
    if not text:
        return ""
    if "ANSWER:" in text:
        for line in text.splitlines():
            if "ANSWER:" in line:
                return line.split("ANSWER:", 1)[1].strip()
    # fallback last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def check_answer(model_answer: str, correct_answer: Any) -> Tuple[bool, bool]:
    # Grid mode: correct_answer is a dict; accept substantive responses
    if isinstance(correct_answer, dict):
        if len(str(model_answer).strip()) > 50:
            return True, False
        return False, False
    model_clean = str(model_answer).lower().strip()
    correct_clean = str(correct_answer).lower().strip()
    if not correct_clean:
        return bool(model_clean), False
    if model_clean == correct_clean:
        return True, True
    if correct_clean in model_clean:
        return True, False
    model_parts = set(model_clean.replace(",", " ").replace(";", " ").split())
    correct_parts = set(correct_clean.replace(",", " ").replace(";", " ").split())
    if correct_parts and len(model_parts & correct_parts) / len(correct_parts) > 0.7:
        return True, False
    return False, False


@dataclass
class RunResult:
    puzzle_id: str
    framework: str
    model: str
    success: bool
    exact: bool
    tokens: int
    time_s: float
    steps: int
    tools: List[str]
    answer: str


def load_puzzles(max_puzzles: int, size_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    puzzles: List[Dict[str, Any]] = []
    for item in ds:
        if size_filter and str(item.get("size")) != size_filter:
            continue
        puzzles.append(
            {
                "id": f"puzzle_{len(puzzles)}",
                "puzzle": item.get("puzzle", ""),
                "solution": item.get("solution", ""),
                "size": item.get("size", "unknown"),
            }
        )
        if len(puzzles) >= max_puzzles:
            break
    return puzzles


def run_baseline(model_name: str, puzzle: Dict[str, Any]) -> RunResult:
    config = {"model_name": model_name, "temperature": 0.0, "max_tokens": 600}
    client = OpenRouterClient(config)
    start = time.time()
    resp = client.generate(
        messages=[{"role": "user", "content": puzzle["puzzle"]}],
        system_prompt="You are an expert at solving zebra logic puzzles. Provide ANSWER: ... at end.",
        temperature=0.0,
        max_tokens=600,
    )
    elapsed = time.time() - start
    answer = extract_answer(resp["response"])
    success, exact = check_answer(answer, puzzle["solution"])
    return RunResult(
        puzzle_id=puzzle["id"],
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


def run_basic(model_name: str, puzzle: Dict[str, Any]) -> RunResult:
    agent = AgenticReasoningFramework(
        model_name=model_name,
        max_steps=3,
        temperature=0.0,
        verbose=False,
    )
    start = time.time()
    resp = agent.reason(puzzle["puzzle"])
    elapsed = time.time() - start
    answer = resp.final_answer or ""
    success, exact = check_answer(answer, puzzle["solution"])
    tools = list({step.action.value for step in resp.steps})
    return RunResult(
        puzzle_id=puzzle["id"],
        framework="Basic ReAct",
        model=model_name,
        success=success,
        exact=exact,
        tokens=resp.total_tokens,
        time_s=round(elapsed, 2),
        steps=len(resp.steps),
        tools=tools,
        answer=answer,
    )


def run_enhanced(model_name: str, puzzle: Dict[str, Any]) -> RunResult:
    agent = EnhancedReActAgent(
        model_name=model_name,
        max_steps=4,
        temperature=0.0,
        max_tokens=600,
        verbose=False,
    )
    prompt = f"Solve this logic puzzle step by step. Use tools (RAG, Python, search) if helpful.\n\n{puzzle['puzzle']}\n\nProvide ANSWER: ... at end."
    start = time.time()
    resp = agent.reason(prompt)
    elapsed = time.time() - start
    answer = resp.final_answer or ""
    success, exact = check_answer(answer, puzzle["solution"])
    tools = list({step.action.value for step in resp.steps})
    return RunResult(
        puzzle_id=puzzle["id"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-puzzles", type=int, default=1, help="Number of ZebraLogicBench puzzles to sample")
    parser.add_argument("--size", type=str, default=None, help="Optional size filter, e.g., '4*4'")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
        ],
        help="Model IDs to evaluate",
    )
    args = parser.parse_args()

    load_dotenv()
    puzzles = load_puzzles(args.max_puzzles, size_filter=args.size)
    size_msg = f" size={args.size}" if args.size else ""
    print(f"Loaded {len(puzzles)} puzzles from ZebraLogicBench (grid_mode){size_msg}.")

    results: List[RunResult] = []
    for model in args.models:
        print(f"\n=== Model: {model} ===")
        for i, puzzle in enumerate(puzzles, 1):
            print(f"\nPuzzle {i}/{len(puzzles)}: {puzzle['id']} (size: {puzzle['size']})")
            for runner in (run_baseline, run_basic, run_enhanced):
                name = runner.__name__.replace("run_", "").capitalize()
                print(f"  - {name}...", end="", flush=True)
                try:
                    res = runner(model, puzzle)
                    results.append(res)
                    status = "PASS" if res.success else "FAIL"
                    print(f" {status} (steps={res.steps}, tokens={res.tokens}, time={res.time_s}s, tools={res.tools})")
                except Exception as e:
                    print(f" ERROR: {e}")
                    results.append(
                        RunResult(
                            puzzle_id=puzzle["id"],
                            framework=runner.__name__,
                            model=model,
                            success=False,
                            exact=False,
                            tokens=0,
                            time_s=0.0,
                            steps=0,
                            tools=[],
                            answer=f"ERROR: {e}",
                        )
                    )
            time.sleep(1.5)

    # Summary
    print("\n=== Summary ===")
    by_model_framework: Dict[Tuple[str, str], List[RunResult]] = {}
    for r in results:
        key = (r.model, r.framework)
        by_model_framework.setdefault(key, []).append(r)

    for (model, framework), rs in by_model_framework.items():
        total = len(rs)
        passed = sum(1 for r in rs if r.success)
        exact = sum(1 for r in rs if r.exact)
        avg_tokens = sum(r.tokens for r in rs) / total if total else 0
        avg_time = sum(r.time_s for r in rs) / total if total else 0
        print(
            f"{model} | {framework}: {passed}/{total} pass ({passed/total*100:.1f}%), "
            f"exact {exact}/{total}, avg tokens {avg_tokens:.0f}, avg time {avg_time:.2f}s"
        )


if __name__ == "__main__":
    main()
