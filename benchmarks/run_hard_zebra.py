"""
Run Enhanced ReAct agent on the harder zebra puzzles.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

# Ensure repository root is on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.enhanced_react_agent import EnhancedReActAgent


def load_puzzles() -> List[Dict[str, Any]]:
    puzzles_path = Path(__file__).parent / "hard_zebra_puzzles.json"
    with puzzles_path.open() as f:
        return json.load(f)


def run_puzzle(agent: EnhancedReActAgent, puzzle: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Solve this zebra puzzle step by step. Use tools if needed (RAG, Python, search).

{puzzle['puzzle']}

Provide a concise final answer."""

    start = time.time()
    response = agent.reason(prompt)
    elapsed = time.time() - start

    expected = puzzle.get("solution")
    agent_success = response.success
    answer = response.final_answer or ""

    # Simple answer check: require all expected parts to appear in the answer (case-insensitive)
    correct = None
    if expected:
        expected_parts = [
            part.strip().lower()
            for part in expected.replace(" and ", ";").split(";")
            if part.strip()
        ]
        answer_lc = answer.lower()
        correct = all(part in answer_lc for part in expected_parts)

    return {
        "id": puzzle["id"],
        "difficulty": puzzle.get("difficulty", "unknown"),
        "agent_success": agent_success,
        "checked_success": bool(correct) if correct is not None else agent_success,
        "answer": answer,
        "tokens": response.total_tokens,
        "steps": len(response.steps),
        "time_s": round(elapsed, 2),
        "tools_used": list({step.action.value for step in response.steps}),
        "expected": expected,
    }


def main():
    load_dotenv()

    model = "anthropic/claude-3.5-sonnet"
    agent = EnhancedReActAgent(
        model_name=model,
        max_steps=18,
        temperature=0.0,
        verbose=False,
    )

    puzzles = load_puzzles()

    print(f"\n=== Running hard zebra puzzles with {model} ===")
    results = []
    for i, puzzle in enumerate(puzzles, 1):
        print(f"\n[{i}/{len(puzzles)}] {puzzle['id']} ({puzzle.get('difficulty', 'n/a')})")
        result = run_puzzle(agent, puzzle)
        results.append(result)
        status = "PASS" if result["checked_success"] else "FAIL"
        print(f"Status: {status} | Steps: {result['steps']} | Tokens: {result['tokens']} | Time: {result['time_s']}s")
        if result["answer"]:
            preview = result["answer"][:200].replace("\n", " ")
            print(f"Answer: {preview}")
        if puzzle.get("solution"):
            print(f"Expected: {puzzle['solution']}")
        time.sleep(2)

    print("\n=== Summary ===")
    for r in results:
        print(
            f"- {r['id']}: {'PASS' if r['checked_success'] else 'FAIL'} | steps={r['steps']} | tokens={r['tokens']} | time={r['time_s']}s | tools={r['tools_used']}"
        )


if __name__ == "__main__":
    main()
