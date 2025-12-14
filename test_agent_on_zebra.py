"""
Author: Aaron Genkin (amg454)
Purpose: Test agents on zebra logic puzzles and constraint satisfaction problems

This script evaluates reasoning agents on classic logic puzzles that require
constraint satisfaction and deductive reasoning.
"""

import sys
from pathlib import Path
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from agents.enhanced_react_agent import EnhancedReActAgent


def test_on_zebra_puzzles(model_name, model_display_name, num_puzzles=3):
    print(f"\n{'='*80}")
    print(f"ZEBRA PUZZLE TEST: {model_display_name}")
    print(f"Model ID: {model_name}")
    print(f"{'='*80}")

    try:
        from datasets import load_dataset
        print("\nLoading ZebraLogicBench dataset...")
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        print(f"Loaded {len(dataset)} puzzles")

        puzzles = []
        for i in range(min(num_puzzles, len(dataset))):
            puzzles.append({
                'id': i,
                'puzzle': dataset[i]['puzzle'],
                'solution': dataset[i]['solution'],
                'size': dataset[i].get('size', 'unknown')
            })

    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("\nUsing a sample puzzle instead...")

        puzzles = [{
            'id': 0,
            'puzzle': """There are 5 houses in a row, numbered 1 to 5 from left to right.

Facts:
1. The Norwegian lives in the first house.
2. The Norwegian lives next to the blue house.
3. The green house is immediately to the right of the ivory house.
4. The Englishman lives in the red house.
5. The Spaniard owns a dog.

What color is the second house?""",
            'solution': 'blue',
            'size': '5x5'
        }]

    agent = EnhancedReActAgent(
        model_name=model_name,
        max_steps=20,
        verbose=True
    )

    results = []

    for i, puzzle in enumerate(puzzles, 1):
        print(f"\n\n{'*'*80}")
        print(f"PUZZLE {i}/{len(puzzles)} (Size: {puzzle['size']})")
        print(f"{'*'*80}")

        enhanced_prompt = f"""Solve this zebra logic puzzle step by step.

First, check the knowledge base for strategies on solving zebra puzzles.
Then use Python if needed for constraint checking.

{puzzle['puzzle']}

Provide your final answer clearly."""

        print(f"\nPuzzle preview: {puzzle['puzzle'][:200]}...")

        try:
            start = time.time()
            response = agent.reason(enhanced_prompt)
            elapsed = time.time() - start

            result = {
                'puzzle_id': puzzle['id'],
                'puzzle_size': puzzle['size'],
                'model': model_display_name,
                'success': response.success,
                'steps': len(response.steps),
                'tokens': response.total_tokens,
                'time': elapsed,
                'answer': response.final_answer,
                'expected': puzzle['solution']
            }

            print(f"\n{'='*80}")
            print(f"RESULT:")
            print(f"  Success: {result['success']}")
            print(f"  Steps: {result['steps']}")
            print(f"  Tokens: {result['tokens']}")
            print(f"  Time: {result['time']:.2f}s")
            if response.final_answer:
                print(f"  Answer: {response.final_answer[:300]}...")
            print(f"  Expected: {puzzle['solution']}")
            print(f"{'='*80}")

            results.append(result)

        except Exception as e:
            print(f"\n[ERROR] Puzzle {i} failed: {e}")
            results.append({
                'puzzle_id': puzzle['id'],
                'model': model_display_name,
                'success': False,
                'error': str(e)
            })

        if i < len(puzzles):
            time.sleep(3)

    return results


def main():
    print("\n" + "#"*80)
    print("ZEBRA PUZZLE TEST WITH ENHANCED AGENT")
    print("Testing agent with RAG (zebra strategies) and Python")
    print("#"*80)

    models = [
        ("deepseek/deepseek-chat", "DeepSeek-v3"),
        ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ]

    all_results = []

    for i, (model_id, model_name) in enumerate(models, 1):
        print(f"\n\n{'='*80}")
        print(f"MODEL {i}/{len(models)}")
        print(f"{'='*80}")

        results = test_on_zebra_puzzles(model_id, model_name, num_puzzles=2)
        all_results.extend(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/zebra_agent_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"results_{model_name.replace(' ', '_')}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] {output_file}")

        if i < len(models):
            print("\n>>> Waiting 5 seconds before next model...")
            time.sleep(5)

    print(f"\n\n{'='*80}")
    print("ZEBRA PUZZLE TEST SUMMARY")
    print(f"{'='*80}")

    by_model = {}
    for r in all_results:
        model = r['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    for model, results in by_model.items():
        print(f"\n{model}:")
        successes = [r for r in results if r.get('success', False)]
        print(f"  Puzzles attempted: {len(results)}")
        print(f"  Completed: {len(successes)}/{len(results)}")

        if successes:
            avg_tokens = sum(r['tokens'] for r in successes) / len(successes)
            avg_time = sum(r['time'] for r in successes) / len(successes)
            avg_steps = sum(r['steps'] for r in successes) / len(successes)

            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Avg steps: {avg_steps:.1f}")

    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    print(f"{'Puzzle':<10} {'Size':<8} {'Model':<20} {'Success':<10} {'Steps':<8} {'Tokens':<10}")
    print("-"*80)

    for r in all_results:
        if 'error' not in r:
            success_str = "[PASS]" if r.get('success') else "[FAIL]"
            print(f"{r['puzzle_id']:<10} {r['puzzle_size']:<8} {r['model']:<20} {success_str:<10} {r['steps']:<8} {r['tokens']:<10}")
        else:
            print(f"{r['puzzle_id']:<10} {'N/A':<8} {r['model']:<20} [ERROR]    {r['error'][:30]}")

    print(f"{'='*80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"combined_zebra_results_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Combined results: {combined_file}")

    print("\n" + "#"*80)
    print("ZEBRA PUZZLE TEST COMPLETE")
    print("#"*80)


if __name__ == "__main__":
    main()
