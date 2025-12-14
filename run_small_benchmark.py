"""
Author: Aaron Genkin (amg454)
Purpose: Benchmark testing suite for evaluating agent performance across different reasoning tasks

This script tests enhanced reasoning agents on a variety of tasks including mathematical
reasoning, logical deduction, and code generation problems.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from agents.enhanced_react_agent import EnhancedReActAgent


def load_sample_tasks():
    task_file = Path("benchmarks/custom_tasks/sample_tasks.json")
    with open(task_file, 'r') as f:
        tasks = json.load(f)
    return tasks


def run_task(agent, task, model_name):
    print(f"\n{'='*70}")
    print(f"Task: {task['id']} ({task['type']}) - {task['difficulty']}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    prompt = task['prompt']
    print(f"\nPrompt: {prompt[:200]}...")

    try:
        start = time.time()
        response = agent.reason(prompt)
        elapsed = time.time() - start

        result = {
            "task_id": task['id'],
            "task_type": task['type'],
            "difficulty": task['difficulty'],
            "model": model_name,
            "success": response.success,
            "steps": len(response.steps),
            "tokens": response.total_tokens,
            "time": elapsed,
            "answer": response.final_answer,
            "expected": task.get('expected_answer', 'N/A')
        }

        print(f"\n{'='*70}")
        print(f"RESULT:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Time: {result['time']:.2f}s")
        if response.final_answer:
            print(f"  Answer: {response.final_answer[:150]}...")
        print(f"{'='*70}")

        return result

    except Exception as e:
        print(f"\n[ERROR] Task failed: {e}")
        return {
            "task_id": task['id'],
            "task_type": task['type'],
            "model": model_name,
            "success": False,
            "error": str(e)
        }


def run_small_benchmark(model_name, model_display_name, max_tasks=3):
    print(f"\n{'#'*70}")
    print(f"BENCHMARK: {model_display_name}")
    print(f"Model ID: {model_name}")
    print(f"{'#'*70}")

    all_tasks = load_sample_tasks()

    # Pick different types of tasks for variety
    selected_tasks = []
    task_types_wanted = ['math', 'logic', 'code']

    for task_type in task_types_wanted:
        for task in all_tasks:
            if task['type'] == task_type and len(selected_tasks) < max_tasks:
                selected_tasks.append(task)
                break

    print(f"\nTesting {len(selected_tasks)} tasks:")
    for task in selected_tasks:
        print(f"  - {task['id']}: {task['type']} ({task['difficulty']})")

    agent = EnhancedReActAgent(
        model_name=model_name,
        max_steps=15,
        verbose=False
    )

    results = []

    for i, task in enumerate(selected_tasks, 1):
        print(f"\n\n{'*'*70}")
        print(f"TASK {i}/{len(selected_tasks)}")
        print(f"{'*'*70}")

        result = run_task(agent, task, model_display_name)
        results.append(result)

        if i < len(selected_tasks):
            time.sleep(2)

    return results


def print_summary(all_results):
    print(f"\n\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    by_model = {}
    for r in all_results:
        model = r['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    for model, results in by_model.items():
        print(f"\n{model}:")
        print(f"  Tasks: {len(results)}")

        successes = [r for r in results if r.get('success', False)]
        success_rate = len(successes)/len(results)*100
        print(f"  Success: {len(successes)}/{len(results)} ({success_rate:.1f}%)")

        if successes:
            avg_tokens = sum(r['tokens'] for r in successes) / len(successes)
            avg_time = sum(r['time'] for r in successes) / len(successes)
            avg_steps = sum(r['steps'] for r in successes) / len(successes)

            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Avg steps: {avg_steps:.1f}")

    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    print(f"{'Task':<15} {'Type':<8} {'Model':<20} {'Success':<10} {'Steps':<8} {'Tokens':<10}")
    print("-"*70)

    for r in all_results:
        success_str = "[PASS]" if r.get('success') else "[FAIL]"
        steps = r.get('steps', 0)
        tokens = r.get('tokens', 0)
        print(f"{r['task_id']:<15} {r['task_type']:<8} {r['model']:<20} {success_str:<10} {steps:<8} {tokens:<10}")

    print(f"{'='*70}")


def main():
    print("\n" + "#"*70)
    print("SMALL BENCHMARK TEST")
    print("Testing enhanced agent on sample tasks")
    print("#"*70)

    models = [
        ("openai/gpt-4o-mini", "GPT-4o-mini"),
        ("deepseek/deepseek-chat", "DeepSeek-v3"),
    ]

    all_results = []

    for i, (model_id, model_name) in enumerate(models, 1):
        print(f"\n\n{'='*70}")
        print(f"MODEL {i}/{len(models)}")
        print(f"{'='*70}")

        results = run_small_benchmark(model_id, model_name, max_tasks=3)
        all_results.extend(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/small_benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"results_{model_name.replace(' ', '_')}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] {output_file}")

        if i < len(models):
            print("\n>>> Waiting 5 seconds before next model...")
            time.sleep(5)

    print_summary(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"combined_results_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Combined results: {combined_file}")

    print("\n" + "#"*70)
    print("BENCHMARK COMPLETE")
    print("#"*70)


if __name__ == "__main__":
    main()
