"""
Author: Aaron Genkin (amg454)
Purpose: Comprehensive comparison of different agent frameworks and their performance

This script compares three approaches:
1. Baseline (no agent - direct prompting)
2. Basic ReAct (search + calculate tools)
3. Enhanced ReAct (Python + RAG + Search tools)

Tests are run on logic puzzles to demonstrate the impact of different toolsets.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from agents.enhanced_react_agent import EnhancedReActAgent
from agents.agentic_reasoning import AgenticReasoningFramework
from src.model_clients.openrouter_client import OpenRouterClient
from dotenv import load_dotenv
import os

load_dotenv()


def get_test_puzzle():
    return {
        'id': 'simple_zebra',
        'puzzle': """There are 5 houses in a row, numbered 1 to 5.

Facts:
1. The Norwegian lives in the first house.
2. The Norwegian lives next to the blue house.
3. The Englishman lives in the red house.
4. The green house is immediately to the right of the ivory house.
5. The Spaniard owns a dog.

Question: What is the color of the second house?""",
        'expected_answer': 'blue',
        'difficulty': 'easy'
    }


def test_baseline(model_name, model_display, puzzle):
    print(f"\n{'='*80}")
    print(f"BASELINE TEST: {model_display}")
    print("No agent framework - direct prompting only")
    print(f"{'='*80}")

    config = {
        "model_name": model_name,
        "temperature": 0.1,
        "max_tokens": 1000
    }

    client = OpenRouterClient(config)

    system_prompt = "You are an expert at solving logic puzzles. Think step by step and provide your final answer."
    user_prompt = f"{puzzle['puzzle']}\n\nProvide your answer clearly."

    try:
        start = time.time()
        response = client.generate(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )
        elapsed = time.time() - start

        result = {
            'framework': 'Baseline (No Agent)',
            'model': model_display,
            'success': True,
            'answer': response['response'],
            'tokens': response['usage']['total_tokens'],
            'time': elapsed,
            'steps': 1,
            'tools_used': []
        }

        print(f"\nAnswer: {response['response'][:200]}...")
        print(f"Tokens: {result['tokens']}, Time: {result['time']:.2f}s")

        return result

    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            'framework': 'Baseline (No Agent)',
            'model': model_display,
            'success': False,
            'error': str(e)
        }


def test_basic_react(model_name, model_display, puzzle):
    print(f"\n{'='*80}")
    print(f"BASIC REACT TEST: {model_display}")
    print("Using: Search + Calculate tools")
    print(f"{'='*80}")

    try:
        agent = AgenticReasoningFramework(
            model_name=model_name,
            max_steps=10,
            temperature=0.1,
            verbose=False
        )

        start = time.time()
        response = agent.reason(puzzle['puzzle'])
        elapsed = time.time() - start

        tools_used = set()
        for step in response.steps:
            tools_used.add(step.action.value)

        result = {
            'framework': 'Basic ReAct',
            'model': model_display,
            'success': response.success,
            'answer': response.final_answer,
            'tokens': response.total_tokens,
            'time': elapsed,
            'steps': len(response.steps),
            'tools_used': list(tools_used)
        }

        print(f"\nSuccess: {result['success']}")
        print(f"Answer: {result['answer'][:200] if result['answer'] else 'None'}...")
        print(f"Steps: {result['steps']}, Tools: {result['tools_used']}")
        print(f"Tokens: {result['tokens']}, Time: {result['time']:.2f}s")

        return result

    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            'framework': 'Basic ReAct',
            'model': model_display,
            'success': False,
            'error': str(e)
        }


def test_enhanced_react(model_name, model_display, puzzle):
    print(f"\n{'='*80}")
    print(f"ENHANCED REACT TEST: {model_display}")
    print("Using: Python + RAG + Search tools")
    print(f"{'='*80}")

    try:
        agent = EnhancedReActAgent(
            model_name=model_name,
            max_steps=12,
            temperature=0.1,
            verbose=False
        )

        enhanced_prompt = f"""Solve this logic puzzle step by step.

First, check the knowledge base for strategies on solving logic puzzles.
Then use Python if needed for constraint tracking.

{puzzle['puzzle']}

Provide your final answer clearly."""

        start = time.time()
        response = agent.reason(enhanced_prompt)
        elapsed = time.time() - start

        tools_used = set()
        for step in response.steps:
            tools_used.add(step.action.value)

        result = {
            'framework': 'Enhanced ReAct',
            'model': model_display,
            'success': response.success,
            'answer': response.final_answer,
            'tokens': response.total_tokens,
            'time': elapsed,
            'steps': len(response.steps),
            'tools_used': list(tools_used)
        }

        print(f"\nSuccess: {result['success']}")
        print(f"Answer: {result['answer'][:200] if result['answer'] else 'None'}...")
        print(f"Steps: {result['steps']}, Tools: {result['tools_used']}")
        print(f"Tokens: {result['tokens']}, Time: {result['time']:.2f}s")

        return result

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            'framework': 'Enhanced ReAct',
            'model': model_display,
            'success': False,
            'error': str(e)
        }


def run_full_comparison(model_id, model_display):
    print(f"\n\n{'#'*80}")
    print(f"TESTING MODEL: {model_display}")
    print(f"Model ID: {model_id}")
    print(f"{'#'*80}")

    puzzle = get_test_puzzle()

    print(f"\nPuzzle: {puzzle['puzzle'][:150]}...")
    print(f"Expected answer: {puzzle['expected_answer']}")

    results = []

    print(f"\n{'*'*80}")
    print("TEST 1/3: BASELINE (NO AGENT)")
    print(f"{'*'*80}")
    result1 = test_baseline(model_id, model_display, puzzle)
    results.append(result1)
    time.sleep(2)

    print(f"\n{'*'*80}")
    print("TEST 2/3: BASIC REACT")
    print(f"{'*'*80}")
    result2 = test_basic_react(model_id, model_display, puzzle)
    results.append(result2)
    time.sleep(2)

    print(f"\n{'*'*80}")
    print("TEST 3/3: ENHANCED REACT")
    print(f"{'*'*80}")
    result3 = test_enhanced_react(model_id, model_display, puzzle)
    results.append(result3)

    return results


def print_comparison_table(all_results):
    print(f"\n\n{'='*100}")
    print("FRAMEWORK COMPARISON RESULTS")
    print(f"{'='*100}")

    print(f"\n{'Model':<20} {'Framework':<20} {'Success':<10} {'Steps':<8} {'Tokens':<10} {'Time (s)':<10} {'Tools Used'}")
    print("-"*100)

    for r in all_results:
        if 'error' not in r:
            success_str = "[PASS]" if r.get('success', False) else "[FAIL]"
            tools_str = ', '.join(r.get('tools_used', [])) if r.get('tools_used') else 'None'
            print(f"{r['model']:<20} {r['framework']:<20} {success_str:<10} {r.get('steps', 0):<8} {r.get('tokens', 0):<10} {r.get('time', 0):<10.2f} {tools_str}")
        else:
            print(f"{r['model']:<20} {r['framework']:<20} [ERROR]    {r['error'][:50]}")

    print(f"{'='*100}")


def print_model_comparison(all_results):
    print(f"\n\n{'='*100}")
    print("PER-MODEL IMPROVEMENT ANALYSIS")
    print(f"{'='*100}")

    by_model = {}
    for r in all_results:
        model = r['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    for model, results in by_model.items():
        print(f"\n{model}:")
        print("-" * 80)

        baseline = next((r for r in results if r['framework'] == 'Baseline (No Agent)'), None)
        basic = next((r for r in results if r['framework'] == 'Basic ReAct'), None)
        enhanced = next((r for r in results if r['framework'] == 'Enhanced ReAct'), None)

        if baseline:
            print(f"  Baseline:       Success: {baseline.get('success', False):<6} Tokens: {baseline.get('tokens', 0):<6} Time: {baseline.get('time', 0):.2f}s")
        if basic:
            print(f"  Basic ReAct:    Success: {basic.get('success', False):<6} Tokens: {basic.get('tokens', 0):<6} Time: {basic.get('time', 0):.2f}s")
        if enhanced:
            print(f"  Enhanced ReAct: Success: {enhanced.get('success', False):<6} Tokens: {enhanced.get('tokens', 0):<6} Time: {enhanced.get('time', 0):.2f}s")

        if baseline and enhanced and baseline.get('tokens') and enhanced.get('tokens'):
            token_diff = enhanced['tokens'] - baseline['tokens']
            time_diff = enhanced.get('time', 0) - baseline.get('time', 0)
            print(f"\n  Improvement (Enhanced vs Baseline):")
            print(f"    Token difference: {token_diff:+d} ({(token_diff/baseline['tokens']*100):+.1f}%)")
            print(f"    Time difference: {time_diff:+.2f}s")


def main():
    print("\n" + "#"*80)
    print("COMPREHENSIVE AGENT FRAMEWORK COMPARISON")
    print("Testing 3 models × 3 frameworks = 9 total tests")
    print("#"*80)

    models = [
        ("deepseek/deepseek-chat", "DeepSeek-v3"),
        ("openai/gpt-4o-mini", "GPT-4o-mini"),
        ("anthropic/claude-3.5-sonnet", "Claude-3.5-Sonnet"),
    ]

    all_results = []

    for i, (model_id, model_name) in enumerate(models, 1):
        print(f"\n\n{'='*80}")
        print(f"MODEL {i}/{len(models)}")
        print(f"{'='*80}")

        try:
            results = run_full_comparison(model_id, model_name)
            all_results.extend(results)

            # Save per-model results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results/framework_comparison")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{model_name.replace(' ', '_')}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n[Saved] {output_file}")

        except Exception as e:
            print(f"[ERROR] Model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

        if i < len(models):
            print("\n>>> Waiting 5 seconds before next model...")
            time.sleep(5)

    # Print comparison tables
    print_comparison_table(all_results)
    print_model_comparison(all_results)

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/framework_comparison")
    combined_file = output_dir / f"combined_comparison_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Combined results: {combined_file}")

    # Summary statistics
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    frameworks = ['Baseline (No Agent)', 'Basic ReAct', 'Enhanced ReAct']
    for framework in frameworks:
        framework_results = [r for r in all_results if r.get('framework') == framework and 'error' not in r]
        if framework_results:
            successes = sum(1 for r in framework_results if r.get('success', False))
            avg_tokens = sum(r.get('tokens', 0) for r in framework_results) / len(framework_results)
            avg_time = sum(r.get('time', 0) for r in framework_results) / len(framework_results)

            print(f"\n{framework}:")
            print(f"  Success rate: {successes}/{len(framework_results)} ({successes/len(framework_results)*100:.1f}%)")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg time: {avg_time:.2f}s")

    print("\n" + "#"*80)
    print("COMPARISON COMPLETE")
    print("#"*80)


if __name__ == "__main__":
    main()
