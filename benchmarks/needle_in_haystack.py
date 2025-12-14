"""
Needle in a Haystack Benchmark

Tests if models can retrieve specific information from long contexts.

The test embeds a "needle" (specific fact) within a "haystack" (long text)
and asks the model to retrieve it. Tests context window utilization.

Varies:
- Context length (1K, 5K, 10K, 20K+ tokens)
- Needle position (beginning, middle, end)
- Haystack content (essays, code, random text)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_clients import OpenRouterClient


@dataclass
class NeedleResult:
    """Result for a single needle test"""
    test_id: int
    context_length: int  # in tokens (approx words)
    needle_position: str  # "beginning", "middle", "end"
    needle: str
    model_output: str
    correct: bool
    exact_match: bool
    error_message: Optional[str] = None


class NeedleInHaystackBenchmark:
    """Tests long-context retrieval capabilities"""

    def __init__(self, model_config: Dict[str, Any], results_dir: str = "results/needle_haystack"):
        self.model_config = model_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenRouterClient(model_config)

    def generate_haystack(self, length_words: int) -> str:
        """Generate filler text for the haystack"""
        # Using Lorem Ipsum-style filler text
        sentences = [
            "The quick brown fox jumps over the lazy dog in the forest near the mountains.",
            "Technology advances rapidly changing how people communicate and work together daily.",
            "Scientists discovered new species in the ocean depths exploring unknown territories.",
            "Ancient civilizations built remarkable structures that still stand today across continents.",
            "Modern medicine continues evolving with breakthrough treatments and innovative research methods.",
            "Climate patterns shift gradually affecting ecosystems and weather systems worldwide.",
            "Artists express creativity through various mediums inspiring audiences with unique perspectives.",
            "Economic factors influence market trends creating opportunities and challenges for businesses.",
            "Education systems adapt to new teaching methods preparing students for future careers.",
            "Space exploration pushes boundaries revealing mysteries of the universe beyond Earth.",
        ]

        haystack = []
        current_length = 0

        while current_length < length_words:
            sentence = random.choice(sentences)
            haystack.append(sentence)
            current_length += len(sentence.split())

        return " ".join(haystack)

    def create_needle_statement(self) -> tuple:
        """Create a unique needle statement and question"""
        # Random unique facts that are unlikely to appear in filler text
        needles = [
            ("The secret code is 7392.", "What is the secret code?", "7392"),
            ("The magic word is 'xylophone'.", "What is the magic word?", "xylophone"),
            ("The meeting is scheduled for 3:47 PM.", "When is the meeting scheduled?", "3:47 PM"),
            ("The password is 'BlueMoon2024'.", "What is the password?", "BlueMoon2024"),
            ("The treasure is buried at coordinates 42.5N, 71.3W.", "Where is the treasure buried?", "42.5N, 71.3W"),
            ("The chef's special ingredient is cardamom.", "What is the chef's special ingredient?", "cardamom"),
            ("The document ID is REF-8472-QX.", "What is the document ID?", "REF-8472-QX"),
            ("The contact number is 555-0123.", "What is the contact number?", "555-0123"),
        ]
        return random.choice(needles)

    def insert_needle(self, haystack: str, needle: str, position: str) -> str:
        """Insert needle at specified position in haystack"""
        words = haystack.split()

        if position == "beginning":
            insert_idx = len(words) // 10  # 10% in
        elif position == "middle":
            insert_idx = len(words) // 2
        else:  # end
            insert_idx = int(len(words) * 0.9)  # 90% in

        words.insert(insert_idx, needle)
        return " ".join(words)

    def check_answer(self, model_output: str, expected_answer: str) -> tuple:
        """Check if model output contains the correct answer"""
        output_lower = model_output.lower()
        expected_lower = expected_answer.lower()

        # Exact match
        if expected_lower == output_lower.strip():
            return True, True

        # Contains the answer
        if expected_lower in output_lower:
            return True, False

        # Partial match (for things like coordinates or times)
        expected_parts = expected_lower.split()
        if len(expected_parts) > 1:
            # Check if all parts are present
            if all(part in output_lower for part in expected_parts):
                return True, False

        return False, False

    def run_single_test(
        self,
        test_id: int,
        context_length: int,
        position: str
    ) -> NeedleResult:
        """Run a single needle test"""

        # Generate haystack
        haystack = self.generate_haystack(context_length)

        # Create needle
        needle_statement, question, expected_answer = self.create_needle_statement()

        # Insert needle
        full_context = self.insert_needle(haystack, needle_statement, position)

        # Build prompt
        system_prompt = (
            "You are a helpful assistant. Answer the question based on the context provided. "
            "Be concise and specific."
        )

        user_message = f"Context:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"

        # Get model response
        try:
            response = self.client.chat(
                user_message=user_message,
                system_prompt=system_prompt
            )
            model_output = response['response'].strip()
            error = None
        except Exception as e:
            model_output = ""
            error = str(e)

        # Check correctness
        if error:
            correct = False
            exact_match = False
        else:
            correct, exact_match = self.check_answer(model_output, expected_answer)

        return NeedleResult(
            test_id=test_id,
            context_length=context_length,
            needle_position=position,
            needle=needle_statement,
            model_output=model_output,
            correct=correct,
            exact_match=exact_match,
            error_message=error
        )

    def run(self, context_lengths: List[int] = None):
        """Run full benchmark"""

        if context_lengths is None:
            # Default test suite: various context lengths
            context_lengths = [500, 1000, 2000, 4000, 8000]

        positions = ["beginning", "middle", "end"]

        print("=" * 70)
        print("Needle in a Haystack Benchmark")
        print("=" * 70)
        print(f"Model: {self.model_config['model_name']}")
        print(f"Context lengths: {context_lengths}")
        print(f"Positions: {positions}")
        print()

        results = []
        test_id = 0

        for length in context_lengths:
            for position in positions:
                test_id += 1
                print(f"[{test_id}] Testing: {length} words, needle at {position}")

                result = self.run_single_test(test_id, length, position)
                results.append(result)

                status = "CORRECT" if result.correct else "WRONG"
                exact = " (exact)" if result.exact_match else ""
                print(f"  Result: {status}{exact}")

                if not result.correct and result.error_message:
                    print(f"  Error: {result.error_message}")
                elif not result.correct:
                    print(f"  Expected: '{result.needle}'")
                    print(f"  Got: '{result.model_output[:100]}...'")

        # Summary
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        exact = sum(1 for r in results if r.exact_match)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"Exact matches: {exact}/{total} ({exact/total*100:.1f}%)")

        # Breakdown by context length
        print("\nBreakdown by context length:")
        for length in context_lengths:
            length_results = [r for r in results if r.context_length == length]
            length_correct = sum(1 for r in length_results if r.correct)
            print(f"  {length:5d} words: {length_correct}/{len(length_results)} ({length_correct/len(length_results)*100:.0f}%)")

        # Breakdown by position
        print("\nBreakdown by needle position:")
        for position in positions:
            pos_results = [r for r in results if r.needle_position == position]
            pos_correct = sum(1 for r in pos_results if r.correct)
            print(f"  {position:10s}: {pos_correct}/{len(pos_results)} ({pos_correct/len(pos_results)*100:.0f}%)")

        print("=" * 70)

        # Save results
        self._save_results(results, correct, total, exact)

    def _save_results(self, results, correct, total, exact):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_config['model_name'].replace('/', '_')

        summary = {
            "model": self.model_config['model_name'],
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "correct": correct,
            "accuracy": correct / total * 100,
            "exact_matches": exact,
            "exact_match_rate": exact / total * 100
        }

        summary_file = self.results_dir / f"summary_{model_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        detailed = [asdict(r) for r in results]
        details_file = self.results_dir / f"detailed_{model_name}_{timestamp}.json"
        with open(details_file, 'w') as f:
            json.dump(detailed, f, indent=2)

        print(f"\nResults saved to {self.results_dir}/")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    config = {
        "provider": "openrouter",
        "model_name": "openai/gpt-3.5-turbo",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 500,
        "temperature": 0.0
    }

    benchmark = NeedleInHaystackBenchmark(config)

    # Test with smaller contexts first (cheap)
    benchmark.run(context_lengths=[500, 1000, 2000, 4000])


if __name__ == "__main__":
    main()
