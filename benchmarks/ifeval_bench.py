"""
IFEval: Instruction Following Evaluation

Tests verifiable format constraints:
- Word count requirements
- Forbidden words
- Structural requirements (bullet points, sections)
- Format constraints (all caps, no punctuation, etc.)

Uses objective grading (regex/logic), not LLM-as-judge.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_clients import OpenRouterClient


@dataclass
class ConstraintResult:
    """Result for a single constraint check"""
    constraint_type: str
    passed: bool
    expected: Any
    actual: Any
    error_message: Optional[str] = None


@dataclass
class IFEvalResult:
    """Result for a single prompt"""
    prompt_id: int
    prompt: str
    constraints: List[str]
    model_output: str
    passed: bool
    constraint_results: List[ConstraintResult]
    pass_rate: float  # % of constraints passed


class ConstraintChecker:
    """Validates format constraints on text output"""

    @staticmethod
    def check_word_count(text: str, min_words: int, max_words: Optional[int] = None) -> ConstraintResult:
        """Check if text meets word count requirements"""
        words = text.split()
        actual = len(words)

        if max_words:
            passed = min_words <= actual <= max_words
            expected = f"{min_words}-{max_words} words"
        else:
            passed = actual >= min_words
            expected = f"at least {min_words} words"

        return ConstraintResult(
            constraint_type="word_count",
            passed=passed,
            expected=expected,
            actual=f"{actual} words",
            error_message=None if passed else f"Got {actual} words, needed {expected}"
        )

    @staticmethod
    def check_forbidden_words(text: str, forbidden: List[str]) -> ConstraintResult:
        """Check if text avoids forbidden words"""
        text_lower = text.lower()
        found = [word for word in forbidden if word.lower() in text_lower]

        passed = len(found) == 0

        return ConstraintResult(
            constraint_type="forbidden_words",
            passed=passed,
            expected=f"No use of: {', '.join(forbidden)}",
            actual=f"Found: {', '.join(found)}" if found else "None found",
            error_message=None if passed else f"Used forbidden words: {', '.join(found)}"
        )

    @staticmethod
    def check_bullet_points(text: str, required_count: int) -> ConstraintResult:
        """Check if text has required number of bullet points"""
        # Match lines starting with -, *, •, or numbered bullets
        bullets = re.findall(r'^[\s]*[-*•\d]+[\.\)]\s+.+', text, re.MULTILINE)
        actual = len(bullets)
        passed = actual >= required_count

        return ConstraintResult(
            constraint_type="bullet_points",
            passed=passed,
            expected=f"{required_count} bullet points",
            actual=f"{actual} bullet points",
            error_message=None if passed else f"Found {actual}, needed {required_count}"
        )

    @staticmethod
    def check_all_caps(text: str) -> ConstraintResult:
        """Check if text is in all caps"""
        # Ignore non-alphabetic characters
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return ConstraintResult("all_caps", False, "All caps", "No letters", "No letters found")

        caps = sum(1 for c in letters if c.isupper())
        passed = caps == len(letters)

        return ConstraintResult(
            constraint_type="all_caps",
            passed=passed,
            expected="All uppercase letters",
            actual=f"{caps}/{len(letters)} uppercase",
            error_message=None if passed else f"Only {caps}/{len(letters)} letters are uppercase"
        )

    @staticmethod
    def check_sections(text: str, required_sections: List[str]) -> ConstraintResult:
        """Check if text contains required section headers"""
        text_lower = text.lower()
        missing = [section for section in required_sections if section.lower() not in text_lower]
        passed = len(missing) == 0

        return ConstraintResult(
            constraint_type="sections",
            passed=passed,
            expected=f"Sections: {', '.join(required_sections)}",
            actual=f"Missing: {', '.join(missing)}" if missing else "All present",
            error_message=None if passed else f"Missing sections: {', '.join(missing)}"
        )

    @staticmethod
    def check_no_punctuation(text: str) -> ConstraintResult:
        """Check if text avoids punctuation"""
        punctuation = [c for c in text if c in '.,!?;:\'"()[]{}']
        passed = len(punctuation) == 0

        return ConstraintResult(
            constraint_type="no_punctuation",
            passed=passed,
            expected="No punctuation",
            actual=f"Found {len(punctuation)} punctuation marks" if punctuation else "No punctuation",
            error_message=None if passed else f"Contains {len(punctuation)} punctuation marks"
        )


class IFEvalBenchmark:
    """Run IFEval-style instruction following tests"""

    def __init__(self, model_config: Dict[str, Any], results_dir: str = "results/ifeval"):
        self.model_config = model_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenRouterClient(model_config)
        self.checker = ConstraintChecker()

    def get_test_prompts(self) -> List[Dict[str, Any]]:
        """Get test prompts with constraints"""
        return [
            {
                "id": 1,
                "prompt": "Write about artificial intelligence.",
                "constraints": ["word_count_min:100", "forbidden:delve,leverage,utilize"],
                "description": "100+ words, avoid AI buzzwords"
            },
            {
                "id": 2,
                "prompt": "List benefits of exercise.",
                "constraints": ["bullet_points:5"],
                "description": "Exactly 5 bullet points"
            },
            {
                "id": 3,
                "prompt": "Write a warning message.",
                "constraints": ["all_caps", "word_count_min:10", "word_count_max:30"],
                "description": "All caps, 10-30 words"
            },
            {
                "id": 4,
                "prompt": "Write an essay with Introduction, Body, and Conclusion sections.",
                "constraints": ["sections:Introduction,Body,Conclusion", "word_count_min:150"],
                "description": "Must have 3 sections, 150+ words"
            },
            {
                "id": 5,
                "prompt": "Write a telegram message (no punctuation allowed).",
                "constraints": ["no_punctuation", "word_count_min:20"],
                "description": "No punctuation, 20+ words"
            },
            {
                "id": 6,
                "prompt": "Explain photosynthesis.",
                "constraints": ["word_count_min:80", "word_count_max:120", "forbidden:plant,leaf"],
                "description": "80-120 words, can't say 'plant' or 'leaf'"
            },
            {
                "id": 7,
                "prompt": "List programming languages.",
                "constraints": ["bullet_points:10", "forbidden:Python,Java"],
                "description": "10 languages, not Python or Java"
            },
            {
                "id": 8,
                "prompt": "Write instructions for making tea.",
                "constraints": ["bullet_points:5", "word_count_min:50"],
                "description": "5 steps, 50+ words total"
            },
            {
                "id": 9,
                "prompt": "Describe the ocean.",
                "constraints": ["word_count_min:60", "forbidden:blue,water,wet"],
                "description": "60+ words, avoid obvious descriptors"
            },
            {
                "id": 10,
                "prompt": "Write a product announcement.",
                "constraints": ["all_caps", "sections:PRODUCT,FEATURES,PRICING", "word_count_min:80"],
                "description": "All caps, 3 sections, 80+ words"
            },
        ]

    def parse_constraints(self, constraint_strs: List[str]) -> List[tuple]:
        """Parse constraint strings into (type, params) tuples"""
        parsed = []
        for c in constraint_strs:
            if ":" in c:
                ctype, params = c.split(":", 1)
                parsed.append((ctype, params))
            else:
                parsed.append((c, None))
        return parsed

    def check_constraints(self, text: str, constraints: List[str]) -> List[ConstraintResult]:
        """Check all constraints on output text"""
        results = []
        parsed = self.parse_constraints(constraints)

        for ctype, params in parsed:
            if ctype == "word_count_min":
                results.append(self.checker.check_word_count(text, int(params)))
            elif ctype == "word_count_max":
                # Handle max with min (find min constraint)
                min_words = 0
                for ct, cp in parsed:
                    if ct == "word_count_min":
                        min_words = int(cp)
                results.append(self.checker.check_word_count(text, min_words, int(params)))
            elif ctype == "forbidden":
                words = params.split(",")
                results.append(self.checker.check_forbidden_words(text, words))
            elif ctype == "bullet_points":
                results.append(self.checker.check_bullet_points(text, int(params)))
            elif ctype == "all_caps":
                results.append(self.checker.check_all_caps(text))
            elif ctype == "sections":
                sections = params.split(",")
                results.append(self.checker.check_sections(text, sections))
            elif ctype == "no_punctuation":
                results.append(self.checker.check_no_punctuation(text))

        return results

    def build_system_prompt(self, constraints: List[str]) -> str:
        """Build system prompt with explicit constraints"""
        parsed = self.parse_constraints(constraints)

        rules = []
        for ctype, params in parsed:
            if ctype == "word_count_min":
                rules.append(f"- Use at least {params} words")
            elif ctype == "word_count_max":
                rules.append(f"- Use no more than {params} words")
            elif ctype == "forbidden":
                rules.append(f"- Do NOT use these words: {params}")
            elif ctype == "bullet_points":
                rules.append(f"- Format as a list with exactly {params} bullet points")
            elif ctype == "all_caps":
                rules.append("- Write in ALL CAPS")
            elif ctype == "sections":
                rules.append(f"- Include these sections: {params}")
            elif ctype == "no_punctuation":
                rules.append("- Do NOT use any punctuation marks")

        prompt = "You are a helpful assistant that follows instructions precisely.\n\nRULES:\n" + "\n".join(rules)
        return prompt

    def run_single_prompt(self, prompt_data: Dict[str, Any]) -> IFEvalResult:
        """Test a single prompt"""
        prompt = prompt_data["prompt"]
        constraints = prompt_data["constraints"]

        system_prompt = self.build_system_prompt(constraints)

        try:
            response = self.client.chat(
                user_message=prompt,
                system_prompt=system_prompt
            )
            output = response['response']
        except Exception as e:
            output = f"[API ERROR: {e}]"

        # Check constraints
        constraint_results = self.check_constraints(output, constraints)
        passed_count = sum(1 for r in constraint_results if r.passed)
        total = len(constraint_results)

        return IFEvalResult(
            prompt_id=prompt_data["id"],
            prompt=prompt,
            constraints=constraints,
            model_output=output,
            passed=passed_count == total,
            constraint_results=constraint_results,
            pass_rate=passed_count / total * 100 if total > 0 else 0
        )

    def run(self):
        """Run full benchmark"""
        prompts = self.get_test_prompts()

        print("=" * 70)
        print("IFEval Benchmark - Instruction Following Evaluation")
        print("=" * 70)
        print(f"Model: {self.model_config['model_name']}")
        print(f"Prompts: {len(prompts)}")
        print()

        results = []
        for i, prompt_data in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Testing: {prompt_data['description']}")
            result = self.run_single_prompt(prompt_data)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"  Result: {status} ({result.pass_rate:.0f}% constraints met)")

            if not result.passed:
                for cr in result.constraint_results:
                    if not cr.passed:
                        print(f"    X {cr.constraint_type}: {cr.error_message}")

        # Summary
        total_prompts = len(results)
        passed_prompts = sum(1 for r in results if r.passed)
        avg_constraint_pass_rate = sum(r.pass_rate for r in results) / total_prompts

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Prompts passed: {passed_prompts}/{total_prompts} ({passed_prompts/total_prompts*100:.1f}%)")
        print(f"Average constraint pass rate: {avg_constraint_pass_rate:.1f}%")
        print("=" * 70)

        # Save results
        self._save_results(results, passed_prompts, total_prompts, avg_constraint_pass_rate)

    def _save_results(self, results, passed, total, avg_rate):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_config['model_name'].replace('/', '_')

        summary = {
            "model": self.model_config['model_name'],
            "timestamp": datetime.now().isoformat(),
            "prompts_passed": passed,
            "total_prompts": total,
            "prompt_pass_rate": passed / total * 100,
            "avg_constraint_pass_rate": avg_rate
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
        "max_tokens": 1000,
        "temperature": 0.0
    }

    benchmark = IFEvalBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()
