"""
JSONSchemaBench: Benchmark for testing constrained JSON generation

This benchmark evaluates how well models can generate valid JSON that adheres
to strict schema constraints - critical for agentic systems that call APIs.

Based on: epfl-dlab/JSONSchemaBench (Jan 2025)
Focus: GlaiveAI-2K subset (function calling scenarios)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from datasets import load_dataset
from jsonschema import validate, ValidationError, SchemaError
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_clients import OpenRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark sample"""
    sample_id: int
    passed: bool
    failure_type: Optional[str]  # 'syntax', 'schema_violation', 'hallucination', 'api_error'
    prompt: str
    schema: Dict[str, Any]
    model_output: str
    error_message: Optional[str]
    latency_seconds: float


@dataclass
class BenchmarkSummary:
    """Overall benchmark results"""
    model_name: str
    total_samples: int
    passed: int
    failed: int
    syntax_errors: int
    schema_violations: int
    hallucinations: int
    api_errors: int
    pass_rate: float
    avg_latency: float
    timestamp: str


class JSONSchemaBenchmark:
    """Runner for JSONSchemaBench evaluation"""

    def __init__(
        self,
        model_config: Dict[str, Any],
        sample_size: int = 50,
        dataset_name: str = "glaiveai/glaive-function-calling-v2",
        save_failures: bool = True,
        results_dir: str = "results/json_schema_bench"
    ):
        """
        Initialize the benchmark

        Args:
            model_config: Configuration dict for OpenRouterClient
            sample_size: Number of samples to test (default 50 for quick eval)
            dataset_name: HuggingFace dataset to use
            save_failures: Whether to save failed examples for analysis
            results_dir: Directory to save results
        """
        self.model_config = model_config
        self.sample_size = sample_size
        self.dataset_name = dataset_name
        self.save_failures = save_failures
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model client
        self.client = OpenRouterClient(model_config)
        logger.info(f"Initialized client for model: {model_config['model_name']}")

        # Results storage
        self.results: List[BenchmarkResult] = []

    def load_dataset_samples(self) -> List[Dict[str, Any]]:
        """Load and prepare dataset samples"""
        logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            # Try to load the official JSONSchemaBench first
            dataset = load_dataset("epfl-dlab/JSONSchemaBench", split="test")
            logger.info("Loaded official JSONSchemaBench dataset")
        except Exception as e:
            logger.warning(f"Could not load official benchmark, using fallback: {e}")
            # Fallback to GlaiveAI function calling dataset
            dataset = load_dataset(self.dataset_name, split="train")
            logger.info(f"Loaded fallback dataset: {self.dataset_name}")

        # Convert to list and take sample
        samples = list(dataset)[:self.sample_size]
        logger.info(f"Loaded {len(samples)} samples for evaluation")

        return samples

    def extract_json_schema(self, sample: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract prompt and JSON schema from dataset sample

        Different datasets have different formats, this handles both
        """
        # Try official JSONSchemaBench format (schema only, no prompt)
        if 'json_schema' in sample:
            schema_str = sample['json_schema']
            schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str

            # Generate a prompt from the schema description
            prompt = self._generate_prompt_from_schema(schema, sample.get('unique_id', 'unknown'))
            return prompt, schema

        # Try format with explicit prompt
        if 'prompt' in sample and 'schema' in sample:
            return sample['prompt'], sample['schema']

        # Try GlaiveAI function calling format
        if 'system' in sample and 'tools' in sample:
            # For function calling, we construct a schema from the function definition
            tools = json.loads(sample['tools']) if isinstance(sample['tools'], str) else sample['tools']

            # Get the first function as our target
            if tools and len(tools) > 0:
                function_def = tools[0]['function']
                prompt = sample.get('chat', 'Generate a valid function call for this schema')
                schema = function_def.get('parameters', {})

                return prompt, schema

        # Fallback: try to parse any 'functions' or 'tools' field
        for field in ['functions', 'tools', 'schema']:
            if field in sample:
                data = sample[field]
                if isinstance(data, str):
                    data = json.loads(data)
                return sample.get('prompt', sample.get('instruction', 'Generate valid JSON')), data

        raise ValueError(f"Could not extract schema from sample: {sample.keys()}")

    def _generate_prompt_from_schema(self, schema: Dict[str, Any], unique_id: str) -> str:
        """Generate a descriptive prompt from a JSON schema"""
        # Extract information from schema
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        # Build a descriptive prompt
        prompt_parts = [f"Generate a valid JSON object (task: {unique_id})"]

        if properties:
            prop_descriptions = []
            for prop_name, prop_info in properties.items():
                prop_type = prop_info.get('type', 'any')
                prop_desc = prop_info.get('description', '')
                is_required = prop_name in required

                desc = f"'{prop_name}' ({prop_type})"
                if prop_desc:
                    desc += f": {prop_desc}"
                if is_required:
                    desc += " [REQUIRED]"
                prop_descriptions.append(desc)

            if prop_descriptions:
                prompt_parts.append("Fields: " + "; ".join(prop_descriptions[:3]))  # First 3 fields

        return " ".join(prompt_parts)

    def classify_failure(
        self,
        model_output: str,
        schema: Dict[str, Any],
        error: Exception
    ) -> Tuple[str, str]:
        """
        Classify the type of failure

        Returns:
            (failure_type, detailed_message)
        """
        # Syntax error - invalid JSON
        if isinstance(error, json.JSONDecodeError):
            return 'syntax', f"Invalid JSON syntax: {str(error)}"

        # Schema violation - valid JSON but doesn't match schema
        if isinstance(error, ValidationError):
            # Check if it's a hallucination (extra fields)
            try:
                json_obj = json.loads(model_output)
                required_fields = set(schema.get('required', []))
                output_fields = set(json_obj.keys())

                if output_fields > required_fields:
                    extra_fields = output_fields - required_fields
                    return 'hallucination', f"Extra fields not in schema: {extra_fields}"

            except:
                pass

            return 'schema_violation', f"Schema validation failed: {error.message}"

        # API or other errors
        return 'api_error', f"Unexpected error: {str(error)}"

    def run_single_sample(self, sample_id: int, sample: Dict[str, Any]) -> BenchmarkResult:
        """Evaluate a single sample"""

        try:
            # Extract prompt and schema
            prompt, schema = self.extract_json_schema(sample)
        except Exception as e:
            logger.error(f"Failed to parse sample {sample_id}: {e}")
            return BenchmarkResult(
                sample_id=sample_id,
                passed=False,
                failure_type='api_error',
                prompt=str(sample),
                schema={},
                model_output="",
                error_message=f"Dataset parsing error: {str(e)}",
                latency_seconds=0.0
            )

        # Construct system prompt - enforce strict JSON output with few-shot examples
        system_prompt = (
            "You are a helpful assistant that generates valid JSON. "
            "You MUST output ONLY a valid JSON object that strictly adheres to the provided schema.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the JSON object - no explanations, no markdown, no code blocks\n"
            "2. Only include fields specified in the schema - no extra fields\n"
            "3. Match exact types: \"type\": \"number\" means 25, NOT \"25\"\n\n"
            "EXAMPLES:\n"
            "Schema: {\"properties\": {\"age\": {\"type\": \"number\"}}, \"required\": [\"age\"]}\n"
            "✓ Correct: {\"age\": 25}\n"
            "✗ Wrong: {\"age\": \"25\"}  // string instead of number\n"
            "✗ Wrong: {\"age\": 25, \"comment\": \"user age\"}  // extra field not in schema\n\n"
            "Schema: {\"properties\": {\"active\": {\"type\": \"boolean\"}}, \"required\": [\"active\"]}\n"
            "✓ Correct: {\"active\": true}\n"
            "✗ Wrong: {\"active\": \"true\"}  // string instead of boolean\n\n"
            "Schema: {\"properties\": {\"name\": {\"type\": \"string\"}, \"count\": {\"type\": \"number\"}}, \"required\": [\"name\"]}\n"
            "✓ Correct: {\"name\": \"example\"}\n"
            "✓ Correct: {\"name\": \"example\", \"count\": 10}  // optional field included\n"
            "✗ Wrong: {\"name\": \"example\", \"id\": 5}  // 'id' not in schema\n"
        )

        # Construct user message with schema
        user_message = f"{prompt}\n\nRequired JSON Schema:\n{json.dumps(schema, indent=2)}"

        # Call model
        import time
        start_time = time.time()

        try:
            response = self.client.chat(
                user_message=user_message,
                system_prompt=system_prompt
            )

            model_output = response['response'].strip()
            latency = time.time() - start_time

            # Clean up output (remove markdown code blocks if present)
            if model_output.startswith('```'):
                # Remove markdown code fence
                lines = model_output.split('\n')
                model_output = '\n'.join(lines[1:-1]) if len(lines) > 2 else model_output
                model_output = model_output.replace('```json', '').replace('```', '').strip()

            # Validate JSON syntax
            json_obj = json.loads(model_output)

            # Validate against schema
            validate(instance=json_obj, schema=schema)

            # Success!
            return BenchmarkResult(
                sample_id=sample_id,
                passed=True,
                failure_type=None,
                prompt=prompt,
                schema=schema,
                model_output=model_output,
                error_message=None,
                latency_seconds=latency
            )

        except Exception as error:
            # Classify and record failure
            failure_type, error_msg = self.classify_failure(
                model_output if 'model_output' in locals() else "",
                schema,
                error
            )

            return BenchmarkResult(
                sample_id=sample_id,
                passed=False,
                failure_type=failure_type,
                prompt=prompt,
                schema=schema,
                model_output=model_output if 'model_output' in locals() else "",
                error_message=error_msg,
                latency_seconds=time.time() - start_time
            )

    def run(self) -> BenchmarkSummary:
        """Run the full benchmark"""
        logger.info("="*70)
        logger.info("Starting JSONSchemaBench Evaluation")
        logger.info(f"Model: {self.model_config['model_name']}")
        logger.info(f"Samples: {self.sample_size}")
        logger.info("="*70)

        # Load dataset
        samples = self.load_dataset_samples()

        # Run evaluation
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            result = self.run_single_sample(i, sample)
            self.results.append(result)

        # Calculate summary statistics
        summary = self._generate_summary()

        # Save results
        self._save_results(summary)

        # Print summary
        self._print_summary(summary)

        return summary

    def _generate_summary(self) -> BenchmarkSummary:
        """Generate summary statistics"""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        syntax_errors = sum(1 for r in self.results if r.failure_type == 'syntax')
        schema_violations = sum(1 for r in self.results if r.failure_type == 'schema_violation')
        hallucinations = sum(1 for r in self.results if r.failure_type == 'hallucination')
        api_errors = sum(1 for r in self.results if r.failure_type == 'api_error')

        avg_latency = sum(r.latency_seconds for r in self.results) / len(self.results)
        pass_rate = (passed / len(self.results)) * 100 if self.results else 0

        return BenchmarkSummary(
            model_name=self.model_config['model_name'],
            total_samples=len(self.results),
            passed=passed,
            failed=failed,
            syntax_errors=syntax_errors,
            schema_violations=schema_violations,
            hallucinations=hallucinations,
            api_errors=api_errors,
            pass_rate=pass_rate,
            avg_latency=avg_latency,
            timestamp=datetime.now().isoformat()
        )

    def _save_results(self, summary: BenchmarkSummary):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.model_config['model_name'].replace('/', '_')

        # Save summary
        summary_file = self.results_dir / f"summary_{model_safe_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        logger.info(f"Saved summary to: {summary_file}")

        # Save all results
        results_file = self.results_dir / f"results_{model_safe_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        logger.info(f"Saved detailed results to: {results_file}")

        # Save failures only (for analysis)
        if self.save_failures:
            failures = [r for r in self.results if not r.passed]
            failures_file = self.results_dir / f"failures_{model_safe_name}_{timestamp}.json"
            with open(failures_file, 'w') as f:
                json.dump([asdict(r) for r in failures], f, indent=2)
            logger.info(f"Saved {len(failures)} failures to: {failures_file}")

    def _print_summary(self, summary: BenchmarkSummary):
        """Print summary to console"""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"Model: {summary.model_name}")
        print(f"Total Samples: {summary.total_samples}")
        print(f"\nPassed: {summary.passed} ({summary.pass_rate:.2f}%)")
        print(f"Failed: {summary.failed}")
        print(f"\nFailure Breakdown:")
        print(f"  - Syntax Errors (invalid JSON): {summary.syntax_errors}")
        print(f"  - Schema Violations (wrong types/values): {summary.schema_violations}")
        print(f"  - Hallucinations (extra fields): {summary.hallucinations}")
        print(f"  - API Errors: {summary.api_errors}")
        print(f"\nAverage Latency: {summary.avg_latency:.2f}s")
        print("="*70)


def main():
    """Example usage"""
    from dotenv import load_dotenv
    load_dotenv()

    # Test with GPT-3.5-turbo (cheapest model for experiments)
    config = {
        "provider": "openrouter",
        "model_name": "openai/gpt-3.5-turbo",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 500,
        "temperature": 0.0  # Zero temperature for deterministic output
    }

    benchmark = JSONSchemaBenchmark(
        model_config=config,
        sample_size=50  # Start small
    )

    summary = benchmark.run()


if __name__ == "__main__":
    main()
