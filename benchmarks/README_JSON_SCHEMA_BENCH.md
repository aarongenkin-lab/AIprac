# JSONSchemaBench Implementation

## What is JSONSchemaBench?

JSONSchemaBench is a benchmark that tests **constrained decoding** - the ability of language models to generate valid JSON that strictly adheres to complex schemas. This is critical for agentic systems because:

- **Agents fail when they output bad JSON** that APIs reject
- **Schema violations break function calling** in production systems
- **Hallucinated fields** cause unexpected API behavior

## Why This Matters for Your Research

Your experiment plan focuses on optimization and strict adherence. This benchmark tests exactly that:

1. **Syntax Correctness**: Can the model generate valid JSON?
2. **Type Constraints**: Does the model respect `integer` vs `string`, `required` fields, etc.?
3. **No Hallucinations**: Does the model add extra fields not in the schema?

## Dataset

We use the **GlaiveAI function calling dataset** which contains ~2,000 real-world API schemas. These are specifically tuned for testing function-calling scenarios that agents encounter.

## Installation

```bash
# Install dependencies (already added to requirements.txt)
pip install datasets jsonschema

# Or install all requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Test Single Model (Fast)

```bash
python benchmarks/json_schema_bench.py
```

This runs GPT-3.5-turbo on 50 samples (~2-3 minutes, costs ~$0.01).

### 2. Test Multiple Models (Comprehensive)

```bash
python run_json_schema_benchmark.py
```

This evaluates all your models from the experiment plan.

## Understanding Results

### Pass Rate
- **>95%**: Excellent - Model can reliably generate constrained JSON
- **80-95%**: Good - Minor issues, likely fixable with better prompting
- **<80%**: Poor - Model struggles with constraints, not suitable for production agents

### Failure Types

1. **Syntax Errors** (The "Stupid" Error)
   - Invalid JSON (missing braces, wrong quotes)
   - **Fix**: Lower temperature, add examples

2. **Schema Violations** (The "Constraint" Error)
   - Valid JSON but wrong types (e.g., string instead of integer)
   - **Fix**: Model needs better instruction following

3. **Hallucinations** (The "Agent" Error)
   - Added fields not in schema
   - **Fix**: Stronger negative constraints in prompt

4. **API Errors**
   - Network issues, rate limits, etc.

## Example Output

```
BENCHMARK RESULTS
======================================================================
Model: openai/gpt-3.5-turbo
Total Samples: 50

Passed: 43 (86.00%)
Failed: 7

Failure Breakdown:
  - Syntax Errors (invalid JSON): 2
  - Schema Violations (wrong types/values): 3
  - Hallucinations (extra fields): 2
  - API Errors: 0

Average Latency: 1.23s
======================================================================
```

## Files Generated

Results are saved to `results/json_schema_bench/`:

- `summary_[model]_[timestamp].json` - High-level statistics
- `results_[model]_[timestamp].json` - Full details for every sample
- `failures_[model]_[timestamp].json` - Only failed samples for debugging

## Analyzing Failures

To understand why your model fails:

```python
import json

# Load failures file
with open('results/json_schema_bench/failures_[model]_[timestamp].json') as f:
    failures = json.load(f)

# Look at specific failure types
syntax_errors = [f for f in failures if f['failure_type'] == 'syntax']
schema_violations = [f for f in failures if f['failure_type'] == 'schema_violation']

# Examine a specific failure
print(failures[0]['prompt'])           # What was asked
print(failures[0]['schema'])           # What schema was required
print(failures[0]['model_output'])     # What model produced
print(failures[0]['error_message'])    # Why it failed
```

## Customization

### Change Sample Size

Edit `run_json_schema_benchmark.py`:
```python
SAMPLE_SIZE = 100  # Test more samples
```

### Test Different Dataset

Edit `benchmarks/json_schema_bench.py`:
```python
benchmark = JSONSchemaBenchmark(
    model_config=config,
    dataset_name="epfl-dlab/JSONSchemaBench",  # Official full benchmark
    sample_size=500
)
```

### Adjust Prompt Engineering

The system prompt is in `json_schema_bench.py` line ~182:
```python
system_prompt = (
    "You are a helpful assistant that generates valid JSON. "
    "You MUST output ONLY a valid JSON object..."
)
```

Modify this to test different prompting strategies.

## Integration with Your Research

### Next Steps

1. **Baseline All Models** (do this first)
   - Run all 5 models from your experiment plan
   - Identify which models are "JSON-safe" for agents

2. **Prompt Optimization**
   - Test different system prompts
   - Add few-shot examples
   - Compare pass rates

3. **Temperature Study**
   - Test temperature 0.0 vs 0.3 vs 0.7
   - Does randomness hurt constraint adherence?

4. **Real-World Test**
   - Use failed schemas to improve your agent prompts
   - Build a "JSON validator" layer for your agents

## Expected Model Rankings (Hypothesis)

Based on architecture and training:

1. **Claude Sonnet 4** - Should excel (trained on code, instruction-following)
2. **GPT-4/GPT-OSS-120B** - Strong baseline
3. **Gemini 2.5 Flash** - Fast but may have more hallucinations
4. **DeepSeek V3.2** - Unknown, interesting to test
5. **Grok 4 Fast** - Speed-optimized, may sacrifice accuracy

**Run the benchmark to validate these hypotheses!**

## Troubleshooting

### "Dataset not found"
The script will automatically fall back to the GlaiveAI dataset if the official benchmark isn't available.

### "API key not found"
Make sure your `.env` file has:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

### Rate limits
Reduce `sample_size` or add delays between samples in the code.

### Out of memory
The benchmark runs one sample at a time, so memory shouldn't be an issue. If you see OOM, reduce `sample_size`.

## Citation

If you use this benchmark in research:

```bibtex
@article{jsonschema2025,
  title={JSONSchemaBench: Evaluating Constrained Decoding in Large Language Models},
  author={EPFL DLAB},
  year={2025}
}
```

## Questions?

See the main code in `benchmarks/json_schema_bench.py` for implementation details.
