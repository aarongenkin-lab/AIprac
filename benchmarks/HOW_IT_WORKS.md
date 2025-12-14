# How the JSONSchemaBench Works

## Overview

The benchmark tests whether an LLM can generate valid JSON that strictly adheres to a given JSON Schema. This is critical for agentic systems that need to call APIs with specific input formats.

## The Complete Flow

### 1. System Prompt (Sets the Rules)

```
You are a helpful assistant that generates valid JSON.
You MUST output ONLY a valid JSON object that strictly adheres to the provided schema.
Do not include any text, explanation, or markdown formatting - only the raw JSON object.
Do not add any fields not specified in the schema.
```

**Purpose:** Establishes strict constraints - no explanations, no extra fields, pure JSON only.

### 2. User Message (Gives the Task + Schema)

The benchmark constructs a message like:

```
Generate a valid JSON object (task: calculate_area_88ee549f)
Fields: 'dimensions' (object) [REQUIRED]; 'shape' (string) [REQUIRED]

Required JSON Schema:
{
  "properties": {
    "dimensions": {
      "properties": {
        "height": {"description": "The height", "type": "number"},
        "radius": {"description": "The radius", "type": "number"},
        "width": {"description": "The width", "type": "number"}
      },
      "required": ["width", "height", "radius"],
      "type": "object"
    },
    "shape": {
      "description": "The shape for which to calculate the area",
      "type": "string"
    }
  },
  "required": ["shape", "dimensions"],
  "type": "object"
}
```

**Key Components:**
- **Task description**: Auto-generated from schema metadata
- **Field hints**: Lists the main required fields upfront
- **Full schema**: Complete JSON Schema specification

### 3. Model Response Examples

#### CORRECT OUTPUT (PASS)

```json
{
  "shape": "rectangle",
  "dimensions": {
    "width": 10.5,
    "height": 20.0,
    "radius": 5.0
  }
}
```

**Why it passes:**
- Valid JSON syntax
- Has required fields: `shape`, `dimensions`
- Correct types: `shape` is string, dimensions have numbers
- No extra fields

#### FAILURE TYPE 1: Syntax Error

```json
{
  "shape": "rectangle",
  "dimensions": {
    "width": 10.5
  }
```

**Problem:** Missing closing brace `}`
**Classification:** `syntax_error`
**Impact:** 4-8% of failures (varies by model)

#### FAILURE TYPE 2: Schema Violation

```json
{
  "shape": "rectangle",
  "dimensions": {
    "width": "10.5",
    "height": 20.0,
    "radius": 5.0
  }
}
```

**Problem:** `width` is string `"10.5"` but schema requires `number`
**Classification:** `schema_violation`
**Impact:** 0-16% of failures (GPT-3.5-turbo worst)

#### FAILURE TYPE 3: Hallucination

```json
{
  "shape": "rectangle",
  "dimensions": {
    "width": 10.5,
    "height": 20.0,
    "radius": 5.0
  },
  "color": "blue",
  "comment": "This is a nice shape"
}
```

**Problem:** Added `color` and `comment` fields not in schema
**Classification:** `hallucination`
**Impact:** 2-14% of failures (GPT-3.5-turbo worst at 14%)

### 4. Validation Process

The benchmark validates in two stages:

```python
# Stage 1: JSON Syntax Validation
json_obj = json.loads(model_output)  # Fails if invalid JSON

# Stage 2: Schema Validation
jsonschema.validate(instance=json_obj, schema=schema)  # Fails if doesn't match schema
```

If either fails, the sample is marked as failed and classified.

## Code Walkthrough

### Location: `benchmarks/json_schema_bench.py`

**Key Function:** `run_single_sample()` (line 222)

```python
def run_single_sample(self, sample_id: int, sample: Dict[str, Any]) -> BenchmarkResult:
    # 1. Extract schema from dataset
    prompt, schema = self.extract_json_schema(sample)

    # 2. Build system prompt
    system_prompt = (
        "You are a helpful assistant that generates valid JSON. "
        "You MUST output ONLY a valid JSON object that strictly adheres to the provided schema. "
        "Do not include any text, explanation, or markdown formatting - only the raw JSON object. "
        "Do not add any fields not specified in the schema."
    )

    # 3. Build user message with schema
    user_message = f"{prompt}\n\nRequired JSON Schema:\n{json.dumps(schema, indent=2)}"

    # 4. Call the model
    response = self.client.chat(
        user_message=user_message,
        system_prompt=system_prompt
    )

    model_output = response['response'].strip()

    # 5. Clean markdown (if model wrapped in ```json blocks)
    if model_output.startswith('```'):
        model_output = model_output.replace('```json', '').replace('```', '').strip()

    # 6. Validate JSON syntax
    json_obj = json.loads(model_output)  # Throws JSONDecodeError if invalid

    # 7. Validate against schema
    jsonschema.validate(instance=json_obj, schema=schema)  # Throws ValidationError if wrong

    # 8. If we get here, it passed!
    return BenchmarkResult(passed=True, ...)
```

## Dataset Structure

**Source:** `epfl-dlab/JSONSchemaBench` on HuggingFace

Each sample contains:
```json
{
  "json_schema": "{\"properties\": {...}, \"required\": [...]}",
  "unique_id": "calculate_area_88ee549f"
}
```

**Notable:** The dataset only provides schemas, not prompts. Our code auto-generates descriptive prompts from the schema properties.

## Metrics Calculated

For each model, we track:

1. **Pass Rate**: `(passed / total) * 100`
2. **Failure Breakdown**:
   - Syntax errors (invalid JSON)
   - Schema violations (wrong types/values)
   - Hallucinations (extra fields)
   - API errors (network/credit issues)
3. **Latency**: Average time per sample
4. **Per-sample results**: Saved for debugging

## Why This Benchmark Matters

### For Agentic Systems

Agents need to call APIs like:
```python
# Agent wants to call Stripe API
stripe.charge.create({
  "amount": 5000,        # Must be integer
  "currency": "usd",     # Must be string
  "source": "tok_visa"   # Must be string
})
```

If the agent generates:
```json
{
  "amount": "5000",  // String instead of int - API REJECTS
  "currency": "usd",
  "source": "tok_visa",
  "description": "Payment"  // Extra field - might be ignored or rejected
}
```

The API call fails, the agent breaks, and the user has a bad experience.

### Real-World Impact

**88% pass rate (Gemini 2.5 Flash)** means:
- **In production:** 12 out of every 100 API calls fail
- **Solution:** Add validation layer + retry logic
- **Cost:** Wasted API calls, slower response times

**62% pass rate (GPT-3.5-turbo)** means:
- **In production:** 38 out of every 100 API calls fail
- **Not acceptable** for production without heavy validation

## Customization

### Testing Different Prompts

Modify line 242-247 to test different system prompts:

```python
# Original
system_prompt = "You are a helpful assistant that generates valid JSON..."

# Experiment: More aggressive
system_prompt = "CRITICAL: You MUST generate ONLY valid JSON. Any deviation will cause system failure..."

# Experiment: Few-shot
system_prompt = """Here are examples of correct outputs:
Example 1: {...}
Example 2: {...}
Now generate valid JSON for the following schema:"""
```

### Testing Different Temperatures

Modify `run_json_schema_benchmark.py` line 50:

```python
# Deterministic (current)
"temperature": 0.0

# Experiment with randomness
"temperature": 0.3  # or 0.7
```

### Larger Sample Size

Modify `run_json_schema_benchmark.py` line 145:

```python
# Quick test (current)
SAMPLE_SIZE = 50

# Full evaluation
SAMPLE_SIZE = 2867  # Entire test set
```

## Performance Comparison

| Model | Pass Rate | Speed | Cost/50 | Best For |
|-------|-----------|-------|---------|----------|
| Gemini 2.5 Flash | 88% | 1.04s | $0.02 | Production |
| GPT-3.5-turbo | 62% | 1.40s | $0.01 | Budget/testing |
| GPT-4o | 60%* | 3.13s | $0.15 | N/A (credit issues) |

*GPT-4o results affected by API credit errors

## Next Steps

1. **Add OpenRouter credits** to test remaining models
2. **Implement validation layer** in your agent
3. **Run larger sample sizes** for statistical confidence
4. **Test prompt variations** to improve pass rates
5. **Integrate into CI/CD** for regression testing

## Questions?

See the full implementation in:
- `benchmarks/json_schema_bench.py` - Core benchmark code
- `run_json_schema_benchmark.py` - Multi-model runner
- `BENCHMARK_RESULTS.md` - Full analysis of results
