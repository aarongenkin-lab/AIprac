# Recommended Models for Agentic Systems (2025)

## Overview

This document lists recommended models for use with the enhanced ReAct agent, based on cost-performance trade-offs and instruction-following capabilities.

---

## Tier 1: Recommended for Production/Experiments

### GPT-4o-mini
- **Model ID**: `openai/gpt-4o-mini`
- **Pricing**: $0.15/M input, $0.60/M output
- **Strengths**:
  - Excellent instruction-following
  - Much better than GPT-3.5-turbo
  - Reliable format adherence
  - Good reasoning capabilities
- **Use for**: Main experiments, demos, production
- **Estimated cost**: ~$0.001 per agent call (with tools)

### Gemini 2.0 Flash Exp
- **Model ID**: `google/gemini-2.0-flash-exp:free`
- **Pricing**: FREE tier available
- **Strengths**:
  - Very fast responses
  - Good instruction-following
  - Free tier for development/testing
  - Decent reasoning
- **Use for**: High-volume testing, development, ablation studies
- **Note**: Already tested in your benchmarks!

### DeepSeek v3
- **Model ID**: `deepseek/deepseek-chat`
- **Pricing**: Very cheap (~$0.14/M input, $0.28/M output)
- **Strengths**:
  - Strong reasoning capabilities
  - Excellent value for money
  - Good instruction-following
  - Competitive with GPT-4o-mini
- **Use for**: Cost-sensitive production, model diversity in report

---

## Tier 2: Budget Options (For Ablation Studies)

### Gemini 1.5 Flash 8B
- **Model ID**: `google/gemini-flash-1.5-8b`
- **Pricing**: Extremely cheap
- **Use for**: Baseline comparisons, testing at scale

### Qwen 2.5 32B
- **Model ID**: `qwen/qwen-2.5-32b-instruct`
- **Pricing**: Very affordable
- **Strengths**: Good reasoning for the price
- **Use for**: Demonstrating model diversity

---

## Tier 3: Premium (If Budget Allows)

### Claude 3.5 Sonnet
- **Model ID**: `anthropic/claude-3.5-sonnet`
- **Pricing**: More expensive (~$3/M input, $15/M output)
- **Strengths**:
  - Excellent reasoning
  - Best-in-class instruction-following
  - Superior code generation
- **Use for**: Quality comparisons, complex reasoning tasks

### GPT-4o
- **Model ID**: `openai/gpt-4o`
- **Pricing**: ~$2.50/M input, $10/M output
- **Strengths**:
  - Very strong reasoning
  - Excellent instruction-following
  - Multimodal capabilities
- **Use for**: Upper bound comparisons

---

## Recommended Testing Strategy

### For Your Report

**Primary Models** (use these for main results):
1. **GPT-4o-mini** - Main workhorse (good quality, affordable)
2. **Gemini 2.0 Flash** - Free tier, speed tests
3. **DeepSeek v3** - Cost-effective alternative

**Comparison Models** (for diversity):
- **GPT-3.5-turbo** - Baseline (what you've been using)
- **Claude 3 Haiku** - Different architecture (you have results)
- **Gemini 2.5 Flash** - Another provider (you have results)

### Budget Estimation

For 100 agent calls with ~10 tool uses each:
- **GPT-4o-mini**: ~$0.10
- **Gemini 2.0 Flash**: FREE
- **DeepSeek v3**: ~$0.05
- **GPT-3.5-turbo**: ~$0.03

**Total for full benchmark suite**: $1-5 depending on scale

---

## Model Selection Guidelines

### Use GPT-4o-mini when:
- You need reliable results
- Instruction-following is critical
- Budget allows $0.15-0.60 per million tokens
- Running demos or important experiments

### Use Gemini 2.0 Flash when:
- You need high volume testing
- Speed is important
- Budget is tight
- Doing development/debugging

### Use DeepSeek v3 when:
- You want strong reasoning at low cost
- Demonstrating cost-effectiveness
- Need an alternative to OpenAI models

### Use GPT-3.5-turbo when:
- Showing baseline/legacy performance
- Demonstrating improvements over older models
- Absolute minimum cost required

---

## Migration from GPT-3.5-turbo

### Why Upgrade?

GPT-3.5-turbo (released 2022) is now outdated. Issues observed:
- ❌ Inconsistent instruction-following (format violations)
- ❌ Weaker reasoning capabilities
- ❌ More hallucinations
- ❌ Doesn't utilize tools as effectively

### Migration Path

**Step 1**: Update `model_name` in agent initialization
```python
# OLD
agent = EnhancedReActAgent(model_name="openai/gpt-3.5-turbo")

# NEW
agent = EnhancedReActAgent(model_name="openai/gpt-4o-mini")
```

**Step 2**: Rerun benchmarks
- Same tasks, new model
- Compare results
- Document improvements

**Step 3**: Update report
- Show GPT-3.5 as baseline
- Show GPT-4o-mini as current
- Highlight improvements

---

## OpenRouter Shortcuts

### Auto-select cheapest
```python
model_name = "openai/gpt-4o-mini:floor"  # Automatically selects cheapest provider
```

### Fallback routing
```python
# Tries models in order until one works
models = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat"
]
```

---

## Performance Notes

### Instruction-Following (Estimated)
- GPT-4o-mini: ⭐⭐⭐⭐⭐ (95%+ format compliance)
- DeepSeek v3: ⭐⭐⭐⭐ (90%+ format compliance)
- Gemini Flash: ⭐⭐⭐⭐ (85-90% format compliance)
- GPT-3.5-turbo: ⭐⭐⭐ (70-80% format compliance)

### Tool Usage Quality
- GPT-4o-mini: Excellent (uses right tool at right time)
- DeepSeek v3: Very Good (efficient tool selection)
- Gemini Flash: Good (sometimes over-uses tools)
- GPT-3.5-turbo: Fair (inconsistent tool use)

### Cost per 1000 Agent Calls (est.)
- GPT-4o-mini: $1.00
- Gemini Flash: FREE
- DeepSeek v3: $0.50
- GPT-3.5-turbo: $0.30

---

## Current Testing Results

You already have results from:
- ✅ GPT-3.5-turbo (baseline)
- ✅ GPT-4o
- ✅ Claude 3 Haiku
- ✅ Claude Sonnet 4
- ✅ Gemini 2.0/2.5 Flash
- ✅ DeepSeek

**This is great!** You can use these for model comparisons in your report.

---

## Recommendations Summary

**For your project, use:**

1. **Main agent**: GPT-4o-mini
   - Best balance of cost/quality
   - Reliable for demos
   - Good for report results

2. **High-volume testing**: Gemini 2.0 Flash (free)
   - Already tested
   - Fast iteration
   - No cost concerns

3. **Cost comparison**: DeepSeek v3
   - Show cost-effective alternative
   - Demonstrate model diversity
   - Competitive performance

4. **Baseline**: Keep GPT-3.5-turbo results
   - Show improvement over older models
   - Common baseline in literature

**Total estimated cost for full research**: $5-10

---

## Notes

- All models available via OpenRouter API
- Same API interface for all models
- Easy to swap models by changing `model_name`
- No code changes required

**Updated**: December 2025
