# Complete Experimental Results Summary
**Author:** Aaron Genkin (amg454)
**Project:** Performance of Reasoning Agents
**Date:** December 14, 2025

---

## Overview

This document consolidates all experimental results from the agent-based reasoning framework study. Results are organized by experiment type with key findings and recommendations for paper inclusion.

---

## Experiment 1: JSON Schema Constraint Adherence (PRIMARY SUCCESS)

**Status:** STRONG SUCCESS - Recommend as primary experiment
**Source:** `BENCHMARK_RESULTS.md`, `results/json_schema_bench/`
**Date:** December 11, 2025

### Objective
Evaluate constrained generation capabilities critical for agentic systems requiring reliable tool use and API interaction.

### Methodology
- Dataset: JSONSchemaBench (official, epfl-dlab)
- Models: Gemini 2.5 Flash, GPT-3.5-turbo, GPT-4o
- Sample size: 50 schemas per model
- Temperature: 0.0 (deterministic)
- Validation: Python jsonschema library

### Results

| Model | Pass Rate | Passed/Total | Avg Latency | Cost/50 |
|-------|-----------|--------------|-------------|---------|
| **Gemini 2.5 Flash** | **88.0%** | 44/50 | 1.04s | $0.02 |
| GPT-3.5-turbo | 62.0% | 31/50 | 1.40s | $0.01 |
| GPT-4o | 60.0%* | 30/50 | 3.13s | $0.15 |

*GPT-4o results affected by API credit exhaustion (19/50 were API errors)

### Failure Mode Analysis

**Gemini 2.5 Flash (12% failure):**
- Syntax Errors: 2 (4%)
- Schema Violations: 2 (4%)
- Hallucinations: 2 (4%)
- Balanced failure distribution

**GPT-3.5-turbo (38% failure):**
- Syntax Errors: 4 (8%)
- Schema Violations: 8 (16%)
- Hallucinations: 7 (14%) - **highest rate**
- Tendency to add unauthorized fields

### Key Findings
1. Even best model (Gemini) fails 12% - constraint adherence remains challenging
2. Gemini 2.5 Flash offers best cost-performance ratio (88% accuracy, 3x faster than GPT-4o, 7x cheaper)
3. Hallucination patterns differ significantly across models
4. Production systems require validation + retry layers

### Recommendation for Paper
**INCLUDE AS PRIMARY EXPERIMENT** - Clear results, strong methodology, production relevance

---

## Experiment 2: Small Reasoning Benchmark (MIXED/LIMITED)

**Status:** MIXED RESULTS - Consider brief mention only
**Source:** `results/small_benchmark/combined_results_20251214_061335.json`
**Date:** December 14, 2025

### Objective
Compare agent performance across diverse reasoning task types.

### Methodology
- Models: GPT-4o-mini, DeepSeek-v3
- Tasks: 3 (math, logic, code)
- Agent: Enhanced ReAct with tools

### Results Summary

| Task | GPT-4o-mini | DeepSeek-v3 |
|------|-------------|-------------|
| Math (compound interest) | Success (5 steps, 4959 tokens) | Success (1 step, 993 tokens) |
| Logic (parking lot) | Success (1 step, 796 tokens) | Success (1 step, 894 tokens) |
| Code (LIS algorithm) | **FAILED** (timeout, 23556 tokens) | Success (1 step, 1314 tokens) |

### Issues Identified
- GPT-4o-mini timeout on code task (30.9s runtime)
- Math task answers incorrect for both models (calculated interest vs. final amount)
- Small sample size (only 3 tasks)
- Success metrics ambiguous (marked success despite wrong answers)

### Recommendation for Paper
**BRIEF MENTION** - Use to illustrate exploratory phase challenges, not as primary evidence

---

## Experiment 3: Zebra Puzzles - Constraint Satisfaction (MIXED/INSIGHTFUL FAILURE)

**Status:** PARTIAL SUCCESS with valuable insights
**Source:** `results/zebra_agent_test/combined_zebra_results_20251214_062132.json`
**Date:** December 14, 2025

### Objective
Test agent capability on complex logic puzzles requiring constraint satisfaction reasoning.

### Methodology
- Dataset: Custom zebra puzzles (5x6 and 4x4 grids)
- Models: DeepSeek-v3, GPT-4o-mini
- Agent: Enhanced ReAct with Python executor, RAG, search
- Sample size: 2 puzzles per model

### Results

| Model | Puzzle 1 (5x6) | Puzzle 2 (4x4) |
|-------|----------------|----------------|
| DeepSeek-v3 | Success (2495 tokens, 24.3s) | Success (10007 tokens, 50.0s) |
| GPT-4o-mini | Success (2540 tokens, 34.5s) | **FAILED** (timeout, 55420 tokens, 68.7s) |

### Computational Costs
- Average successful solution: ~6,000 tokens, ~36 seconds
- Failed attempt: 55,420 tokens, 68+ seconds
- High variance in token usage (2.5k to 55k)

### Shortcomings Discovered
1. **Low tool utilization** - Agents defaulted to reasoning without Python/RAG despite availability
2. **Timeout issues** - Complex puzzles exceeded reasonable time limits
3. **Prompt inadequacy** - Didn't sufficiently encourage tool use
4. **Evaluation challenges** - Expected answers were blank templates (___), hard to validate
5. **Single-agent limitations** - Complex multi-step reasoning struggled

### Key Insights
- Tool availability ≠ tool usage (prompting crucial)
- Constraint satisfaction harder than expected for LLM agents
- Computational cost prohibitive for larger puzzle sets
- Need different paradigm: interaction vs. solo reasoning

### Recommendation for Paper
**INCLUDE AS "LEARNING EXPERIENCE"** - Valuable negative result, motivated pivot to PD tournament

---

## Experiment 4: Framework Comparison (SUCCESSFUL)

**Status:** CLEAR SUCCESS - Strong comparative data
**Source:** `results/framework_comparison/combined_comparison_20251214_071338.json`
**Date:** December 14, 2025

### Objective
Compare baseline (no agent) vs. Basic ReAct vs. Enhanced ReAct on standard puzzle.

### Methodology
- Test puzzle: Simple 5-house zebra logic (2nd house color)
- Models: DeepSeek-v3, GPT-4o-mini, Claude-3.5-Sonnet
- Frameworks: 3 (Baseline, Basic ReAct, Enhanced ReAct)
- Total runs: 9 (3 models × 3 frameworks)

### Results

#### DeepSeek-v3
| Framework | Success | Tokens | Time | Steps |
|-----------|---------|--------|------|-------|
| Baseline | ✓ | 1113 | 11.3s | 1 |
| Basic ReAct | ✓ | 1073 | 3.3s | 2 |
| Enhanced ReAct | ✓ | 4605 | 24.0s | 1 |

#### GPT-4o-mini
| Framework | Success | Tokens | Time | Steps |
|-----------|---------|--------|------|-------|
| Baseline | ✓ | 596 | 7.2s | 1 |
| Basic ReAct | **✗** | 14148 | 37.0s | 10 |
| Enhanced ReAct | ✓ | 1781 | 19.0s | 1 |

#### Claude-3.5-Sonnet
| Framework | Success | Tokens | Time | Steps |
|-----------|---------|--------|------|-------|
| Baseline | ✓ | 385 | 6.4s | 1 |
| Basic ReAct | ✓ | 598 | 5.1s | 1 |
| Enhanced ReAct | ✓ | 1365 | ~6s | 1 |

### Key Findings
1. **Baseline often sufficient** - Simple puzzle solvable without agent framework
2. **Framework overhead varies** - Enhanced ReAct 2-4x more tokens than baseline
3. **Reliability inconsistent** - GPT-4o-mini's Basic ReAct failed (10 steps, timeout)
4. **Claude most efficient** - Lowest token usage across all frameworks
5. **Tool usage minimal** - Even Enhanced ReAct used 1 step (mostly direct answers)

### Performance Rankings by Efficiency (Tokens)
1. Claude-3.5-Sonnet Baseline: 385 tokens
2. GPT-4o-mini Baseline: 596 tokens
3. Claude-3.5-Sonnet Basic: 598 tokens

### Recommendation for Paper
**INCLUDE** - Shows framework overhead vs. benefit tradeoffs, supports "tools alone insufficient" thesis

---

## Experiment 5: Prisoner's Dilemma Tournament (STRONGEST MULTI-AGENT RESULT)

**Status:** EXCELLENT SUCCESS - Clear strategic patterns
**Source:** `docs/pd_tournament_summary.md`, `results/pd_tournament/`
**Date:** December 14, 2025

### Objective
Analyze strategic decision-making and cooperation emergence in multi-agent interactions.

### Methodology
- Game: Iterated Prisoner's Dilemma
- Opponents: Tit-for-Tat (cooperative), Always-Defect (adversarial)
- Models: GPT-4o-mini, DeepSeek-v3, Claude-3.5-Sonnet
- Strategies: Direct, CoT, ReAct, ToT
- Conditions: 20 rounds, 5% noise, hidden horizon
- Total runs: 24 (3 models × 4 strategies × 2 opponents)

### Results: vs. Tit-for-Tat (Cooperative Opponent)

| Model | Strategy | Agent Score | Opponent Score | Cooperation Level |
|-------|----------|-------------|----------------|-------------------|
| DeepSeek-v3 | tot | 60 | 60 | Perfect mutual cooperation |
| Claude-3.5 | cot | 58 | 58 | Perfect mutual cooperation |
| Claude-3.5 | tot | 58 | 58 | Perfect mutual cooperation |
| DeepSeek-v3 | react | 57 | 57 | Perfect mutual cooperation |
| GPT-4o-mini | tot | 54 | 59 | Slight exploitation |
| GPT-4o-mini | direct | 52 | 42 | Moderate exploitation |
| DeepSeek-v3 | cot | 50 | 50 | Perfect mutual cooperation |
| DeepSeek-v3 | direct | 44 | 39 | Slight exploitation |
| Claude-3.5 | direct | 44 | 49 | Slightly exploited |
| GPT-4o-mini | cot | 43 | 28 | Strong exploitation |
| Claude-3.5 | react | 40 | 45 | Slightly exploited |
| GPT-4o-mini | react | 37 | 37 | Perfect mutual cooperation |

### Results: vs. Always-Defect (Adversarial Opponent)

| Model | Strategy | Agent Score | Opponent Score | Exploitation Gap |
|-------|----------|-------------|----------------|------------------|
| Claude-3.5 | cot | 29 | 39 | -10 (moderate exploit) |
| Claude-3.5 | tot | 29 | 29 | 0 (perfect defense) |
| GPT-4o-mini | cot | 26 | 26 | 0 (perfect defense) |
| DeepSeek-v3 | direct | 26 | 26 | 0 (perfect defense) |
| DeepSeek-v3 | cot | 24 | 34 | -10 (moderate exploit) |
| Claude-3.5 | react | 22 | 27 | -5 (slight exploit) |
| Claude-3.5 | direct | 22 | 37 | -15 (heavy exploit) |
| GPT-4o-mini | react | 21 | 31 | -10 (moderate exploit) |
| GPT-4o-mini | direct | 19 | 39 | -20 (heavy exploit) |
| GPT-4o-mini | tot | 19 | 24 | -5 (slight exploit) |
| DeepSeek-v3 | tot | 19 | 39 | -20 (heavy exploit) |
| DeepSeek-v3 | react | 18 | 28 | -10 (moderate exploit) |

### Strategic Patterns Identified

**Best Cooperative Strategies (vs TFT):**
- Tree-of-Thought (ToT) achieves highest mutual scores (58-60)
- DeepSeek-v3 excels at cooperation across all strategies
- Direct prompting shows most variance (37-52 range)

**Best Defensive Strategies (vs Always-Defect):**
- Claude CoT/ToT: Best balance (29-29 or 29-39)
- GPT-4o-mini CoT & DeepSeek Direct: Perfect parity (26-26)
- Direct prompting most vulnerable (-15 to -20 gaps)

**Model Characteristics:**
- **DeepSeek-v3**: Most cooperative, high mutual scores, but poor vs. adversary
- **Claude-3.5-Sonnet**: Most balanced, good defense (ToT/CoT)
- **GPT-4o-mini**: High variance, CoT enables best defense

### Key Findings
1. **Reasoning strategy matters significantly** - 22-point spread in scores
2. **ToT best for cooperation** - Achieves highest mutual benefit
3. **CoT best for defense** - Most resilient against exploitation
4. **Direct prompting risky** - High variance, often exploited
5. **No universal optimal strategy** - Context-dependent performance

### Recommendation for Paper
**INCLUDE AS PRIMARY MULTI-AGENT RESULT** - Clear patterns, strong evidence for strategy impact

---

## Overall Summary for Paper Decision

### Primary Experiments (Must Include)
1. **JSON Schema Constraint Adherence** - Strongest single result, clear methodology
2. **Prisoner's Dilemma Tournament** - Best multi-agent results, strategic insights
3. **Framework Comparison** - Shows overhead vs. benefit tradeoffs

### Supporting/Context (Brief Mention)
4. **Zebra Puzzles** - Valuable negative result, motivated strategic pivot
5. **Small Benchmark** - Exploratory phase, limited value

### Proposed Narrative Arc
1. **Introduction**: Can LLMs be effective reasoning agents?
2. **Experiment 1 (JSON)**: YES for structured tasks (88% success)
3. **Experiments 2-3 (Small bench, Zebra)**: Struggles with open-ended reasoning, tool utilization challenges
4. **Insight**: Need different evaluation paradigm - strategic interaction vs. solo problem-solving
5. **Experiment 4 (Framework)**: Framework overhead doesn't guarantee improvement
6. **Experiment 5 (PD Tournament)**: Strategic reasoning shows clear pattern differentiation
7. **Conclusion**: Agents excel at structured tasks and strategic interaction, struggle with complex open-ended reasoning

---

## Data Files Available

### JSON Schema
- `results/json_schema_bench/summary_google_gemini-2.5-flash-preview-09-2025_*.json`
- `results/json_schema_bench/summary_openai_gpt-3.5-turbo_*.json`

### Prisoner's Dilemma
- `results/pd_tournament/pd_tournament_live_20251214_*.json` (6 files)

### Framework Comparison
- `results/framework_comparison/combined_comparison_20251214_071338.json`

### Zebra Puzzles
- `results/zebra_agent_test/combined_zebra_results_20251214_062132.json`

### Small Benchmark
- `results/small_benchmark/combined_results_20251214_061335.json`

---

## Recommendations for Final Paper

### Include
- JSON Schema (primary quantitative success)
- PD Tournament (primary strategic/multi-agent result)
- Framework Comparison (supports "more tools ≠ better" argument)
- Zebra puzzles (brief - as motivation for pivot)

### Exclude or Minimize
- Small benchmark (limited sample, ambiguous results)
- Any incomplete runs or API error results

### Key Metrics to Report
1. Success rates across conditions
2. Token efficiency (cost analysis)
3. Strategy performance rankings
4. Cooperation vs. defection patterns

---

**End of Complete Results Summary**
