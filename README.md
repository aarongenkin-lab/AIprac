# Agent-Based Reasoning Framework: Comparative Study

**Author:** Aaron Genkin (amg454)
**Institution:** Cornell University
**Project Period:** October 2025 - December 2025

## Overview

This project implements and evaluates multiple agent-based reasoning frameworks for large language models (LLMs). The research examines how different reasoning strategies, tool integrations, and framework architectures affect performance across diverse problem types including mathematical reasoning, logical deduction, constraint satisfaction, and strategic decision-making.

## Research Objectives

The project systematically investigates:

1. **Reasoning Strategy Comparison**: How do different reasoning patterns (ReAct, Chain-of-Thought, Direct) affect problem-solving performance?
2. **Tool Integration Impact**: What is the effect of tool access (Python execution, search, RAG) on reasoning quality across task types?
3. **Framework Architecture**: How do custom implementations compare to established frameworks (LangChain, LlamaIndex, CrewAI)?
4. **Multi-Agent Dynamics**: What patterns emerge in strategic multi-agent interactions (game theory scenarios)?

## Project Structure

```
AIpracticum/
├── agents/                      # Agent implementations
│   ├── agentic_reasoning.py    # Base ReAct framework
│   ├── enhanced_react_agent.py # Extended ReAct with tools
│   └── tools/                  # Tool implementations
├── src/
│   ├── model_clients/          # LLM API clients
│   │   ├── base_client.py
│   │   └── openrouter_client.py
│   └── experiment_runner.py    # Experiment orchestration
├── benchmarks/
│   └── custom_tasks/           # Curated benchmark tasks
├── knowledge_base/             # RAG knowledge sources
├── prompts/                    # System prompt templates
├── config/                     # Model and experiment configs
├── results/                    # Experimental outputs
├── docs/                       # Additional documentation
├── run_small_benchmark.py      # Core reasoning benchmark
├── test_agent_on_zebra.py      # Logic puzzle evaluation
├── compare_agent_frameworks.py # Framework comparison study
├── run_pd_tournament.py        # Game theory tournament
└── requirements.txt            # Python dependencies
```

## Core Components

### Agent Implementations

**Enhanced ReAct Agent** (`agents/enhanced_react_agent.py`)
- Implements the Reason-Act pattern for iterative problem solving
- Integrates multiple tools: Python execution, web search, RAG retrieval
- Supports configurable reasoning strategies and step limits
- Tracks token usage and performance metrics

**Agentic Reasoning Framework** (`agents/agentic_reasoning.py`)
- Basic ReAct implementation with search and calculation tools
- Demonstrates fundamental agent architecture patterns
- Used as baseline for framework comparisons

### Tool Integration

**Python Executor**
- Sandboxed code execution environment
- Supports mathematical computation and algorithmic problem solving
- Error handling and output capture

**Web Search**
- DuckDuckGo integration via ddgs package
- Information retrieval for knowledge-intensive tasks
- Result parsing and summarization

**RAG Retrieval**
- Knowledge base integration for domain-specific information
- Vector-based document retrieval
- Context-aware response generation

### Model Clients

**OpenRouter Client** (`src/model_clients/openrouter_client.py`)
- Unified interface for multiple LLM providers (OpenAI, Anthropic, etc.)
- Automatic retry logic with exponential backoff
- Token usage tracking and cost monitoring
- Support for GPT-4, Claude, DeepSeek, and other models

**Base Client** (`src/model_clients/base_client.py`)
- Abstract base class for all model clients
- Standardized response format across providers
- Error handling and logging infrastructure

## Experimental Framework

### Experiment 1: Reasoning Strategy Benchmark

**Script:** `run_small_benchmark.py`

**Objective:** Evaluate agent performance across diverse reasoning tasks

**Design:**
- Models: GPT-4o-mini, DeepSeek-v3
- Tasks: Mathematical reasoning, logical deduction, code generation
- Metrics: Success rate, token usage, reasoning steps, execution time

**Task Categories:**
- Mathematical computation (compound interest, optimization)
- Logical reasoning (multi-step inference chains)
- Code generation (algorithm implementation)
- Common sense reasoning
- Pattern recognition

### Experiment 2: Constraint Satisfaction

**Script:** `test_agent_on_zebra.py`

**Objective:** Test agent capability on complex logic puzzles requiring constraint satisfaction

**Design:**
- Task: Zebra puzzle (Einstein's riddle) and variants
- Dataset: ZebraLogicBench (allenai)
- Models: DeepSeek-v3, GPT-4o-mini
- Evaluation: Solution correctness, reasoning trace quality

**Implementation Details:**
- Agents prompted to use knowledge base for solving strategies
- Python executor available for constraint checking
- Step-by-step reasoning trace analysis
- Performance metrics include success rate, steps to solution, token efficiency

### Experiment 3: Framework Comparison

**Script:** `compare_agent_frameworks.py`

**Objective:** Benchmark custom framework against established alternatives

**Frameworks Tested:**
1. Baseline: Direct prompting (no agent framework)
2. Basic ReAct: Search and calculation tools only
3. Enhanced ReAct: Full tool suite (Python, RAG, search)

**Comparison Dimensions:**
- Task success rate
- Token efficiency
- Response time
- Tool utilization patterns
- Error handling robustness

**Test Protocol:**
- Consistent test puzzle across all frameworks
- Models: DeepSeek-v3, GPT-4o-mini, Claude-3.5-Sonnet
- Metrics: Accuracy, computational cost, reasoning quality

### Experiment 4: Multi-Agent Strategic Behavior

**Script:** `run_pd_tournament.py`

**Objective:** Analyze cooperation and strategic decision-making in iterated Prisoner's Dilemma

**Design:**
- Agent strategies: Direct, Chain-of-Thought, ReAct, Tree-of-Thought
- Opponent: Tit-for-Tat scripted agent
- Game parameters: Configurable rounds, noise, horizon visibility

**Strategic Behaviors Tested:**
- Cooperation emergence
- Defection patterns
- Learning from history
- Strategy adaptation

**Implementation:**
- Round-by-round decision logging
- Reasoning trace capture
- Payoff tracking
- Statistical analysis of strategy effectiveness

### Experiment 5: JSON Schema Constraint Adherence

**Results:** `BENCHMARK_RESULTS.md`

**Objective:** Evaluate constrained generation capabilities for agentic systems

**Findings:**
- Google Gemini 2.5 Flash: 88% pass rate (best performance)
- GPT-3.5-turbo: 62% pass rate (baseline)
- GPT-4o: 60% pass rate (limited by API issues)

**Implications:**
- Constraint adherence remains challenging even for frontier models
- Validation layers essential for production systems
- Cost-performance tradeoffs favor Gemini for this task type

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)
- API keys for LLM providers (OpenRouter, OpenAI, or Anthropic)

### Installation Steps

1. Clone the repository
```bash
cd AIpracticum
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure API keys

Create a `.env` file in the project root:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
# Or use individual provider keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Configuration

**Model Configuration** (`config/model_configs.yaml`)
- Define which models to test
- Set model-specific parameters (temperature, max tokens)
- Configure API endpoints and credentials

**Experiment Configuration** (`config/experiment_configs.yaml`)
- Specify experimental conditions
- Set tool availability
- Define task selection criteria

## Running Experiments

### Quick API Test
```bash
python test_api.py
```
Verifies API connectivity and model availability.

### Small Benchmark
```bash
python run_small_benchmark.py
```
Runs core reasoning benchmark across multiple models.

### Zebra Puzzle Test
```bash
python test_agent_on_zebra.py
```
Evaluates agents on constraint satisfaction problems.

### Framework Comparison
```bash
python compare_agent_frameworks.py
```
Compares baseline, basic, and enhanced agent frameworks.

### Prisoner's Dilemma Tournament
```bash
python run_pd_tournament.py --iterations 5 --rounds 10 --models all
```
Runs strategic interaction experiments.

### Custom Experiments

```python
from src.experiment_runner import ExperimentRunner

runner = ExperimentRunner()
results = runner.run_batch(
    models=["openai/gpt-4o-mini"],
    prompts=["standard_cot.txt"],
    task_types=["math", "logic"]
)
```

## Results and Analysis

### Output Structure

All experimental results are saved to `results/` in JSON format:

```json
{
  "experiment_id": "unique_hash",
  "timestamp": "2025-12-14T...",
  "model": "model_identifier",
  "task_id": "task_identifier",
  "success": true,
  "steps": 5,
  "tokens": 1234,
  "time": 2.5,
  "answer": "model response",
  "expected": "correct answer"
}
```

### Key Metrics

1. **Success Rate**: Percentage of correctly solved tasks
2. **Token Efficiency**: Average tokens per successful solution
3. **Reasoning Steps**: Number of iterations to solution
4. **Execution Time**: Wall-clock time per task
5. **Tool Utilization**: Frequency and effectiveness of tool use

## Technical Implementation Details

### Reasoning Strategies

**ReAct Pattern:**
```
Thought: [Analyze the problem and plan approach]
Action: [Select tool: search, calculate, python, rag]
Action Input: [Specific query or code]
Observation: [Tool output]
... [Repeat until solution found]
Final Answer: [Synthesized response]
```

**Chain-of-Thought:**
- Step-by-step breakdown of reasoning process
- Explicit statement of intermediate conclusions
- Self-verification of logical consistency

**Direct Prompting:**
- Single-shot generation without iteration
- Baseline for comparison with agent-based approaches

### Error Handling

- Automatic retry with exponential backoff for API failures
- Timeout management for long-running operations
- Graceful degradation when tools unavailable
- Comprehensive logging for debugging and analysis

## Dependencies

### Core Libraries
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.18.0` - Anthropic API client
- `requests>=2.31.0` - HTTP requests
- `pyyaml>=6.0` - Configuration parsing
- `python-dotenv>=1.0.0` - Environment variable management
- `ddgs>=0.9.0` - DuckDuckGo search

### Analysis and Visualization
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization

### Optional
- `langchain` - For framework comparison experiments
- `llama-index` - For framework comparison experiments
- `datasets` - For loading benchmark datasets

## Cost Estimates

Approximate costs per experiment using OpenRouter:

- GPT-4o-mini: ~$0.05 per task
- Claude-3.5-Sonnet: ~$0.04 per task
- DeepSeek-v3: ~$0.01 per task
- Gemini 2.0 Flash: ~$0.01 per task

For a full experimental suite (400 runs across 3 models): $40-60

## Limitations and Future Work

### Current Limitations
1. Limited to text-based tasks (no vision or audio)
2. Synchronous execution (no parallelization)
3. English language only
4. Fixed set of tools (extensible but requires implementation)

### Future Research Directions
1. Multi-modal agent integration
2. Hierarchical agent architectures
3. Meta-learning for strategy selection
4. Human-in-the-loop evaluation
5. Real-world task deployment

## Citation

If you use this framework in your research, please cite:

```
Genkin, A. (2025). Agent-Based Reasoning Framework: Comparative Study.
Cornell University.
```

## License

Academic research project for Cornell University. Contact author for usage permissions.

## Contact

Aaron Genkin
amg454@cornell.edu
Cornell University

## Acknowledgments

- Based on OpenRouter API infrastructure
- Inspired by LongBench v2, Berkeley Function-Calling Leaderboard
- Built using established agent architecture patterns from recent LLM research
