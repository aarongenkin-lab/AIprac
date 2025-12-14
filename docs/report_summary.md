Report Draft Notes
==================

Methodology
-----------
- Agent Architectures: Baseline single-pass prompting; Basic ReAct (thought–action–observation with search/calc); Enhanced ReAct integrating Python scratchpad, RAG over local knowledge base, and web search.
- Models: OpenRouter-hosted LLMs: deepseek/deepseek-chat, openai/gpt-4o-mini, anthropic/claude-3.5-sonnet. (Gemini dropped after 400 errors.)
- Tools/Libraries: Custom agents (`agents/agentic_reasoning.py`, `agents/enhanced_react_agent.py`), tool wrappers (`agents/tools/*`), OpenRouter client (`src/model_clients/openrouter_client.py`), RAG over local docs, Python execution sandbox, DuckDuckGo search. Dataset handling via `datasets` (for ZebraLogicBench) and local WTQ clone; evaluation scripts in `benchmarks/`.
- Baselines: Direct model prompting with a structured system prompt, no tools. Comparators: Basic ReAct (search/calc) and Enhanced ReAct (Python+RAG+search).
- Implementation: Python codebase; local virtualenv; `.env` keys for OpenRouter. New runners: `benchmarks/run_zebra_logic_frameworks.py`, `benchmarks/run_hard_zebra.py`, `benchmarks/run_wtq_mini.py`; data prep: `benchmarks/wtq_mini/prepare_wtq_mini.py`.

Evaluation
----------
- Datasets / Tasks:
  - Zebra puzzles: simple 5-house puzzle; medium 4×4 grid; harder 5×6 grid; two classic Einstein/Zebra puzzles; ZebraLogicBench sampled via Hugging Face; WTQ mini-set (19 examples from random-split-1-dev with CSV tables).
- Metrics: Exact/partial match on answers; pass rate; tokens and wall-clock time per puzzle; steps and tools used; qualitative failure modes (wrong entity, incomplete answer, no tool use).
- Key Results (representative samples):
  - Simple zebra (5-house): all models solved baseline; Enhanced added 3–7× tokens/time with no accuracy gain.
  - Hard zebra (Einstein, zebra/water): Enhanced produced wrong owners despite “success”; highlights need for answer checking.
  - ZebraLogicBench, 5×6 (1 puzzle, tight budgets): DeepSeek Baseline pass; all others fail with agents. ZebraLogicBench, 4×4 (1 puzzle): GPT-4o-mini and Claude Baseline pass; agents fail under tight caps. Overall: under low step/token budgets and self-contained puzzles, baseline outperformed agents.
  - WTQ mini (2 samples): DeepSeek Baseline 1/2 exact; Enhanced 1/2 (no exact), with higher cost/time. Larger runs timed out at 4 minutes—needs batching/longer timeouts.
- Strengths/Weaknesses: Baseline excels on small, self-contained tasks. Agents incur overhead; tool use was sparse when budgets were tight or tools were misaligned (search/calc not helpful for table logic; RAG not forced). Enhanced helps theoretically on complex, multi-hop or numeric tasks, but requires room (steps/tokens) and domain-aligned prompts.

Conclusion and Discussion
-------------------------
- Findings: Task complexity and resource budgets determine whether agentic tooling helps. For simple/medium zebra puzzles, baseline prompting is cheaper and as accurate. Agents need larger budgets and domain-specific prompting to justify overhead. Tool inclusion must be matched to task (e.g., Python for numerical/table ops; RAG for document lookup).
- Improvements:
  - Relax step/token caps for agent runs; force initial retrieve/python steps on table/numeric tasks.
  - Strengthen answer checking (exact “ANSWER:” match; structured parses) to avoid false passes.
  - Add ablations (Enhanced minus RAG, minus Python) to quantify each tool’s contribution.
  - Expand WTQ and ZebraLogicBench samples and tune prompts per domain.
  - Parallel/multi-agent coordinator for throughput on large batches.
- Limitations: Small sample sizes in reported runs; some timeouts; limited tool utilization under current prompts; partial coverage of WTQ.

Citations (examples)
--------------------
- WikiTableQuestions: Pasupat & Liang, 2015.
- ZebraLogicBench: allenai/ZebraLogicBench (Hugging Face dataset).
- ReAct: Yao et al., 2022 (Reason+Act framework).
- Tool use/RAG: Lewis et al., 2020 (Retrieval-Augmented Generation).
- Models via OpenRouter: DeepSeek-v3, OpenAI GPT-4o-mini, Anthropic Claude-3.5-Sonnet.
