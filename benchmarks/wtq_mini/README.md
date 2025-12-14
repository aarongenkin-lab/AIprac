WikiTableQuestions Mini (offline)
=================================

Goal: run a small, local slice of WikiTableQuestions (WTQ) to test table reasoning with and without tools (Python, RAG-ish context).

What you need
-------------
1) Clone/download the WTQ repo: `https://github.com/ppasupat/WikiTableQuestions`
2) From that repo, point to the `data` directory (it contains `random-split-1/train.tsv`, `dev.tsv`, `test.tsv` and `tables/` with the HTML tables).

Prepare a mini set
------------------
Use the helper script to build a small JSONL file with table text inlined:

```
.\venv\Scripts\python.exe benchmarks\wtq_mini\prepare_wtq_mini.py ^
  --wtq-root C:\path\to\WikiTableQuestions\data ^
  --split random-split-1-dev.tsv ^
  --max 20 ^
  --out benchmarks\wtq_mini\wtq_mini.jsonl
```

Notes:
- The TSV is expected to have columns: `id`, `question`, `table_id`, `answer` (standard WTQ format). `answer` may be a list string like `["value"]`; the script will keep the raw string.
- Tables are read from `<wtq-root>/tables/{table_id}.html`.
- HTML tables are converted to a plain-text grid (header + rows). This is coarse but sufficient for small-scale evaluation.

Run the mini benchmark
----------------------
After `wtq_mini.jsonl` is built:
```
.\venv\Scripts\python.exe benchmarks\run_wtq_mini.py --max-samples 10 --models deepseek/deepseek-chat anthropic/claude-3.5-sonnet openai/gpt-4o-mini
```

What it does:
- Baseline: sends question + table text directly.
- Enhanced: uses the Enhanced ReAct agent; prompt nudges use of Python for any calculations.
- Metrics: exact/partial match, tokens, time, steps, tools used.

Tuning:
- Adjust `--max-samples` for cost/time.
- You can further filter by table size via `--max-rows` if tables are large.

Files
-----
- `prepare_wtq_mini.py`: builds `wtq_mini.jsonl` from the WTQ data.
- `run_wtq_mini.py`: runs baseline vs enhanced on the mini set.
- `wtq_mini.jsonl`: generated; not checked in.
