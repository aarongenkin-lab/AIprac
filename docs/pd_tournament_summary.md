# Prisoner’s Dilemma Tournament (Small Batches, 2 iterations × 10 rounds, TFT opponent)

Source runs (JSON outputs):
- `results/pd_tournament/pd_tournament_live_20251214_084307.json` (DeepSeek-v3)
- `results/pd_tournament/pd_tournament_live_20251214_085251.json` (GPT-4o-mini)
- `results/pd_tournament/pd_tournament_live_20251214_085635.json` (Claude-3.5-Sonnet)

Setup
- Opponent: tit-for-tat scripted agent.
- Strategies tested: direct (control), cot (chain-of-thought, no tools).
- Horizon: 10 rounds per match.
- Iterations: 2 per (model, strategy).

Results (LLM score is Player A; opponent score is Player B)

| Model               | Strategy | Iteration | Score A | Score B |
| ------------------- | -------- | --------- | ------- | ------- |
| DeepSeek-v3         | direct   | 1         | 30      | 30      |
| DeepSeek-v3         | direct   | 2         | 30      | 30      |
| DeepSeek-v3         | cot      | 1         | 30      | 30      |
| DeepSeek-v3         | cot      | 2         | 30      | 30      |
| GPT-4o-mini         | direct   | 1         | 30      | 30      |
| GPT-4o-mini         | direct   | 2         | 30      | 30      |
| GPT-4o-mini         | cot      | 1         | 30      | 30      |
| GPT-4o-mini         | cot      | 2         | 30      | 30      |
| Claude-3.5-Sonnet   | direct   | 1         | 30      | 30      |
| Claude-3.5-Sonnet   | direct   | 2         | 30      | 30      |
| Claude-3.5-Sonnet   | cot      | 1         | 30      | 30      |
| Claude-3.5-Sonnet   | cot      | 2         | 30      | 30      |

## Noise + Hidden Horizon (1 iteration × 20 rounds, TFT opponent, noise=0.05, horizon hidden)

Source run: `results/pd_tournament/pd_tournament_live_20251214_091646.json`

| Model               | Strategy | Score A | Score B |
| ------------------- | -------- | ------- | ------- |
| GPT-4o-mini         | direct   | 52      | 42      |
| GPT-4o-mini         | cot      | 43      | 28      |
| DeepSeek-v3         | direct   | 44      | 39      |
| DeepSeek-v3         | cot      | 50      | 50      |
| Claude-3.5-Sonnet   | direct   | 44      | 49      |
| Claude-3.5-Sonnet   | cot      | 58      | 58      |

## Noise + Hidden Horizon + Adversarial Opponent (Always-Defect, 1 iteration × 20 rounds, noise=0.05, horizon hidden)

Source run: `results/pd_tournament/pd_tournament_live_20251214_092813.json`

| Model               | Strategy | Score A | Score B |
| ------------------- | -------- | ------- | ------- |
| GPT-4o-mini         | direct   | 19      | 39      |
| GPT-4o-mini         | cot      | 26      | 26      |
| DeepSeek-v3         | direct   | 26      | 26      |
| DeepSeek-v3         | cot      | 24      | 34      |
| Claude-3.5-Sonnet   | direct   | 22      | 37      |
| Claude-3.5-Sonnet   | cot      | 29      | 39      |

Notes
- Baseline (no noise, horizon shown) runs converged to full cooperation (30–30) vs tit-for-tat across models/strategies.
- With noise (5%) and hidden horizon, behaviors diverged; some strategies exploited small advantages (e.g., GPT-4o-mini direct 52–42) while others stayed symmetric.
- With an adversarial opponent (always-defect), at least one defection is guaranteed; payoffs drop and differentiation appears (e.g., GPT-4o-mini direct underperforms, GPT-4o-mini/DeepSeek cot reach parity).
- For deeper contrast, run more iterations and include additional opponents (grim-trigger, random-p) with noise.
