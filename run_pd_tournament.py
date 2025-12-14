"""
Author: Aaron Genkin (amg454)
Purpose: Tournament system for testing agent strategies in iterated Prisoner's Dilemma

This script runs a comprehensive tournament where LLM agents with different reasoning
strategies play against classic game theory strategies. Tests cooperation and
strategic decision-making across multiple models and reasoning approaches.

Usage examples:
  python run_pd_tournament.py --iterations 5 --rounds 10 --strategies direct,cot,react,tot --models all
  python run_pd_tournament.py --iterations 1 --rounds 2 --strategies direct --models none --dry-run
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from run_prisoners_dilemma import (
    scripted_agent,
    llm_agent,
    RoundRecord,
    PrisonersDilemmaMatch,
    extract_move_from_text,
)

DEFAULT_MODELS = {
    "deepseek": {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek-v3",
    },
    "gpt4omini": {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o-mini",
    },
    "claude35": {
        "id": "anthropic/claude-3.5-sonnet",
        "name": "Claude-3.5-Sonnet",
    },
}

DEFAULT_STRATEGIES = ["direct", "cot", "react", "tot"]


class LoggedMatch:
    def __init__(self, horizon: int, agent_a, agent_b, noise: float = 0.0):
        self.horizon = horizon
        self.match = PrisonersDilemmaMatch(horizon)
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.round_logs: List[Dict[str, Any]] = []
        self.noise = noise

    def _maybe_flip(self, move: str) -> str:
        if self.noise > 0 and random.random() < self.noise:
            return "D" if move == "C" else "C"
        return move

    def play(self):
        for round_num in range(1, self.horizon + 1):
            move_a, reasoning_a = self.agent_a(round_num, self.horizon, self.match.history)
            move_b, reasoning_b = self.agent_b(round_num, self.horizon, self.match.history)
            move_a = self._maybe_flip(move_a)
            move_b = self._maybe_flip(move_b)
            record: RoundRecord = self.match.play_round(round_num, move_a, move_b, reasoning_a, reasoning_b)
            self.round_logs.append({
                "round": round_num,
                "move_a": record.move_a,
                "move_b": record.move_b,
                "payoff_a": record.payoff_a,
                "payoff_b": record.payoff_b,
                "reasoning_a": reasoning_a,
                "reasoning_b": reasoning_b,
            })
        score_a, score_b = self.match.scores()
        return score_a, score_b


def run_single_match(model_info: Dict[str, Any], mk: str, strat: str, iteration: int, horizon: int, noise: float, hide_horizon: bool, dry_run: bool, opponent: str) -> Dict[str, Any]:
    if dry_run:
        agent_a = scripted_agent("always-cooperate", "A")
    else:
        agent_a = llm_agent(strat, model_info["id"], "A", show_horizon=not hide_horizon)
    agent_b = scripted_agent(opponent, "B")

    logged = LoggedMatch(horizon, agent_a, agent_b, noise=noise)
    score_a, score_b = logged.play()
    return {
        "model_key": mk,
        "model_id": model_info["id"],
        "model_name": model_info["name"],
        "strategy": strat,
        "iteration": iteration,
        "horizon": horizon,
        "noise": noise,
        "hide_horizon": hide_horizon,
        "opponent": opponent,
        "rounds": logged.round_logs,
        "final_scores": {"player_a": score_a, "player_b": score_b},
    }


def run_tournament(iterations: int, horizon: int, strategies: List[str], model_keys: List[str], dry_run: bool, noise: float, hide_horizon: bool, parallel_workers: int, opponent: str) -> List[Dict[str, Any]]:
    if dry_run and not model_keys:
        model_keys = ["dryrun"]
        DEFAULT_MODELS_LOCAL = {"dryrun": {"id": "dryrun/manual", "name": "DryRun-Manual"}}
    else:
        DEFAULT_MODELS_LOCAL = DEFAULT_MODELS

    jobs = []
    for mk in model_keys:
        model_info = DEFAULT_MODELS_LOCAL[mk]
        for strat in strategies:
            for i in range(1, iterations + 1):
                jobs.append((model_info, mk, strat, i))

    results: List[Dict[str, Any]] = []
    if parallel_workers <= 1:
        for model_info, mk, strat, i in jobs:
            try:
                results.append(run_single_match(model_info, mk, strat, i, horizon, noise, hide_horizon, dry_run, opponent))
            except Exception as e:
                results.append({
                    "model_key": mk,
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "strategy": strat,
                    "iteration": i,
                    "horizon": horizon,
                    "error": str(e),
                })
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            future_to_job = {
                ex.submit(run_single_match, model_info, mk, strat, i, horizon, noise, hide_horizon, dry_run, opponent): (model_info, mk, strat, i)
                for model_info, mk, strat, i in jobs
            }
            for fut in as_completed(future_to_job):
                model_info, mk, strat, i = future_to_job[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({
                        "model_key": mk,
                        "model_id": model_info["id"],
                        "model_name": model_info["name"],
                        "strategy": strat,
                        "iteration": i,
                        "horizon": horizon,
                        "error": str(e),
                    })
    return results


def save_results(results: List[Dict[str, Any]], tag: str) -> Path:
    out_dir = Path("results") / "pd_tournament"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pd_tournament_{tag}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run a PD tournament across models and strategies.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of repetitions per (model, strategy)")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per match")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="Comma-separated strategies: direct,cot,react,tot")
    parser.add_argument("--models", default="all", help="Comma-separated model keys: deepseek,gpt4omini,claude35 | all | none (dry-run only)")
    parser.add_argument("--dry-run", action="store_true", help="Use scripted agents for Player A (no LLM calls)")
    parser.add_argument("--noise", type=float, default=0.0, help="Action flip probability (e.g., 0.05)")
    parser.add_argument("--hide-horizon", action="store_true", help="Do not reveal horizon to the LLM")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Number of parallel workers for matches")
    parser.add_argument("--opponent", default="tit-for-tat", help="Scripted opponent: tit-for-tat | always-defect | always-cooperate | random | random-<p> | grim-trigger")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in DEFAULT_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {s}")

    if args.models == "all":
        model_keys = list(DEFAULT_MODELS.keys())
    elif args.models == "none":
        model_keys = []
        if not args.dry_run:
            raise ValueError("models=none requires --dry-run")
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in model_keys:
            if m not in DEFAULT_MODELS:
                raise ValueError(f"Unsupported model key: {m}")

    if not model_keys and not args.dry_run:
        raise ValueError("No models specified and not in dry-run mode")

    results = run_tournament(args.iterations, args.rounds, strategies, model_keys, args.dry_run, args.noise, args.hide_horizon, args.parallel_workers, args.opponent)
    tag = "dry" if args.dry_run else "live"
    out_path = save_results(results, tag)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
