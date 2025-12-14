"""
Prisoner's Dilemma experiment runner.

Features:
- Manual-vs-manual dry run (no API calls) for protocol sanity.
- Scripted opponents (always-cooperate, always-defect, tit-for-tat).
- LLM hooks via OpenRouter with four strategies aligned to the experiment design:
  * direct   : single-shot minimal instruction (control)
  * cot      : single-shot with step-by-step reasoning instruction (no tools)
  * react    : iterative Thought/Action/Observation with search + calculator tools (10-step cap)
  * tot      : Tree-of-Thought BFS prompt (beam width 3, depth 5) producing a recommended move

Usage examples (manual only):
  python run_prisoners_dilemma.py --rounds 3 --agent-a manual --agent-b manual
  python run_prisoners_dilemma.py --rounds 5 --agent-a manual --agent-b tit-for-tat

LLM examples (requires API keys and network):
  python run_prisoners_dilemma.py --agent-a llm --llm-strategy direct \
      --model-id "openai/gpt-4o-mini" --agent-b tit-for-tat --rounds 10

This script defaults to manual mode so you can test the flow before calling any LLMs.
"""

import argparse
import sys
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from agents.agentic_reasoning import AgenticReasoningFramework
    from agents.enhanced_react_agent import EnhancedReActAgent
    from src.model_clients.openrouter_client import OpenRouterClient
except Exception:
    # Imports are only needed when running LLM modes.
    AgenticReasoningFramework = None
    EnhancedReActAgent = None
    OpenRouterClient = None


PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


@dataclass
class RoundRecord:
    round_num: int
    move_a: str
    move_b: str
    payoff_a: int
    payoff_b: int
    reasoning_a: str = ""
    reasoning_b: str = ""


class PrisonersDilemmaMatch:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.history: List[RoundRecord] = []

    def play_round(self, round_num: int, move_a: str, move_b: str, reasoning_a: str, reasoning_b: str) -> RoundRecord:
        move_a = normalize_move(move_a)
        move_b = normalize_move(move_b)
        payoff_a, payoff_b = PAYOFFS[(move_a, move_b)]
        record = RoundRecord(round_num, move_a, move_b, payoff_a, payoff_b, reasoning_a, reasoning_b)
        self.history.append(record)
        return record

    def scores(self) -> Tuple[int, int]:
        a = sum(r.payoff_a for r in self.history)
        b = sum(r.payoff_b for r in self.history)
        return a, b


def normalize_move(raw: str) -> str:
    raw = (raw or "").strip().upper()
    if raw not in {"C", "D"}:
        raise ValueError("Move must be C or D")
    return raw


def load_base_prompt(prompt_path: Path = ROOT / "prompts" / "prisoners_dilemma_base.txt") -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing base prompt at {prompt_path}")
    return prompt_path.read_text().strip()


def format_history(history: List[RoundRecord]) -> str:
    if not history:
        return "No prior rounds."
    lines = []
    for r in history:
        lines.append(
            f"Round {r.round_num}: A={r.move_a}, B={r.move_b}, payoff A={r.payoff_a}, payoff B={r.payoff_b}"
        )
    return "\n".join(lines)


def extract_move_from_text(text: str) -> str:
    match = re.search(r"Move\s*=\s*([CD])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([CD])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    raise ValueError("Could not extract move (C or D) from model output")


def format_history_for_player(history: List[RoundRecord], player_label: str) -> str:
    """Format history from the perspective of a given player."""
    if not history:
        return "No prior rounds."
    lines = []
    for r in history:
        if player_label == "A":
            my_move, opp_move, my_payoff, opp_payoff = r.move_a, r.move_b, r.payoff_a, r.payoff_b
        else:
            my_move, opp_move, my_payoff, opp_payoff = r.move_b, r.move_a, r.payoff_b, r.payoff_a
        lines.append(
            f"Round {r.round_num}: You={my_move}, Opponent={opp_move}, Your payoff={my_payoff}, Opponent payoff={opp_payoff}"
        )
    return "\n".join(lines)


class LLMAgent:
    def __init__(self, strategy: str, model_id: str, player_label: str, show_horizon: bool = True, temperature: float = 0.1, max_tokens: int = 300):
        if strategy not in {"direct", "cot", "react", "tot"}:
            raise ValueError("strategy must be one of: direct, cot, react, tot")
        self.strategy = strategy
        self.model_id = model_id
        self.player_label = player_label
        self.show_horizon = show_horizon
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_prompt = load_base_prompt()

    def _build_prompt(self, round_num: int, horizon: int, history: List[RoundRecord]) -> str:
        history_block = format_history_for_player(history, self.player_label)
        horizon_line = f"Horizon: {horizon} rounds.\n" if self.show_horizon else ""
        return (
            f"You are Player {self.player_label} in a repeated Prisoner's Dilemma.\n"
            f"{horizon_line}"
            f"History so far (from your perspective):\n{history_block}\n\n"
            f"You are now at round {round_num}. Follow the protocol and output your move."
        )

    def decide(self, round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        if self.strategy == "direct":
            return self._decide_direct(round_num, horizon, history)
        if self.strategy == "cot":
            return self._decide_cot(round_num, horizon, history)
        if self.strategy == "react":
            return self._decide_react(round_num, horizon, history)
        return self._decide_tot(round_num, horizon, history)

    def _decide_direct(self, round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        if OpenRouterClient is None:
            raise RuntimeError("OpenRouterClient not available; cannot run direct mode here")
        client = OpenRouterClient({"model_name": self.model_id, "temperature": self.temperature, "max_tokens": self.max_tokens})
        prompt = self._build_prompt(round_num, horizon, history)
        response = client.generate(messages=[{"role": "user", "content": prompt}], system_prompt=self.base_prompt)
        move = extract_move_from_text(response["response"])
        reasoning = response["response"].strip()
        return move, reasoning

    def _decide_cot(self, round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        if OpenRouterClient is None:
            raise RuntimeError("OpenRouterClient not available; cannot run cot mode here")
        client = OpenRouterClient({"model_name": self.model_id, "temperature": self.temperature, "max_tokens": self.max_tokens})
        prompt = self._build_prompt(round_num, horizon, history) + "\nThink step-by-step and then provide your move."
        response = client.generate(messages=[{"role": "user", "content": prompt}], system_prompt=self.base_prompt)
        move = extract_move_from_text(response["response"])
        reasoning = response["response"].strip()
        return move, reasoning

    def _decide_react(self, round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        if AgenticReasoningFramework is None:
            raise RuntimeError("AgenticReasoningFramework not available; cannot run react mode here")
        agent = AgenticReasoningFramework(model_name=self.model_id, max_steps=6, temperature=self.temperature, verbose=False)
        prompt = self._build_prompt(round_num, horizon, history)
        result = agent.reason(prompt)
        move = extract_move_from_text(result.final_answer or "")
        return move, result.final_answer or ""

    def _decide_tot(self, round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        if OpenRouterClient is None:
            raise RuntimeError("OpenRouterClient not available; cannot run tot mode here")
        client = OpenRouterClient({"model_name": self.model_id, "temperature": self.temperature, "max_tokens": max(self.max_tokens, 600)})
        history_block = format_history_for_player(history, self.player_label)
        user_prompt = (
            f"You are Player {self.player_label} in a repeated Prisoner's Dilemma. Horizon: {horizon} rounds.\n"
            f"History so far (from your perspective):\n{history_block}\n\n"
            "Plan using Tree-of-Thoughts with breadth-first search:\n"
            "- Beam width W=3 candidate reasoning paths per level.\n"
            "- Maximum depth D=5 reasoning steps.\n"
            "- Each node should state a candidate next move (C or D) and a heuristic expected cumulative payoff for you.\n"
            "- Expand the frontier up to depth D, keep top 3 nodes by heuristic each level.\n"
            "- After the search, recommend the immediate move to take now.\n\n"
            "Output format:\n"
            "Candidates:\n"
            "1) Path=<...> | Next Move=<C or D> | Heuristic=<score> | Rationale=<short>\n"
            "2) ...\n"
            "Recommendation:\n"
            "Move = <C or D>\n"
            "Reasoning: <1-2 sentences>"
        )
        response = client.generate(messages=[{"role": "user", "content": user_prompt}], system_prompt=self.base_prompt)
        move = extract_move_from_text(response["response"])
        reasoning = response["response"].strip()
        return move, reasoning


def manual_agent(label: str) -> Callable[[int, int, List[RoundRecord]], Tuple[str, str]]:
    def _agent(round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        while True:
            move = input(f"[{label}] Round {round_num}/{horizon} move (C/D): ").strip().upper()
            if move in {"C", "D"}:
                break
            print("Please enter C or D.")
        rationale = input(f"[{label}] Brief reasoning: ").strip()
        return move, rationale or "manual input"

    return _agent


def scripted_agent(kind: str, player_label: str) -> Callable[[int, int, List[RoundRecord]], Tuple[str, str]]:
    kind = kind.lower()
    if kind == "always-cooperate":
        return lambda r, h, hist: ("C", "Always cooperate")
    if kind == "always-defect":
        return lambda r, h, hist: ("D", "Always defect")
    if kind.startswith("random"):
        # Allow "random" or "random-0.3" style
        p = 0.3
        if "-" in kind:
            try:
                p = float(kind.split("-", 1)[1])
            except ValueError:
                p = 0.3
        def _rand(round_num: int, horizon: int, history: List[RoundRecord]):
            move = "D" if random.random() < p else "C"
            return move, f"Random with p_defect={p}"
        return _rand
    if kind == "grim-trigger":
        def _grim(round_num: int, horizon: int, history: List[RoundRecord]):
            if not history:
                return "C", "Grim: start cooperate"
            # If opponent ever defected, defect forever
            opp_defected = any((r.move_b if player_label == "A" else r.move_a) == "D" for r in history)
            if opp_defected:
                return "D", "Grim: punish forever after first defection"
            return "C", "Grim: cooperate until defect seen"
        return _grim
    if kind == "tit-for-tat":
        def _tft(round_num: int, horizon: int, history: List[RoundRecord]):
            if not history:
                return "C", "Open with cooperation"
            if player_label == "A":
                last_opponent = history[-1].move_b
            else:
                last_opponent = history[-1].move_a
            return last_opponent, f"Mirror last opponent move ({last_opponent})"
        return _tft
    raise ValueError(f"Unknown scripted agent: {kind}")


def llm_agent(strategy: str, model_id: str, player_label: str, show_horizon: bool = True) -> Callable[[int, int, List[RoundRecord]], Tuple[str, str]]:
    llm = LLMAgent(strategy=strategy, model_id=model_id, player_label=player_label, show_horizon=show_horizon)

    def _agent(round_num: int, horizon: int, history: List[RoundRecord]) -> Tuple[str, str]:
        move, reasoning = llm.decide(round_num, horizon, history)
        return move, reasoning

    return _agent


def run_match(horizon: int, agent_a: Callable, agent_b: Callable) -> PrisonersDilemmaMatch:
    match = PrisonersDilemmaMatch(horizon)
    for round_num in range(1, horizon + 1):
        move_a, reasoning_a = agent_a(round_num, horizon, match.history)
        move_b, reasoning_b = agent_b(round_num, horizon, match.history)
        record = match.play_round(round_num, move_a, move_b, reasoning_a, reasoning_b)
        print(f"Round {round_num}: A={record.move_a} (payoff {record.payoff_a}) | B={record.move_b} (payoff {record.payoff_b})")
    score_a, score_b = match.scores()
    print("\nFinal scores -> Player A: {0} | Player B: {1}".format(score_a, score_b))
    return match


def run_match_with_noise(horizon: int, agent_a: Callable, agent_b: Callable, noise: float = 0.0) -> PrisonersDilemmaMatch:
    match = PrisonersDilemmaMatch(horizon)
    def maybe_flip(move: str) -> str:
        if noise > 0 and random.random() < noise:
            return "D" if move == "C" else "C"
        return move

    for round_num in range(1, horizon + 1):
        move_a, reasoning_a = agent_a(round_num, horizon, match.history)
        move_b, reasoning_b = agent_b(round_num, horizon, match.history)
        move_a = maybe_flip(move_a)
        move_b = maybe_flip(move_b)
        record = match.play_round(round_num, move_a, move_b, reasoning_a, reasoning_b)
        print(f"Round {round_num}: A={record.move_a} (payoff {record.payoff_a}) | B={record.move_b} (payoff {record.payoff_b})")
    score_a, score_b = match.scores()
    print("\nFinal scores -> Player A: {0} | Player B: {1}".format(score_a, score_b))
    return match


def main():
    parser = argparse.ArgumentParser(description="Run a Prisoner's Dilemma experiment with manual or LLM agents.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds (horizon)")
    parser.add_argument("--agent-a", default="manual", help="Agent for Player A: manual | always-cooperate | always-defect | tit-for-tat | random | random-<p> | grim-trigger | llm")
    parser.add_argument("--agent-b", default="manual", help="Agent for Player B: manual | always-cooperate | always-defect | tit-for-tat | random | random-<p> | grim-trigger | llm")
    parser.add_argument("--llm-strategy", default="direct", help="LLM strategy when agent is llm: direct | cot | react | tot")
    parser.add_argument("--model-id", default="openai/gpt-4o-mini", help="Model ID for LLM agents")
    parser.add_argument("--noise", type=float, default=0.0, help="Action flip probability (e.g., 0.05)")
    parser.add_argument("--hide-horizon", action="store_true", help="Do not reveal horizon in prompts")
    args = parser.parse_args()

    def build_agent(spec: str, label: str) -> Callable:
        if spec == "manual":
            return manual_agent(label)
        if spec in {"always-cooperate", "always-defect", "tit-for-tat"}:
            return scripted_agent(spec, label[-1])  # label expected "Player A"/"Player B"
        if spec.startswith("random") or spec == "grim-trigger":
            return scripted_agent(spec, label[-1])
        if spec == "llm":
            player_label = "A" if label.endswith("A") else "B"
            return llm_agent(args.llm_strategy, args.model_id, player_label, show_horizon=not args.hide_horizon)
        raise ValueError(f"Unsupported agent type: {spec}")

    agent_a = build_agent(args.agent_a, "Player A")
    agent_b = build_agent(args.agent_b, "Player B")

    run_match_with_noise(args.rounds, agent_a, agent_b, noise=args.noise)


if __name__ == "__main__":
    main()
