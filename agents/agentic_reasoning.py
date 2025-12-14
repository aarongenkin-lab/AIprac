"""
Agentic Reasoning Framework
Implements ReAct-style reasoning with thought-action-observation loops
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_clients.openrouter_client import OpenRouterClient


class ActionType(Enum):
    """Available action types for the agent"""
    SEARCH = "search"
    CALCULATE = "calculate"
    ANSWER = "answer"
    CLARIFY = "clarify"


@dataclass
class AgentStep:
    """Represents a single reasoning step"""
    thought: str
    action: ActionType
    action_input: str
    observation: Optional[str] = None
    step_number: int = 0


@dataclass
class AgentResponse:
    """Complete agent response with reasoning chain"""
    steps: List[AgentStep]
    final_answer: Optional[str] = None
    total_tokens: int = 0
    success: bool = False


class AgenticReasoningFramework:
    """
    Framework for agentic reasoning with ReAct pattern.
    Supports multi-step reasoning with action execution and observation feedback.
    """

    SYSTEM_PROMPT = """You are an intelligent reasoning agent. You solve problems through a structured thinking process.

For each step, you must follow this exact format:

Thought: [Your reasoning about what to do next]
Action: [One of: search, calculate, answer, clarify]
Action Input: [The specific input for the action]

After you output an action, you will receive an observation. Then continue with another thought-action pair.

Available Actions:
- search: Search for information (use when you need external knowledge)
- calculate: Perform calculations (use for math operations)
- answer: Provide the final answer (use when you have enough information)
- clarify: Ask for clarification (use when the question is ambiguous)

Rules:
1. Always start with a Thought
2. Always follow with exactly one Action and Action Input
3. Wait for Observation before the next step
4. Use 'answer' action only when you're confident in your response
5. Be concise but thorough in your reasoning

Example:
Thought: I need to find the current price of Bitcoin to answer this question.
Action: search
Action Input: current Bitcoin price USD

[You will receive an Observation here]

Thought: Based on the observation, I can now calculate the total value.
Action: calculate
Action Input: 50000 * 2

[You will receive an Observation here]

Thought: I have all the information needed to provide the final answer.
Action: answer
Action Input: The total value of 2 Bitcoin at current price is $100,000."""

    def __init__(
        self,
        model_name: str = "openai/gpt-3.5-turbo",
        max_steps: int = 10,
        temperature: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize the agentic reasoning framework.

        Args:
            model_name: Model to use for reasoning
            max_steps: Maximum number of reasoning steps
            temperature: Temperature for generation
            verbose: Whether to print reasoning steps
        """
        config = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": 500,
        }
        self.client = OpenRouterClient(config)
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose
        self.conversation_history: List[Dict[str, str]] = []

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

    def parse_response(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse the model's response to extract thought, action, and action input.

        Returns:
            Tuple of (thought, action, action_input)
        """
        lines = text.strip().split('\n')
        thought = None
        action = None
        action_input = None

        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip().lower()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()

        return thought, action, action_input

    def execute_action(self, action: str, action_input: str) -> str:
        """
        Execute the specified action and return an observation.

        Args:
            action: Action type to execute
            action_input: Input for the action

        Returns:
            Observation string
        """
        try:
            action_enum = ActionType(action)
        except ValueError:
            return f"Error: Unknown action '{action}'. Use: search, calculate, answer, or clarify."

        if action_enum == ActionType.SEARCH:
            return self._execute_search(action_input)
        elif action_enum == ActionType.CALCULATE:
            return self._execute_calculate(action_input)
        elif action_enum == ActionType.ANSWER:
            return "Final answer provided."
        elif action_enum == ActionType.CLARIFY:
            return "Clarification needed from user."

        return "Action execution failed."

    def _execute_search(self, query: str) -> str:
        """Execute a search action"""
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if results:
                    observations = []
                    for i, result in enumerate(results[:3], 1):
                        observations.append(f"{i}. {result.get('title', '')}: {result.get('body', '')}")
                    return "\n".join(observations)
                else:
                    return "No search results found."
        except Exception as e:
            return f"Search failed: {str(e)}"

    def _execute_calculate(self, expression: str) -> str:
        """Execute a calculation action"""
        try:
            # Safe eval with limited scope
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'len': len
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation failed: {str(e)}"

    def reason(self, question: str) -> AgentResponse:
        """
        Main reasoning loop. Takes a question and returns the agent's response
        with full reasoning chain.

        Args:
            question: The question to answer

        Returns:
            AgentResponse with reasoning steps and final answer
        """
        self.reset_conversation()
        self.conversation_history.append({"role": "user", "content": question})

        steps: List[AgentStep] = []
        final_answer = None
        total_tokens = 0

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"STEP {step_num}")
                print(f"{'='*60}")

            # Get model's response
            try:
                response = self.client.generate(
                    messages=self.conversation_history,
                    temperature=self.temperature,
                    max_tokens=500
                )

                model_output = response['response']
                usage = response.get('usage', {})
                total_tokens += usage.get('total_tokens', 0)

                if self.verbose:
                    print(f"\nModel Output:\n{model_output}")

                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": model_output
                })

            except Exception as e:
                if self.verbose:
                    print(f"Error getting model response: {e}")
                return AgentResponse(
                    steps=steps,
                    final_answer=None,
                    total_tokens=total_tokens,
                    success=False
                )

            # Parse the response
            thought, action, action_input = self.parse_response(model_output)

            if not thought or not action or not action_input:
                if self.verbose:
                    print(f"Warning: Failed to parse response properly")
                    print(f"Thought: {thought}, Action: {action}, Input: {action_input}")
                continue

            # Create step object
            step = AgentStep(
                thought=thought,
                action=ActionType(action),
                action_input=action_input,
                step_number=step_num
            )

            # Check if this is the final answer
            if action == "answer":
                final_answer = action_input
                step.observation = "Task completed."
                steps.append(step)

                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"FINAL ANSWER")
                    print(f"{'='*60}")
                    print(final_answer)

                return AgentResponse(
                    steps=steps,
                    final_answer=final_answer,
                    total_tokens=total_tokens,
                    success=True
                )

            # Execute the action
            observation = self.execute_action(action, action_input)
            step.observation = observation
            steps.append(step)

            if self.verbose:
                print(f"\nObservation:\n{observation}")

            # Add observation to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        # Max steps reached without final answer
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MAX STEPS REACHED")
            print(f"{'='*60}")

        return AgentResponse(
            steps=steps,
            final_answer=None,
            total_tokens=total_tokens,
            success=False
        )


def main():
    """Demo the agentic reasoning framework"""
    print("Agentic Reasoning Framework Demo")
    print("="*60)

    # Initialize agent
    agent = AgenticReasoningFramework(
        model_name="openai/gpt-3.5-turbo",
        max_steps=10,
        temperature=0.1,
        verbose=True
    )

    # Example questions
    questions = [
        "What is 15% of 250?",
        "Who won the 2024 US presidential election and by how many electoral votes?",
        "If I have 3 apples and buy 2.5 times more, how many do I have?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"QUESTION {i}: {question}")
        print(f"{'#'*60}")

        result = agent.reason(question)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total steps: {len(result.steps)}")
        print(f"Success: {result.success}")
        print(f"Total tokens: {result.total_tokens}")
        print(f"Final answer: {result.final_answer}")

        if i < len(questions):
            input("\nPress Enter for next question...")


if __name__ == "__main__":
    main()
