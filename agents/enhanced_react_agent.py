"""
Enhanced ReAct Agent with Multiple Tools
Supports search, Python execution, RAG, and calculations
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_clients.openrouter_client import OpenRouterClient
from agents.tools.search_tool import SearchTool
from agents.tools.python_tool import PythonScratchpad
from agents.tools.rag_tool import RAGTool


class ActionType(Enum):
    """Available action types for the agent"""
    SEARCH = "search"
    PYTHON = "python"
    RETRIEVE = "retrieve"
    ANSWER = "answer"


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
    total_time: float = 0.0
    success: bool = False


class EnhancedReActAgent:
    """
    Enhanced ReAct agent with multiple tools:
    - Web search (DuckDuckGo)
    - Python scratchpad (code execution)
    - RAG (document retrieval from knowledge base)
    """

    SYSTEM_PROMPT = """You are an expert reasoning agent equipped with powerful tools.

You solve problems through structured thinking and tool use.

AVAILABLE TOOLS:
1. search - Search the web for current information
2. python - Execute Python code in a scratchpad environment
3. retrieve - Retrieve relevant information from the knowledge base
4. answer - Provide your final answer

TOOL USAGE GUIDELINES:

**search**: Use when you need current information, facts, or external knowledge
Example: "search: current GDP of United States 2024"

**python**: Use for calculations, data manipulation, logic problems, or algorithmic tasks
Example: "python: (1 + 0.045/12)**(12*7) * 10000"
The Python environment persists variables across calls, and includes: math, statistics, itertools, collections

**retrieve**: Use when you need domain-specific strategies or best practices
Example: "retrieve: strategies for solving logic puzzles"
The knowledge base contains expert guides for various problem types

**answer**: Use only when you have enough information to confidently answer
Example: "answer: The compound interest amount is $14,376.03"

FORMAT:
For each step, use this EXACT format:

Thought: [Your reasoning about what to do next]
Action: [search|python|retrieve|answer]
Action Input: [Specific input for the action]

Then wait for an Observation before continuing.

STRATEGY:
1. Start by understanding the problem
2. Check if knowledge base has relevant strategies (use retrieve)
3. Gather information (search) or compute (python) as needed
4. Show your work step-by-step
5. Provide final answer when confident

EXAMPLE:

Thought: I need to calculate compound interest. Let me first check if there's a formula in the knowledge base.
Action: retrieve
Action Input: compound interest formula

Observation: [Knowledge base returns the formula A = P(1 + r/n)^(nt)]

Thought: Great! Now I can use Python to calculate with P=10000, r=0.045, n=12, t=7.
Action: python
Action Input: P = 10000; r = 0.045; n = 12; t = 7; A = P * (1 + r/n)**(n*t); print(f"Final amount: ${A:.2f}")

Observation: Final amount: $14376.03

Thought: I have the answer now.
Action: answer
Action Input: The final amount after 7 years is $14,376.03

Remember:
- Use tools to enhance your capabilities
- Show your reasoning clearly
- Verify calculations when possible
- Cite sources when using retrieved information"""

    def __init__(
        self,
        model_name: str = "openai/gpt-3.5-turbo",
        max_steps: int = 15,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        knowledge_base_path: str = "knowledge_base",
        verbose: bool = True
    ):
        """
        Initialize the enhanced ReAct agent

        Args:
            model_name: LLM model to use
            max_steps: Maximum reasoning steps
            temperature: Generation temperature
            knowledge_base_path: Path to knowledge base directory
            verbose: Print reasoning steps
        """
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()

        # Initialize LLM client
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("HORIZON_BETA_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Set OPENROUTER_API_KEY or HORIZON_BETA_API_KEY")

        config = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        self.client = OpenRouterClient(config)
        self.model_name = model_name
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Initialize tools
        self.search_tool = SearchTool(max_results=5)
        self.python_tool = PythonScratchpad()
        self.rag_tool = RAGTool(knowledge_base_path=knowledge_base_path)

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def reset_conversation(self):
        """Reset conversation history and Python environment"""
        self.conversation_history = []
        self.python_tool.reset()

    def parse_response(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse model output to extract thought, action, and action input

        Returns:
            (thought, action, action_input)
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
        Execute the specified action using appropriate tool

        Args:
            action: Action type
            action_input: Input for the action

        Returns:
            Observation string
        """
        try:
            action_enum = ActionType(action)
        except ValueError:
            return f"Error: Unknown action '{action}'. Valid actions: search, python, retrieve, answer"

        if action_enum == ActionType.SEARCH:
            if self.verbose:
                print(f"  [SEARCH] {action_input}")
            return self.search_tool(action_input)

        elif action_enum == ActionType.PYTHON:
            if self.verbose:
                print(f"  [PYTHON] Executing code...")
            return self.python_tool(action_input)

        elif action_enum == ActionType.RETRIEVE:
            if self.verbose:
                print(f"  [RETRIEVE] {action_input}")
            return self.rag_tool(action_input, top_k=2)

        elif action_enum == ActionType.ANSWER:
            return "Final answer provided."

        return "Action execution failed."

    def reason(self, question: str) -> AgentResponse:
        """
        Main reasoning loop

        Args:
            question: The question/task to solve

        Returns:
            AgentResponse with full reasoning chain
        """
        import time
        start_time = time.time()

        self.reset_conversation()

        # Add initial question
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        steps: List[AgentStep] = []
        final_answer = None
        total_tokens = 0

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"QUESTION: {question}")
            print(f"{'='*70}\n")

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step_num} ---")

            # Get model response
            try:
                response = self.client.generate(
                    messages=self.conversation_history,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                model_output = response['response']
                total_tokens += response['usage']['total_tokens']

                if self.verbose:
                    print(f"Model:\n{model_output}\n")

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": model_output
                })

            except Exception as e:
                if self.verbose:
                    print(f"ERROR: {e}")
                return AgentResponse(
                    steps=steps,
                    final_answer=None,
                    total_tokens=total_tokens,
                    total_time=time.time() - start_time,
                    success=False
                )

            # Parse response
            thought, action, action_input = self.parse_response(model_output)

            if not thought or not action or not action_input:
                if self.verbose:
                    print(f"WARNING: Failed to parse response")
                continue

            # Create step
            step = AgentStep(
                thought=thought,
                action=ActionType(action),
                action_input=action_input,
                step_number=step_num
            )

            # Check if final answer
            if action == "answer":
                final_answer = action_input
                step.observation = "Task completed."
                steps.append(step)

                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"FINAL ANSWER: {final_answer}")
                    print(f"{'='*70}")

                return AgentResponse(
                    steps=steps,
                    final_answer=final_answer,
                    total_tokens=total_tokens,
                    total_time=time.time() - start_time,
                    success=True
                )

            # Execute action
            observation = self.execute_action(action, action_input)
            step.observation = observation
            steps.append(step)

            if self.verbose:
                print(f"Observation:\n{observation[:500]}{'...' if len(observation) > 500 else ''}\n")

            # Add observation to conversation
            self.conversation_history.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        # Max steps reached
        if self.verbose:
            print(f"\n{'='*70}")
            print("MAX STEPS REACHED")
            print(f"{'='*70}")

        return AgentResponse(
            steps=steps,
            final_answer=None,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            success=False
        )

    def print_summary(self, response: AgentResponse):
        """Print summary of agent execution"""
        print(f"\n{'='*70}")
        print("EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {len(response.steps)}")
        print(f"Success: {response.success}")
        print(f"Total tokens: {response.total_tokens}")
        print(f"Total time: {response.total_time:.2f}s")
        print(f"Final answer: {response.final_answer or 'None'}")
        print(f"{'='*70}\n")


def main():
    """Demo the enhanced ReAct agent"""
    print("Enhanced ReAct Agent Demo")
    print("=" * 70)

    # Initialize agent
    agent = EnhancedReActAgent(
        model_name="openai/gpt-3.5-turbo",
        max_steps=15,
        temperature=0.1,
        verbose=True
    )

    # Test questions
    questions = [
        "Calculate the compound interest on $10,000 invested at 4.5% annual rate, compounded monthly, over 7 years.",

        "What strategies should I use to solve a zebra logic puzzle efficiently?",

        "There are 5 houses in a row. The Norwegian lives in the first house. The Norwegian lives next to the blue house. What color is the second house?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST {i}/{len(questions)}")
        print(f"{'#'*70}")

        response = agent.reason(question)
        agent.print_summary(response)

        if i < len(questions):
            input("\nPress Enter to continue to next question...")


if __name__ == "__main__":
    main()
