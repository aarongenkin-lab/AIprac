"""
Main experiment runner for long-horizon reasoning evaluation
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from model_clients import OpenRouterClient, OpenAIClient, AnthropicClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main class for running LLM reasoning experiments"""

    def __init__(self, config_path: str = "config/experiment_configs.yaml"):
        """
        Initialize the experiment runner

        Args:
            config_path: Path to experiment configuration file
        """
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.model_configs = self._load_model_configs()
        self.clients = {}
        self.results = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration"""
        full_path = self.project_root / config_path
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configurations"""
        config_path = self.project_root / "config" / "model_configs.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_client(self, model_key: str):
        """Get or create a model client"""
        if model_key in self.clients:
            return self.clients[model_key]

        model_config = self.model_configs["models"][model_key]
        provider = model_config["provider"]

        # Create appropriate client based on provider
        if provider == "openrouter":
            client = OpenRouterClient(model_config)
        elif provider == "openai":
            client = OpenAIClient(model_config)
        elif provider == "anthropic":
            client = AnthropicClient(model_config)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.clients[model_key] = client
        return client

    def _load_system_prompt(self, prompt_filename: str) -> str:
        """Load a system prompt from file"""
        prompt_path = self.project_root / "prompts" / prompt_filename
        with open(prompt_path, 'r') as f:
            return f.read().strip()

    def _load_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load benchmark tasks"""
        tasks = []

        # Load custom tasks
        custom_tasks_path = self.project_root / "benchmarks" / "custom_tasks" / "sample_tasks.json"
        if custom_tasks_path.exists():
            with open(custom_tasks_path, 'r') as f:
                custom_tasks = json.load(f)
                tasks.extend(custom_tasks)

        # Filter by task type if specified
        if task_type:
            tasks = [t for t in tasks if t["type"] == task_type]

        return tasks

    def _generate_experiment_id(self, model: str, prompt: str, task_id: str, tools: List[str]) -> str:
        """Generate unique experiment ID"""
        components = f"{model}_{prompt}_{task_id}_{'_'.join(sorted(tools))}"
        return hashlib.md5(components.encode()).hexdigest()[:12]

    def run_single_experiment(
        self,
        model_key: str,
        system_prompt_file: str,
        task: Dict[str, Any],
        tools: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single experiment

        Args:
            model_key: Key identifying the model in config
            system_prompt_file: Filename of system prompt
            task: Task dictionary
            tools: List of enabled tools
            **kwargs: Additional parameters

        Returns:
            Experiment result dictionary
        """
        tools = tools or []
        experiment_id = self._generate_experiment_id(
            model_key, system_prompt_file, task["id"], tools
        )

        logger.info(f"Running experiment {experiment_id}")
        logger.info(f"  Model: {model_key}")
        logger.info(f"  Prompt: {system_prompt_file}")
        logger.info(f"  Task: {task['id']} ({task['type']})")
        logger.info(f"  Tools: {tools if tools else 'None'}")

        try:
            # Get client and system prompt
            client = self._get_client(model_key)
            system_prompt = self._load_system_prompt(system_prompt_file)

            # Run the task
            start_time = datetime.now()
            response = client.chat(
                user_message=task["prompt"],
                system_prompt=system_prompt,
                **kwargs
            )
            end_time = datetime.now()

            # Compile results
            result = {
                "experiment_id": experiment_id,
                "timestamp": start_time.isoformat(),
                "model": model_key,
                "system_prompt_file": system_prompt_file,
                "task_id": task["id"],
                "task_type": task["type"],
                "task_difficulty": task.get("difficulty", "unknown"),
                "tools_enabled": tools,
                "model_response": response["response"],
                "expected_answer": task.get("expected_answer"),
                "usage": response["usage"],
                "metadata": response["metadata"],
                "runtime_seconds": (end_time - start_time).total_seconds(),
                "success": None,  # To be filled by evaluator
                "evaluation_scores": {}  # To be filled by evaluator
            }

            # Save result
            self._save_result(result)
            self.results.append(result)

            logger.info(f"  ✓ Completed in {result['runtime_seconds']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"  ✗ Experiment failed: {e}")
            error_result = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "model": model_key,
                "system_prompt_file": system_prompt_file,
                "task_id": task["id"],
                "task_type": task["type"],
                "tools_enabled": tools,
                "error": str(e),
                "success": False
            }
            self._save_result(error_result)
            return error_result

    def _save_result(self, result: Dict[str, Any]) -> None:
        """Save experiment result to file"""
        output_dir = self.project_root / "results" / "raw_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{result['experiment_id']}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

    def run_batch(
        self,
        models: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        tool_conditions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a batch of experiments

        Args:
            models: List of model keys (None = use config)
            prompts: List of prompt files (None = use config)
            task_types: List of task types (None = use config)
            tool_conditions: List of tool condition names (None = use config)

        Returns:
            List of experiment results
        """
        # Use config defaults if not specified
        models = models or self.config["test_models"]
        prompts = prompts or self.config["system_prompts"]
        task_types = task_types or self.config["task_types"]
        tool_conditions = tool_conditions or list(self.config["tool_conditions"].keys())

        batch_results = []

        # Load all tasks
        all_tasks = self._load_tasks()

        # Generate experiment combinations
        total_experiments = 0
        for model in models:
            for prompt in prompts:
                for task in all_tasks:
                    # Filter by task type
                    if task["type"] not in task_types:
                        continue

                    for tool_condition in tool_conditions:
                        tools = self.config["tool_conditions"][tool_condition]["enabled_tools"]

                        # Run experiment
                        result = self.run_single_experiment(
                            model_key=model,
                            system_prompt_file=prompt,
                            task=task,
                            tools=tools
                        )
                        batch_results.append(result)
                        total_experiments += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch complete: {total_experiments} experiments run")
        logger.info(f"Results saved to: {self.project_root / 'results' / 'raw_outputs'}")
        logger.info(f"{'='*60}\n")

        return batch_results

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        if not self.results:
            return {"message": "No results to summarize"}

        total = len(self.results)
        successful = len([r for r in self.results if r.get("success") is not False])
        failed = total - successful

        # Group by dimensions
        by_model = {}
        by_prompt = {}
        by_task_type = {}

        for result in self.results:
            model = result.get("model", "unknown")
            prompt = result.get("system_prompt_file", "unknown")
            task_type = result.get("task_type", "unknown")

            by_model[model] = by_model.get(model, 0) + 1
            by_prompt[prompt] = by_prompt.get(prompt, 0) + 1
            by_task_type[task_type] = by_task_type.get(task_type, 0) + 1

        summary = {
            "total_experiments": total,
            "successful": successful,
            "failed": failed,
            "breakdown": {
                "by_model": by_model,
                "by_prompt": by_prompt,
                "by_task_type": by_task_type
            }
        }

        return summary


def main():
    """Main entry point for running experiments"""
    runner = ExperimentRunner()

    # Example: Run a small pilot
    logger.info("Starting pilot experiment run...")

    # Run just 1 model, 2 prompts, all tasks, no tools
    results = runner.run_batch(
        models=["openrouter_gpt4"],
        prompts=["minimal.txt", "standard_cot.txt"],
        tool_conditions=["none"]
    )

    # Generate summary
    summary = runner.generate_summary_report()
    logger.info("\nExperiment Summary:")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
