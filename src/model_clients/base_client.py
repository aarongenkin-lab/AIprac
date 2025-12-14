"""
Base client interface for LLM interactions
All model clients should inherit from this class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


class BaseModelClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model client

        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model_name = config.get("model_name")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.1)

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the model

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing:
                - response: The model's text response
                - usage: Token usage statistics
                - metadata: Additional metadata (latency, model version, etc.)
        """
        pass

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simplified chat interface for single-turn or multi-turn conversations

        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            conversation_history: Previous messages in the conversation
            **kwargs: Additional parameters

        Returns:
            Same format as generate()
        """
        messages = conversation_history or []
        messages.append({"role": "user", "content": user_message})

        return self.generate(messages, system_prompt, **kwargs)

    def _track_usage(self, usage: Dict[str, Any]) -> None:
        """Track token usage for cost estimation"""
        # Could implement cost tracking here
        logger.debug(f"Token usage: {usage}")

    def _retry_with_backoff(self, func, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Retry a function with exponential backoff

        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds

        Returns:
            Result of the function
        """
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
