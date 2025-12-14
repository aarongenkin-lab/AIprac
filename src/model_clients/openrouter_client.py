"""
Author: Aaron Genkin (amg454)
Purpose: API client for accessing multiple LLM providers through OpenRouter

This module provides a unified interface to interact with various LLM providers
including OpenAI, Anthropic, and others through the OpenRouter API gateway.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional
import logging

from .base_client import BaseModelClient

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseModelClient):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            self.api_key = os.getenv("HORIZON_BETA_API_KEY")

        if not self.api_key:
            raise ValueError(f"API key not found in environment variable {api_key_env}")

        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.initial_delay = config.get("initial_delay", 1.0)

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Long Horizon Reasoning Research"
        }

        data = {
            "model": self.model_name,
            "messages": full_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        def _make_request():
            start = time.time()
            response = requests.post(self.endpoint, headers=headers, json=data, timeout=self.timeout)
            latency = time.time() - start

            response.raise_for_status()
            result = response.json()

            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError(f"Invalid response format: {result}")

            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            self._track_usage(usage)

            return {
                "response": content,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "metadata": {
                    "model": self.model_name,
                    "latency_seconds": latency,
                    "finish_reason": result["choices"][0].get("finish_reason"),
                    "provider": "openrouter"
                }
            }

        try:
            return self._retry_with_backoff(
                _make_request,
                max_retries=self.max_retries,
                initial_delay=self.initial_delay
            )
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise


class OpenAIClient(BaseModelClient):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"API key not found in environment variable {api_key_env}")

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        def _make_request():
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            latency = time.time() - start

            content = response.choices[0].message.content
            usage = response.usage

            return {
                "response": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "metadata": {
                    "model": self.model_name,
                    "latency_seconds": latency,
                    "finish_reason": response.choices[0].finish_reason,
                    "provider": "openai"
                }
            }

        return self._retry_with_backoff(_make_request)


class AnthropicClient(BaseModelClient):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"API key not found in environment variable {api_key_env}")

        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:

        def _make_request():
            start = time.time()

            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt or "",
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            latency = time.time() - start

            content = response.content[0].text
            usage = response.usage

            return {
                "response": content,
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens
                },
                "metadata": {
                    "model": self.model_name,
                    "latency_seconds": latency,
                    "finish_reason": response.stop_reason,
                    "provider": "anthropic"
                }
            }

        return self._retry_with_backoff(_make_request)
