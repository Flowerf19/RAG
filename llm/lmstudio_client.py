"""
LMStudio LLM Client - Local LLM via LMStudio OpenAI-compatible API
Implements BaseLLMClient interface
"""

from typing import List, Dict, Optional
from openai import OpenAI

from llm.base_client import BaseLLMClient

try:
    from llm.config_loader import resolve_lmstudio_settings
except ImportError:
    from config_loader import resolve_lmstudio_settings


class LMStudioClient(BaseLLMClient):
    """
    LMStudio LLM Client
    Uses OpenAI-compatible API (no format conversion needed)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LMStudio client

        Args:
            config: Configuration dict with keys:
                - base_url: LMStudio server URL
                - api_key: API key (usually "lm-studio")
                - model: Model name
                - temperature: Sampling temperature
                - top_p: Top-p sampling
                - max_tokens: Max output tokens
        """
        # Resolve settings from config file + overrides
        resolved = resolve_lmstudio_settings(
            override_base_url=config.get("base_url") if config else None,
            override_api_key=config.get("api_key") if config else None,
            override_model=config.get("model") if config else None,
            override_temperature=config.get("temperature") if config else None,
            override_top_p=config.get("top_p") if config else None,
            override_max_tokens=config.get("max_tokens") if config else None,
        )

        super().__init__(resolved)

        # Create OpenAI client pointing to LMStudio
        self._client = OpenAI(
            base_url=self.config["base_url"], api_key=self.config["api_key"]
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate response from LMStudio

        Args:
            messages: OpenAI format messages (used directly!)
            temperature: Override temperature
            max_tokens: Override max_tokens
            **kwargs: Additional OpenAI-compatible parameters

        Returns:
            Generated text
        """
        try:
            # LMStudio hỗ trợ OpenAI format sẵn - không cần convert!
            response = self._client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=temperature or self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.95),
                max_tokens=max_tokens or self.config.get("max_tokens", 512),
                **kwargs,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"LMStudio API error: {e}")

    def is_available(self) -> bool:
        """
        Check if LMStudio server is reachable

        Returns:
            True if server responds
        """
        try:
            # Try to list models as health check
            self._client.models.list()
            return True
        except Exception:
            return False
