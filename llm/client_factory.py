"""
LLM Client Factory - Factory pattern for creating LLM clients
Provides convenient methods to create pre-configured clients
"""
from typing import Dict, Optional
from enum import Enum

from llm.base_client import BaseLLMClient
from llm.gemini_client import GeminiClient
from llm.lmstudio_client import LMStudioClient


class LLMProvider(Enum):
    """Enum for supported LLM providers"""
    GEMINI = "gemini"
    LMSTUDIO = "lmstudio"


class LLMClientFactory:
    """
    Factory for creating LLM clients with standard configurations
    Sử dụng Factory Pattern để tạo clients dễ dàng
    """
    
    @staticmethod
    def create(
        provider: LLMProvider,
        config: Optional[Dict] = None
    ) -> BaseLLMClient:
        """
        Create LLM client for specified provider
        
        Args:
            provider: LLMProvider enum value
            config: Optional configuration overrides
        
        Returns:
            Configured LLM client instance
        
        Raises:
            ValueError: If provider is not supported
        
        Example:
            >>> client = LLMClientFactory.create(LLMProvider.GEMINI)
            >>> response = client.generate(messages)
        """
        if provider == LLMProvider.GEMINI:
            return GeminiClient(config)
        elif provider == LLMProvider.LMSTUDIO:
            return LMStudioClient(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def create_gemini(
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> GeminiClient:
        """
        Create Gemini client with custom settings
        
        Args:
            api_key: Gemini API key (if None, uses config/env)
            model: Model name (if None, uses config default)
            temperature: Sampling temperature
            max_tokens: Max output tokens
        
        Returns:
            Configured Gemini client
        
        Example:
            >>> client = LLMClientFactory.create_gemini(temperature=0.9)
        """
        config = {
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Remove None values to let resolve_gemini_settings handle defaults
        config = {k: v for k, v in config.items() if v is not None}
        return GeminiClient(config)
    
    @staticmethod
    def create_lmstudio(
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> LMStudioClient:
        """
        Create LMStudio client with custom settings
        
        Args:
            base_url: LMStudio server URL (if None, uses config default)
            model: Model name (if None, uses config default)
            temperature: Sampling temperature
            max_tokens: Max output tokens
        
        Returns:
            Configured LMStudio client
        
        Example:
            >>> client = LLMClientFactory.create_lmstudio(
            ...     base_url="http://localhost:1234/v1"
            ... )
        """
        config = {
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Remove None values to let resolve_lmstudio_settings handle defaults
        config = {k: v for k, v in config.items() if v is not None}
        return LMStudioClient(config)
    
    @staticmethod
    def create_from_string(provider_name: str) -> BaseLLMClient:
        """
        Create client from string name (convenient for UI)
        
        Args:
            provider_name: "gemini" or "lmstudio"
        
        Returns:
            Configured LLM client
        
        Raises:
            ValueError: If provider name is invalid
        
        Example:
            >>> client = LLMClientFactory.create_from_string("gemini")
        """
        try:
            provider = LLMProvider(provider_name.lower())
            return LLMClientFactory.create(provider)
        except ValueError:
            raise ValueError(
                f"Invalid provider name: {provider_name}. "
                f"Valid options: {[p.value for p in LLMProvider]}"
            )
