"""
Gemini LLM Client - Google Gemini API integration
Implements BaseLLMClient interface
"""
import os
import logging
from typing import List, Dict, Optional

import google.generativeai as genai

from llm.base_client import BaseLLMClient

try:
    from llm.config_loader import resolve_gemini_settings
except ImportError:
    from config_loader import resolve_gemini_settings


# Configure logging to suppress warnings
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.WARNING)
os.environ['GRPC_VERBOSITY'] = 'ERROR'


class GeminiClient(BaseLLMClient):
    """
    Gemini LLM Client
    Converts OpenAI format → Gemini format and calls Gemini API
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Gemini client
        
        Args:
            config: Configuration dict with keys:
                - api_key: Gemini API key (optional, can use env var)
                - model: Model name (default from config_loader)
                - temperature: Sampling temperature
                - top_p: Top-p sampling
                - max_tokens: Max output tokens
        """
        # Resolve settings from config file + overrides
        resolved = resolve_gemini_settings(
            override_api_key=config.get("api_key") if config else None,
            override_model=config.get("model") if config else None,
            override_temperature=config.get("temperature") if config else None,
            override_top_p=config.get("top_p") if config else None,
            override_max_tokens=config.get("max_tokens") if config else None,
        )
        
        super().__init__(resolved)
        
        # Configure Gemini API
        genai.configure(api_key=self.config["api_key"])
        self._model = genai.GenerativeModel(self.config["model"])
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from Gemini
        
        Args:
            messages: OpenAI format messages
            temperature: Override temperature
            max_tokens: Override max_tokens
            **kwargs: Additional Gemini-specific parameters
        
        Returns:
            Generated text
        """
        try:
            # Convert OpenAI format → Gemini string format
            prompt = self._convert_to_gemini_format(messages)
            
            # Prepare generation config
            gen_config = {
                "temperature": temperature or self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.95),
                "max_output_tokens": max_tokens or self.config.get("max_tokens", 1024),
            }
            
            # Generate
            response = self._model.generate_content(
                prompt,
                generation_config=gen_config
            )
            
            # Clean response (remove "Assistant:" prefix if exists)
            return self._clean_response(response.text)
        
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Gemini is configured correctly
        
        Returns:
            True if API key is set
        """
        return bool(self.config.get("api_key"))
    
    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI Chat Format → Gemini String Format
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}, ...]
        
        Returns:
            "System: ...\nUser: ...\nAssistant: ...\nUser: ..."
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if not content:
                continue
            
            # Map roles
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    def _clean_response(self, response_text: str) -> str:
        """
        Remove "Assistant:" prefix if Gemini adds it
        
        Args:
            response_text: Raw response from Gemini
        
        Returns:
            Cleaned text
        """
        text = response_text.strip()
        
        # Remove "Assistant:" prefix (case-insensitive)
        prefixes = ["Assistant:", "Assistant :", "assistant:", "ASSISTANT:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        return text
