"""
Base LLM Client - Abstract interface for all LLM providers
Theo nguyên tắc OOP: Interface segregation và polymorphism
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseLLMClient(ABC):
    """
    Abstract base class cho tất cả LLM clients
    Định nghĩa contract chung mà mọi provider phải implement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LLM client với configuration
        
        Args:
            config: Configuration dict (model, temperature, etc.)
        """
        self.config = config or {}
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response từ LLM
        
        Args:
            messages: OpenAI format messages
                [{"role": "system/user/assistant", "content": "..."}]
            temperature: Sampling temperature (override config)
            max_tokens: Max output tokens (override config)
            **kwargs: Provider-specific parameters
        
        Returns:
            Generated text response
        
        Raises:
            Exception: Nếu có lỗi khi gọi API
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if LLM service is available and configured correctly
        
        Returns:
            True if service is ready, False otherwise
        """
        pass
    
    def get_model_name(self) -> str:
        """
        Get current model name
        
        Returns:
            Model name string
        """
        return self.config.get("model", "unknown")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.get_model_name()})"
