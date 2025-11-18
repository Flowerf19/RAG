"""
Token Counter Utility
Uses tiktoken to accurately count tokens for different models and operations.
"""

import tiktoken
from typing import List
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """Utility class for counting tokens using tiktoken."""

    # Mapping of model names to tiktoken encodings
    MODEL_ENCODINGS = {
        # OpenAI models
        'gpt-4': 'cl100k_base',
        'gpt-4-turbo': 'cl100k_base',
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        'gpt-3.5-turbo': 'cl100k_base',

        # Anthropic models (approximate with cl100k_base)
        'claude-3': 'cl100k_base',
        'claude-3-sonnet': 'cl100k_base',
        'claude-3-haiku': 'cl100k_base',

        # Google models (approximate with cl100k_base)
        'gemini': 'cl100k_base',
        'gemini-pro': 'cl100k_base',
        'gemini-flash': 'cl100k_base',

        # Default encoding for unknown models
        'default': 'cl100k_base'
    }

    def __init__(self):
        """Initialize token counter with available encodings."""
        self.encodings = {}
        for model, encoding_name in self.MODEL_ENCODINGS.items():
            try:
                self.encodings[model] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load encoding {encoding_name} for model {model}: {e}")

    def count_tokens(self, text: str, model: str = 'default') -> int:
        """
        Count tokens in text for a specific model.

        Args:
            text: Text to count tokens for
            model: Model name to determine encoding

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        # Get encoding for model
        encoding = self.encodings.get(model, self.encodings.get('default'))

        if encoding is None:
            logger.warning(f"No encoding available for model {model}, using character count / 4")
            return len(text) // 4  # Rough approximation

        try:
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to encode text with tiktoken: {e}, using character count / 4")
            return len(text) // 4  # Rough approximation

    def count_tokens_batch(self, texts: List[str], model: str = 'default') -> List[int]:
        """
        Count tokens for multiple texts.

        Args:
            texts: List of texts to count tokens for
            model: Model name to determine encoding

        Returns:
            List of token counts
        """
        return [self.count_tokens(text, model) for text in texts]

    def estimate_cost(self, tokens: int, model: str, operation: str = 'input') -> float:
        """
        Estimate cost for tokens (rough approximation).

        Args:
            tokens: Number of tokens
            model: Model name
            operation: 'input' or 'output'

        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates per 1K tokens (as of 2024)
        costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'claude-3': {'input': 0.015, 'output': 0.075},
            'gemini-pro': {'input': 0.00025, 'output': 0.0005},
            'default': {'input': 0.001, 'output': 0.002}
        }

        model_costs = costs.get(model, costs['default'])
        cost_per_1k = model_costs.get(operation, model_costs['input'])

        return (tokens / 1000) * cost_per_1k


# Global token counter instance
token_counter = TokenCounter()


def count_tokens(text: str, model: str = 'default') -> int:
    """Convenience function to count tokens."""
    return token_counter.count_tokens(text, model)


def count_tokens_batch(texts: List[str], model: str = 'default') -> List[int]:
    """Convenience function to count tokens for multiple texts."""
    return token_counter.count_tokens_batch(texts, model)