"""
E5 Large Instruct Embedder
==========================
Specialized embedder for intfloat/multilingual-e5-large-instruct model.
"""

from typing import List, Optional, Any
from .hf_local_embedder import HuggingFaceLocalEmbedder


class E5LargeInstructEmbedder(HuggingFaceLocalEmbedder):
    """
    E5 Large Instruct Embedder for multilingual retrieval tasks.

    Model: intfloat/multilingual-e5-large-instruct
    Dimensions: 1024
    Languages: >100
    Size: ~1.3GB
    """

    MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

    def __init__(
        self,
        device: str = "cpu",
        max_seq_length: int = 512,
        **kwargs: Any
    ):
        """
        Initialize E5 Large Instruct embedder.

        Args:
            device: Device for inference
            max_seq_length: Maximum sequence length (stored but not passed to parent)
            **kwargs: Additional arguments
        """
        # Remove max_seq_length from kwargs if present since parent doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_seq_length'}
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            **filtered_kwargs
        )


class E5BaseEmbedder(HuggingFaceLocalEmbedder):
    """
    E5 Base Embedder for multilingual similarity tasks.

    Model: intfloat/multilingual-e5-base
    Dimensions: 768
    Languages: >100
    Size: ~0.3GB
    """

    MODEL_NAME = "intfloat/multilingual-e5-base"

    def __init__(
        self,
        device: str = "cpu",
        max_seq_length: int = 512,
        **kwargs: Any
    ):
        """
        Initialize E5 Base embedder.

        Args:
            device: Device for inference
            max_seq_length: Maximum sequence length (stored but not passed to parent)
            **kwargs: Additional arguments
        """
        # Remove max_seq_length from kwargs if present since parent doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_seq_length'}
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            **filtered_kwargs
        )


class GTEMultilingualBaseEmbedder(HuggingFaceLocalEmbedder):
    """
    GTE Multilingual Base Embedder for multilingual dense retrieval.

    Model: Alibaba-NLP/gte-multilingual-base
    Dimensions: 768
    Languages: >70
    Size: ~0.5GB
    """

    MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

    def __init__(
        self,
        device: str = "cpu",
        max_seq_length: int = 512,
        **kwargs: Any
    ):
        """
        Initialize GTE Multilingual Base embedder.

        Args:
            device: Device for inference
            max_seq_length: Maximum sequence length (stored but not passed to parent)
            **kwargs: Additional arguments
        """
        # Remove max_seq_length from kwargs if present since parent doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_seq_length'}
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            trust_remote_code=True,  # Required for this model
            **filtered_kwargs
        )


class ParaphraseMPNetBaseV2Embedder(HuggingFaceLocalEmbedder):
    """
    Paraphrase MPNet Base V2 Embedder for semantic search.

    Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    Dimensions: 768
    Languages: >50
    Size: ~0.5GB
    """

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(
        self,
        device: str = "cpu",
        max_seq_length: int = 512,
        **kwargs: Any
    ):
        """
        Initialize Paraphrase MPNet Base V2 embedder.

        Args:
            device: Device for inference
            max_seq_length: Maximum sequence length (stored but not passed to parent)
            **kwargs: Additional arguments
        """
        # Remove max_seq_length from kwargs if present since parent doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_seq_length'}
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            **filtered_kwargs
        )


class ParaphraseMiniLML12V2Embedder(HuggingFaceLocalEmbedder):
    """
    Paraphrase MiniLM L12 V2 Embedder for lightweight semantic tasks.

    Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    Dimensions: 384
    Languages: >50
    Size: ~0.2GB
    """

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        device: str = "cpu",
        max_seq_length: int = 512,
        **kwargs: Any
    ):
        """
        Initialize Paraphrase MiniLM L12 V2 embedder.

        Args:
            device: Device for inference
            max_seq_length: Maximum sequence length (stored but not passed to parent)
            **kwargs: Additional arguments
        """
        # Remove max_seq_length from kwargs if present since parent doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_seq_length'}
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            **filtered_kwargs
        )