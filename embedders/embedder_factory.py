"""
Embedding Factory
=================
Creates embedder instances from configuration.
"""

from typing import Dict, Optional, Any

from .embedder_type import EmbedderType
from .model.embedding_profile import EmbeddingProfile
from .i_embedder import IEmbedder
from .providers.ollama_embedder import OllamaEmbedder
from .providers.ollama.gemma_embedder import GemmaEmbedder
from .providers.ollama.bge3_embedder import BGE3Embedder
from .providers.huggingface.hf_api_embedder import HuggingFaceApiEmbedder
from .providers.huggingface.hf_local_embedder import HuggingFaceLocalEmbedder


class EmbedderFactory:
    """
    Factory to build embedders based on profile and type.
    Single Responsibility: centralize embedder instantiation logic.
    """

    def __init__(self, registry: Dict[EmbedderType, type] | None = None):
        self._registry = registry or {}

    def create(self, embedder_type: EmbedderType, profile: EmbeddingProfile, **kwargs: Any) -> IEmbedder:
        """Create the requested embedder."""
        if embedder_type == EmbedderType.OLLAMA:
            # Default to GemmaEmbedder for OLLAMA type
            return GemmaEmbedder(
                profile=profile,
                base_url=kwargs.get('base_url', 'http://localhost:11434')
            ) # pyright: ignore[reportAbstractUsage]
        elif embedder_type == EmbedderType.HUGGINGFACE:
            # Create HuggingFace embedder (default to API mode)
            use_api = kwargs.get('use_api', True)
            if use_api:
                return HuggingFaceApiEmbedder(
                    profile=profile,
                    model_name=kwargs.get('model_name'),
                    api_token=kwargs.get('api_token')
                ) # pyright: ignore[reportAbstractUsage]
            else:
                return HuggingFaceLocalEmbedder(
                    profile=profile,
                    model_name=kwargs.get('model_name'),
                    device=kwargs.get('device', 'cpu')
                ) # pyright: ignore[reportAbstractUsage]
        raise ValueError(f"Unsupported embedder type: {embedder_type!r}")

    def create_ollama_nomic(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> OllamaEmbedder:
        """
        Factory method cho Ollama Nomic embedder.
        
        Note: Config nằm trong OllamaEmbedder class.

        Args:
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            OllamaEmbedder: Ollama Nomic embedder
        """
        return OllamaEmbedder.create_default(base_url=base_url)

    def create_bge_m3(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> BGE3Embedder:
        """
        Factory method cho Ollama BGE-M3 embedder.
        
        Note: Config nằm trong BGE3Embedder class (MODEL_ID, DIMENSION, MAX_TOKENS).

        Args:
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            BGE3Embedder: Ollama BGE-M3 embedder
        """
        return BGE3Embedder.create_default(base_url=base_url)
    
    def create_gemma(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> GemmaEmbedder:
        """
        Factory method cho Ollama Embedding Gemma embedder.
        
        Note: Config nằm trong GemmaEmbedder class (MODEL_ID, DIMENSION, MAX_TOKENS).

        Args:
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            GemmaEmbedder: Ollama Embedding Gemma embedder
        """
        return GemmaEmbedder.create_default(base_url=base_url)

    def create_huggingface_api(
        self,
        model_name: Optional[str] = None,
        api_token: Optional[str] = None,
        **kwargs: Any
    ) -> HuggingFaceApiEmbedder:
        """
        Factory method cho HuggingFace API embedder.

        Args:
            model_name: HF model name (default: BAAI/bge-small-en-v1.5)
            api_token: HF API token (auto-loaded if None)
            **kwargs: Additional arguments

        Returns:
            HuggingFaceApiEmbedder: HuggingFace API embedder
        """
        return HuggingFaceApiEmbedder.create_default(
            api_token=api_token,
            **kwargs
        )
    
    def create_huggingface_local(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        **kwargs: Any
    ) -> HuggingFaceLocalEmbedder:
        """
        Factory method cho HuggingFace Local embedder.

        Args:
            model_name: HF model name (default: BAAI/bge-small-en-v1.5)
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            HuggingFaceLocalEmbedder: HuggingFace Local embedder
        """
        return HuggingFaceLocalEmbedder.create_default(
            device=device,
            **kwargs
        )