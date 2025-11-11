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
from .providers.huggingface.multilingual_embedders import (
    E5LargeInstructEmbedder,
    E5BaseEmbedder,
    GTEMultilingualBaseEmbedder,
    ParaphraseMPNetBaseV2Embedder,
    ParaphraseMiniLML12V2Embedder,
)


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
            model_name: HF model name (default: intfloat/multilingual-e5-large - 1024 dim)
            api_token: HF API token (auto-loaded if None)
            **kwargs: Additional arguments

        Returns:
            HuggingFaceApiEmbedder: HuggingFace API embedder (E5-Large Multilingual 1024-dim, FREE)
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
            model_name: HF model name (default: BAAI/bge-m3 - 1024 dim)
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            HuggingFaceLocalEmbedder: HuggingFace Local embedder (BGE-M3 1024-dim)
        """
        return HuggingFaceLocalEmbedder.create_default(
            device=device,
            **kwargs
        )
    
    def create_gemma_hf_local(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        token: Optional[str] = None,
        **kwargs: Any
    ) -> GemmaEmbedder:
        """
        Factory method cho Gemma HuggingFace Local embedder.

        Args:
            model_name: HF Gemma model name (default: google/embeddinggemma-300m - 768 dim)
            device: Device for local inference
            token: HuggingFace token for accessing gated Gemma models
            **kwargs: Additional arguments

        Returns:
            GemmaEmbedder: Gemma HF Local embedder (768-dim)
        """
        return GemmaEmbedder.create_default(
            device=device,
            token=token,
            **kwargs
        )

    def create_e5_large_instruct(
        self,
        device: str = "cpu",
        **kwargs: Any
    ) -> E5LargeInstructEmbedder:
        """
        Factory method cho E5 Large Instruct embedder.

        Model: intfloat/multilingual-e5-large-instruct
        Đặc điểm: Instruct-based, retrieval/similarity, context 512-2048 tokens
        Ngôn ngữ: >100
        Kích thước: ~1.3GB (560M params)
        Lý do: Score cao tương tự BGE-M3 trên MTEB multilingual, tốt cho RAG, nhanh local

        Args:
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            E5LargeInstructEmbedder: E5 Large Instruct embedder (1024-dim)
        """
        return E5LargeInstructEmbedder(
            device=device,
            **kwargs
        )

    def create_e5_base(
        self,
        device: str = "cpu",
        **kwargs: Any
    ) -> E5BaseEmbedder:
        """
        Factory method cho E5 Base embedder.

        Model: intfloat/multilingual-e5-base
        Đặc điểm: Contrastive learning cho similarity/retrieval, nhẹ hơn large
        Ngôn ngữ: >100
        Kích thước: ~0.3GB (278M params)
        Lý do: Performance ổn định multilingual, dễ so sánh tốc độ với BGE-M3

        Args:
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            E5BaseEmbedder: E5 Base embedder (768-dim)
        """
        return E5BaseEmbedder(
            device=device,
            **kwargs
        )

    def create_gte_multilingual_base(
        self,
        device: str = "cpu",
        **kwargs: Any
    ) -> GTEMultilingualBaseEmbedder:
        """
        Factory method cho GTE Multilingual Base embedder.

        Model: Alibaba-NLP/gte-multilingual-base
        Đặc điểm: Dense embedding, multi-task, context 8192 tokens
        Ngôn ngữ: >70
        Kích thước: ~0.5GB (278M params)
        Lý do: Hiệu suất gần BGE-M3 trên MIRACL, nhẹ và tối ưu local

        Args:
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            GTEMultilingualBaseEmbedder: GTE Multilingual Base embedder (768-dim)
        """
        return GTEMultilingualBaseEmbedder(
            device=device,
            **kwargs
        )

    def create_paraphrase_mpnet_base_v2(
        self,
        device: str = "cpu",
        **kwargs: Any
    ) -> ParaphraseMPNetBaseV2Embedder:
        """
        Factory method cho Paraphrase MPNet Base V2 embedder.

        Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        Đặc điểm: Semantic search/paraphrase, nhanh inference
        Ngôn ngữ: >50
        Kích thước: ~0.5GB (278M params)
        Lý do: Classic, performance multilingual ổn, tốt cho so sánh speed

        Args:
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            ParaphraseMPNetBaseV2Embedder: Paraphrase MPNet Base V2 embedder (768-dim)
        """
        return ParaphraseMPNetBaseV2Embedder(
            device=device,
            **kwargs
        )

    def create_paraphrase_minilm_l12_v2(
        self,
        device: str = "cpu",
        **kwargs: Any
    ) -> ParaphraseMiniLML12V2Embedder:
        """
        Factory method cho Paraphrase MiniLM L12 V2 embedder.

        Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        Đặc điểm: Paraphrase detection, rất nhẹ
        Ngôn ngữ: >50
        Kích thước: ~0.2GB (118M params)
        Lý do: Lightweight baseline, tương đương BGE-M3 ở tasks cơ bản multilingual

        Args:
            device: Device for local inference
            **kwargs: Additional arguments

        Returns:
            ParaphraseMiniLML12V2Embedder: Paraphrase MiniLM L12 V2 embedder (384-dim)
        """
        return ParaphraseMiniLML12V2Embedder(
            device=device,
            **kwargs
        )