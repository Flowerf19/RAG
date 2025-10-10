"""
BGE3 Embedder
=============
Concrete embedder for BGE3 models.
"""

from __future__ import annotations

from typing import Optional

from .hf_embedder import HFEmbedder


class BGE3Embedder(HFEmbedder):
    """
    BGE3 embedding provider wrapping HFEmbedder.
    Single Responsibility: expose BGE3 defaults.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_tokens: int = 4096,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            device=device,
            normalize=normalize,
        )
