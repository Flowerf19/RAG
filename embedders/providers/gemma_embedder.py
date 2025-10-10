"""
Gemma Embedder
==============
Concrete embedder for Gemma models (local or API-backed).
"""

from __future__ import annotations

from typing import Optional

from ..i_embedder import IEmbedder
from .hf_embedder import HFEmbedder


class GemmaEmbedder(HFEmbedder):
    """
    Gemma embedding provider leveraging HFEmbedder base implementation.
    Single Responsibility: supply Gemma-specific defaults and overrides.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        max_tokens: int = 8192,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            device=device,
            normalize=normalize,
        )
