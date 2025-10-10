"""
Embedding Result Model
======================
Value object produced after embedding execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loaders.model.base import LoaderBaseModel


@dataclass
class EmbeddingResult(LoaderBaseModel):
    """
    Output of running embeddings on a text chunk.
    Single Responsibility: capture embedding vector and provenance metadata.
    """

    chunk_id: str
    embedding: List[float]
    text_embedded: str
    token_count: int
    model_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    lang: Optional[str] = None

    def validate(self) -> bool:
        """Ensure the result contains the required data."""
        return (
            bool(self.chunk_id)
            and isinstance(self.embedding, list)
            and len(self.embedding) > 0
            and bool(self.model_name)
        )
