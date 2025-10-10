"""
Embed Request Model
===================
Normalization-ready request payload for embedding providers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from loaders.model.base import LoaderBaseModel


@dataclass
class EmbedRequest(LoaderBaseModel):
    """
    Dataclass representing normalized embedding inputs.
    Single Responsibility: carry all information about the text chunk to embed.
    """

    text: str
    chunk_id: str
    doc_id: str
    lang: Optional[str] = None
    is_table: bool = False
    tokens_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None
    section_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def validate(self) -> bool:
        """Basic validation to ensure request is ready for embedding."""
        if not self.text or not self.text.strip():
            return False
        if not self.chunk_id or not self.doc_id:
            return False
        return True
