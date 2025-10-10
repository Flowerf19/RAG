"""
ChunkSet Embedder
=================
Transform normalized ChunkSet objects into embedding vectors.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from chunkers.model.chunk import Chunk
from chunkers.model.chunk_set import ChunkSet
from loaders.model.base import LoaderBaseModel

from .embedding_profile import EmbeddingProfile
from .i_embedder import IEmbedder
from .model.embed_request import EmbedRequest
from .model.embedding_result import EmbeddingResult


class ChunkSetEmbedder:
    """
    Coordinates embedding generation for a ChunkSet.
    Single Responsibility: build embedding requests and invoke the embedder.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        profile: Optional[EmbeddingProfile] = None,
        include_titles: bool = False,
        include_table_metadata: bool = True,
        include_captions: bool = True,
    ):
        self.embedder = embedder
        self.profile = profile
        self.include_titles = include_titles
        self.include_table_metadata = include_table_metadata
        self.include_captions = include_captions

    def embed_chunk_set(self, chunk_set: ChunkSet) -> List[EmbeddingResult]:
        """Embed all chunks (and optional extras) in the chunk set."""
        requests = self._build_requests(chunk_set)
        return self.embedder.embed_batch_req(requests)

    def _build_requests(self, chunk_set: ChunkSet) -> List[EmbedRequest]:
        requests: List[EmbedRequest] = []
        for chunk in chunk_set.chunks:
            requests.append(self._chunk_request(chunk, chunk_set))
            requests.extend(self._additional_requests(chunk, chunk_set))
        return requests

    def _chunk_request(self, chunk: Chunk, chunk_set: ChunkSet) -> EmbedRequest:
        metadata = self._base_metadata(chunk, chunk_set)
        metadata["content_role"] = "chunk"
        lang = chunk.metadata.get("lang") or chunk.metadata.get("language")
        is_table = self._is_table_chunk(chunk)
        return EmbedRequest(
            text=chunk.textForEmbedding,
            chunk_id=chunk.chunk_id,
            doc_id=chunk_set.doc_id,
            lang=lang,
            is_table=is_table,
            tokens_estimate=chunk.token_count,
            metadata=metadata,
            title=chunk.section_title,
            section_path=chunk.metadata.get("section_path"),
        )

    def _additional_requests(self, chunk: Chunk, chunk_set: ChunkSet) -> Iterable[EmbedRequest]:
        """Optionally embed titles or table metadata."""
        metadata = self._base_metadata(chunk, chunk_set)
        requests: List[EmbedRequest] = []
        if self.include_titles and chunk.section_title:
            title_meta = metadata.copy()
            title_meta["content_role"] = "title"
            requests.append(
                EmbedRequest(
                    text=chunk.section_title.strip(),
                    chunk_id=f"{chunk.chunk_id}::title",
                    doc_id=chunk_set.doc_id,
                    lang=chunk.metadata.get("lang") or chunk.metadata.get("language"),
                    is_table=False,
                    tokens_estimate=self._estimate_tokens(chunk.section_title),
                    metadata=title_meta,
                    title=chunk.section_title,
                    section_path=chunk.metadata.get("section_path"),
                )
            )

        if self._is_table_chunk(chunk):
            if self.include_table_metadata:
                table_title = chunk.metadata.get("table_title")
                if table_title:
                    table_title_meta = metadata.copy()
                    table_title_meta["content_role"] = "table_title"
                    requests.append(
                        EmbedRequest(
                            text=table_title.strip(),
                            chunk_id=f"{chunk.chunk_id}::table_title",
                            doc_id=chunk_set.doc_id,
                            lang=chunk.metadata.get("lang") or chunk.metadata.get("language"),
                            is_table=True,
                            tokens_estimate=self._estimate_tokens(table_title),
                            metadata=table_title_meta,
                            title=table_title,
                            section_path=chunk.metadata.get("section_path"),
                        )
                    )
            if self.include_captions:
                table_caption = chunk.metadata.get("table_caption")
                if table_caption:
                    caption_meta = metadata.copy()
                    caption_meta["content_role"] = "table_caption"
                    requests.append(
                        EmbedRequest(
                            text=table_caption.strip(),
                            chunk_id=f"{chunk.chunk_id}::table_caption",
                            doc_id=chunk_set.doc_id,
                            lang=chunk.metadata.get("lang") or chunk.metadata.get("language"),
                            is_table=True,
                            tokens_estimate=self._estimate_tokens(table_caption),
                            metadata=caption_meta,
                            title=chunk.metadata.get("table_title"),
                            section_path=chunk.metadata.get("section_path"),
                        )
                    )
        return requests

    def _base_metadata(self, chunk: Chunk, chunk_set: ChunkSet) -> Dict[str, Optional[object]]:
        """Construct metadata payload preserving provenance."""
        provenance_payload = None
        if chunk.provenance:
            provenance = chunk.provenance
            provenance_payload = {
                "doc_id": provenance.doc_id or chunk_set.doc_id,
                "file_path": provenance.file_path or chunk_set.file_path,
                "page_numbers": sorted(list(provenance.page_numbers)),
                "source_blocks": list(provenance.source_blocks),
                "spans": [span.to_dict() for span in provenance.spans],
                "metadata": self._sanitize_metadata(provenance.metadata),
                "group_type": chunk.metadata.get("group_type"),
            }
        chunk_metadata = self._sanitize_metadata(chunk.metadata)
        return {
            "doc_id": chunk_set.doc_id,
            "file_path": chunk_set.file_path,
            "chunk_type": chunk.chunk_type.value,
            "strategy": chunk.strategy.value if chunk.strategy else None,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "chunk_metadata": chunk_metadata,
            "chunk_hash": chunk_metadata.get("hash")
            or chunk_metadata.get("content_hash")
            or chunk_metadata.get("md5"),
            "provenance": provenance_payload,
            "profile": self.profile.model_id if self.profile else None,
            "profile_settings": self._profile_payload(),
        }

    @staticmethod
    def _is_table_chunk(chunk: Chunk) -> bool:
        metadata = chunk.metadata or {}
        group_type = metadata.get("group_type") or metadata.get("block_type")
        if group_type:
            return group_type == "table"
        if "table_payload" in metadata:
            return True
        return False

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rudimentary heuristic for token estimation."""
        cleaned = text.strip()
        if not cleaned:
            return 0
        return max(1, len(cleaned) // 4)

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert LoaderBaseModel instances to dicts to keep metadata JSON-serializable."""
        if not metadata:
            return {}

        def convert(value: Any) -> Any:
            if isinstance(value, LoaderBaseModel):
                return value.to_dict()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [convert(v) for v in value]
            return value

        return {key: convert(val) for key, val in metadata.items()}

    def _profile_payload(self) -> Optional[Dict[str, Any]]:
        if not self.profile:
            return None
        return {
            "model_id": self.profile.model_id,
            "dimension": self.profile.dimension,
            "max_tokens": self.profile.max_tokens,
            "normalize": self.profile.normalize,
            "pooling": self.profile.pooling,
            "endpoint": self.profile.endpoint,
        }
