"""
Embedding Processor Module
==========================
Handles embedding generation for chunks with progress tracking.
Single Responsibility: Convert text chunks to embedding vectors.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Processes chunks to generate embeddings with metadata and progress tracking.
    """

    def __init__(self, embedder, timestamp: str):
        """
        Initialize embedding processor.
        
        Args:
            embedder: Embedder instance (Ollama or HuggingFace)
            timestamp: Timestamp string for provenance
        """
        self.embedder = embedder
        self.timestamp = timestamp

    def _to_serializable(self, value: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable format.
        
        Args:
            value: Any value to convert
            
        Returns:
            JSON-serializable representation
        """
        if hasattr(value, "to_dict"):
            return self._to_serializable(value.to_dict())
        if isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_serializable(v) for v in value]
        return value

    def process_chunks(
        self,
        chunk_set: Any,
        pdf_path: Any,
        chunk_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Generate embeddings for all chunks in a ChunkSet.
        
        Args:
            chunk_set: ChunkSet object from chunker
            pdf_path: Path to source PDF file
            chunk_callback: Optional callback function(current, total) for progress
            
        Returns:
            Tuple of (embeddings_data list, skipped_chunks count)
        """
        embeddings_data = []
        skipped_chunks = 0
        total_chunks = len(chunk_set.chunks)
        seen_hashes = set()  # Track processed content hashes

        # Call callback with initial state
        if chunk_callback:
            chunk_callback(0, total_chunks)

        logger.info(f"Starting embedding generation for {total_chunks} chunks...")

        for idx, chunk in enumerate(chunk_set.chunks, 1):
            # Create content hash for duplicate checking
            content_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
            
            # Skip duplicate chunks
            if content_hash in seen_hashes:
                logger.debug(f"Skipping duplicate chunk {idx} (hash: {content_hash[:8]})")
                skipped_chunks += 1
                continue
                
            seen_hashes.add(content_hash)

            # Test connection on first chunk (warn but don't fail)
            if idx == 1:
                if not self.embedder.test_connection():
                    logger.warning("⚠️ Cannot connect to embedder server! Using zero vectors as fallback.")
                    self._embedder_available = False
                else:
                    self._embedder_available = True
                    logger.info("✅ Embedder connection successful")

            # Log progress for every chunk
            logger.info(
                f"📝 Embedding chunk {idx}/{total_chunks} ({(idx/total_chunks)*100:.1f}%) - {len(chunk.text)} chars"
            )

            # Generate embedding
            if hasattr(self, '_embedder_available') and not self._embedder_available:
                # Use zero vector fallback
                embedding = [0.0] * self.embedder.dimension
                logger.debug(f"Using zero vector for chunk {idx} (embedder unavailable)")
            else:
                try:
                    embedding = self.embedder.embed(chunk.text)
                except Exception as e:
                    logger.warning(f"Error embedding chunk {idx}: {e}")
                    embedding = [0.0] * self.embedder.dimension

            # Build chunk embedding data
            chunk_embedding = self._build_chunk_embedding(
                chunk, embedding, pdf_path, idx - 1, chunk_set
            )

            embeddings_data.append(chunk_embedding)

            # Update progress via callback
            if chunk_callback:
                chunk_callback(idx, total_chunks)

            if (idx - skipped_chunks) % 10 == 0:
                logger.info(
                    f"Processed {idx}/{total_chunks} chunks ({skipped_chunks} skipped)..."
                )

        logger.info(f"Generated {len(embeddings_data)} embeddings ({skipped_chunks} chunks skipped)")
        return embeddings_data, skipped_chunks

    def _build_chunk_embedding(
        self,
        chunk: Any,
        embedding: List[float],
        pdf_path: Any,
        chunk_index: int,
        chunk_set: Any,
    ) -> Dict[str, Any]:
        """
        Build embedding data dictionary with full metadata.
        
        Args:
            chunk: Chunk object
            embedding: Embedding vector
            pdf_path: Path to source PDF
            chunk_index: Index of chunk
            chunk_set: Parent ChunkSet
            
        Returns:
            Complete embedding data dictionary
        """
        chunk_embedding = {
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk_index,
            "text": chunk.text,
            "text_length": len(chunk.text),
            "token_count": chunk.token_count,
            # Embedding vector
            "embedding": embedding,
            "embedding_dimension": len(embedding),
            "embedding_model": self.embedder.profile.model_id,
            # Source metadata
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "page_number": (
                list(chunk.provenance.page_numbers)[0]
                if chunk.provenance and chunk.provenance.page_numbers
                else None
            ),
            "page_numbers": (
                sorted(list(chunk.provenance.page_numbers)) if chunk.provenance else []
            ),
            # Block tracing
            "block_type": chunk.metadata.get("block_type") or chunk.metadata.get("type"),
            "block_ids": chunk.provenance.source_blocks if chunk.provenance else [],
            # Table detection
            "is_table": chunk.metadata.get("block_type") == "table",
            # Provenance
            "provenance": {
                "source_file": str(pdf_path),
                "extraction_method": "PDFLoader",
                "chunking_strategy": chunk_set.chunk_strategy or "unknown",
                "embedding_model": self.embedder.profile.model_id,
                "timestamp": self.timestamp,
            },
        }

        # Add table data if applicable
        if chunk_embedding["is_table"]:
            table_payload = chunk.metadata.get("table_payload")
            if table_payload:
                chunk_embedding["table_data"] = {
                    "table_id": getattr(table_payload, "id", None),
                    "header": getattr(table_payload, "header", []),
                    "num_rows": len(getattr(table_payload, "rows", [])),
                    "page_number": getattr(table_payload, "page_number", None),
                }

        # Attach full metadata
        metadata_serialized = self._to_serializable(chunk.metadata or {})
        chunk_embedding["metadata"] = metadata_serialized

        source_blocks_meta_raw = (
            chunk.metadata.get("source_blocks", []) if chunk.metadata else []
        )
        source_blocks_meta = [self._to_serializable(meta) for meta in source_blocks_meta_raw]
        chunk_embedding["source_blocks"] = source_blocks_meta

        # Handle figure metadata
        is_figure = False
        figure_meta = None

        if chunk.metadata:
            block_type = chunk.metadata.get("block_type")
            if block_type == "figure":
                figure_info = chunk.metadata.get("figure")
                if figure_info:
                    is_figure = True
                    figure_meta = self._to_serializable(figure_info)

        if not is_figure and source_blocks_meta:
            for block_meta in source_blocks_meta:
                if (block_meta or {}).get("block_type") == "figure":
                    is_figure = True
                    figure_meta = block_meta
                    break

        chunk_embedding["is_figure"] = is_figure
        if figure_meta:
            chunk_embedding["figure"] = {
                "caption": figure_meta.get("caption"),
                "image_path": figure_meta.get("image_path"),
                "bbox": figure_meta.get("bbox"),
                "page_number": figure_meta.get("page_number"),
                "figure_order": figure_meta.get("figure_order"),
            }

        return chunk_embedding
