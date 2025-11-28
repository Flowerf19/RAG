"""
Retriever - Vector Similarity Search
====================================
Handles similarity search against FAISS vector indexes using cosine similarity.
Single Responsibility: Vector similarity search and result formatting.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import faiss

from embedders.i_embedder import IEmbedder

logger = logging.getLogger(__name__)


class Retriever:
    """
    Performs similarity search against FAISS vector indexes.
    Single Responsibility: Vector search operations and result formatting.
    """

    def __init__(self, embedder: IEmbedder):
        """
        Initialize Retriever.

        Args:
            embedder: Embedder instance for query encoding
        """
        self.embedder = embedder
        # Cache for loaded FAISS indexes to avoid reloading on every query
        self._faiss_cache: Dict[str, tuple[faiss.Index, Dict[int, Dict[str, Any]]]] = {}

    def clear_cache(self) -> None:
        """
        Clear the FAISS index cache. Call this when indexes are rebuilt.
        """
        self._faiss_cache.clear()
        logger.info("Cleared FAISS index cache")

    def search_similar(
        self,
        faiss_file: Path,
        metadata_map_file: Path,
        query_text: str | None,
        top_k: int = 10,
        *,
        query_embedding: List[float] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity with FAISS.

        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search (optional when query_embedding provided)
            top_k: Number of results to return
            query_embedding: Optional precomputed embedding to use instead of encoding

        Returns:
            List of similar chunks with metadata and cosine similarity scores
        """
        # Load FAISS index and metadata (with caching)
        cache_key = f"{faiss_file}:{metadata_map_file}"
        if cache_key not in self._faiss_cache:
            index, metadata_map = self._load_index_and_metadata(faiss_file, metadata_map_file)
            self._faiss_cache[cache_key] = (index, metadata_map)
            logger.debug("Loaded FAISS index from disk: %s", faiss_file)
        else:
            index, metadata_map = self._faiss_cache[cache_key]
            logger.debug("Using cached FAISS index: %s", faiss_file)

        # Generate or reuse query embedding and normalize
        if query_embedding is None:
            if not query_text:
                raise ValueError("Either query_text or query_embedding must be provided.")
            query_embedding = self.embedder.embed(query_text)

        query_vector = np.array([query_embedding], dtype='float32')
        query_normalized = self._normalize_vectors(query_vector)

        # Perform search using inner product (cosine similarity for normalized vectors)
        similarities, indices = index.search(query_normalized, top_k)  # type: ignore[call-arg]

        # Format results (similarities are cosine similarities for normalized vectors)
        # Filter out invalid indices (-1) which FAISS returns when no match found
        results = []
        invalid_count = 0
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx >= 0 and idx < len(metadata_map):  # Check for valid index
                result = metadata_map[idx].copy()
                result["similarity_score"] = float(similarity)
                results.append(result)
            elif idx == -1:
                # FAISS returns -1 when insufficient results (index too small)
                invalid_count += 1
            else:
                logger.warning(f"Invalid index {idx} returned by FAISS search (out of bounds)")

        if invalid_count > 0:
            logger.debug(f"Skipped {invalid_count} invalid indices (-1) from FAISS search results")

        # Sort results by similarity score (descending) to ensure correct ordering
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        log_query = query_text or "<embedding>"
        # Use DEBUG level here to avoid noisy INFO logs for each index when
        # multiple FAISS indexes are searched. The orchestrator will emit a
        # consolidated INFO-level summary instead.
        logger.debug("Similarity search completed: %d result(s) for query %s", len(results), log_query)
        return results

    def _load_index_and_metadata(self, faiss_file: Path, metadata_map_file: Path) -> tuple[faiss.Index, Dict[int, Dict[str, Any]]]:
        """
        Load FAISS index and metadata from disk.

        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file

        Returns:
            Tuple of (faiss_index, metadata_map)
        """
        index = faiss.read_index(str(faiss_file))

        import pickle
        with open(metadata_map_file, 'rb') as f:
            metadata_map = pickle.load(f)

        return index, metadata_map

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a single vector for cosine similarity.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize multiple vectors for cosine similarity.

        Args:
            vectors: Input vectors array

        Returns:
            Normalized vectors array
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
