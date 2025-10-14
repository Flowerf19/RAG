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

    def search_similar(self, faiss_file: Path, metadata_map_file: Path,
                      query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity with FAISS.

        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search
            top_k: Number of results to return

        Returns:
            List of similar chunks with metadata and cosine similarity scores
        """
        # Load FAISS index and metadata
        index, metadata_map = self._load_index_and_metadata(faiss_file, metadata_map_file)

        # Generate query embedding and normalize
        query_embedding = self.embedder.embed(query_text)
        query_vector = np.array([query_embedding], dtype='float32')
        query_normalized = self._normalize_vectors(query_vector)

        # Perform search using inner product (cosine similarity for normalized vectors)
        similarities, indices = index.search(query_normalized, top_k)  # type: ignore[call-arg]

        # Format results (similarities are already cosine similarities for normalized vectors)
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(metadata_map):
                result = metadata_map[idx].copy()
                result["cosine_similarity"] = float(similarity)
                result["distance"] = 1.0 - float(similarity)  # Convert to distance for compatibility
                result["similarity_score"] = float(similarity)  # Cosine similarity as score
                results.append(result)

        logger.info(f"Cosine similarity search completed: found {len(results)} results for query")
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