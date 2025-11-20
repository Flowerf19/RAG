"""
Query Processor Module
======================
Handles query enhancement and embedding generation for retrieval.
Single Responsibility: Query preprocessing and embedding fusion.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes queries through enhancement and embedding generation.
    """

    def __init__(self, qem_module=None, embedder=None):
        """
        Initialize query processor.
        
        Args:
            qem_module: QueryEnhancementModule instance (optional)
            embedder: Embedder instance (optional)
        """
        self.qem = qem_module
        self.embedder = embedder

    def enhance_query(self, query_text: str, use_enhancement: bool = True) -> List[str]:
        """
        Enhance query using QEM if available and enabled.
        
        Args:
            query_text: Original query text
            use_enhancement: Whether to use query enhancement
            
        Returns:
            List of query variants (including original)
        """
        if not use_enhancement or not self.qem:
            return [query_text]

        try:
            logger.info("Enhancing query...")
            expanded_queries = self.qem.enhance(query_text)
            expanded_queries = [q for q in expanded_queries if q and q.strip()]
            
            if not expanded_queries:
                expanded_queries = [query_text]
                
            logger.info(f"Query enhanced: {len(expanded_queries)} queries generated")
            return expanded_queries
            
        except Exception as exc:
            logger.warning("Query enhancement failed, using original query: %s", exc)
            return [query_text]

    def fuse_query_embeddings(self, queries: List[str]) -> Optional[List[float]]:
        """
        Generate embeddings for each query and return their mean vector.
        
        Args:
            queries: List of query strings
            
        Returns:
            Fused embedding vector or None if failed
        """
        if not self.embedder:
            return None

        vectors: List[np.ndarray] = []
        
        for query in queries:
            try:
                embedding = self.embedder.embed(query)
                if embedding:
                    vectors.append(np.asarray(embedding, dtype=np.float32))
            except Exception as exc:
                logger.warning("Failed to embed query '%s': %s", query, exc)

        if not vectors:
            return None

        stacked = np.stack(vectors, axis=0)
        mean_vector = stacked.mean(axis=0)
        
        return mean_vector.astype(np.float32).tolist()


def create_query_processor(
    use_query_enhancement: bool = True,
    embedder=None,
) -> QueryProcessor:
    """
    Factory function to create QueryProcessor with QEM if enabled.
    
    Args:
        use_query_enhancement: Whether to enable query enhancement
        embedder: Embedder instance
        
    Returns:
        Configured QueryProcessor instance
    """
    qem_module = None
    
    if use_query_enhancement:
        try:
            from query_enhancement import (
                QueryEnhancementModule,
                load_qem_settings,
            )
            from llm.config_loader import get_config

            logger.info("Initializing Query Enhancement Module...")
            app_config = get_config()
            qem_settings = load_qem_settings()
            qem_module = QueryEnhancementModule(app_config, qem_settings, logger=logger)
            
        except Exception as exc:
            logger.warning("Failed to initialize QEM: %s", exc)

    return QueryProcessor(qem_module, embedder)
