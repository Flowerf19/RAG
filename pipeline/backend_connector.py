"""
RAG Retrieval Backend Connector
================================
Backward compatibility layer - imports from refactored modules.
Main functionality moved to specialized modules:
- retrieval_service.py: RAGRetrievalService class
- query_processor.py: Query enhancement and embedding
- score_fusion.py: Score normalization and merging
- retrieval_orchestrator.py: fetch_retrieval() function

Usage:
    from pipeline.backend_connector import RAGRetrievalService, fetch_retrieval
    
    # Or use new modules directly:
    from pipeline.retrieval_service import RAGRetrievalService
    from pipeline.retrieval_orchestrator import fetch_retrieval
"""

from __future__ import annotations

# Re-export main classes and functions for backward compatibility
from pipeline.retrieval.retrieval_service import RAGRetrievalService
from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval
from query_enhancement.query_processor import QueryProcessor, create_query_processor
from pipeline.retrieval.score_fusion import ScoreFusion

__all__ = [
    "RAGRetrievalService",
    "fetch_retrieval",
    "QueryProcessor",
    "create_query_processor",
    "ScoreFusion",
]
