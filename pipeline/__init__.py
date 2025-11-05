"""
RAG Pipeline Package
====================
Modular RAG pipeline với composition architecture.

Components:
- RAGPipeline: Main orchestrator
- VectorStore: FAISS index management
- SummaryGenerator: Document/batch summaries
- Retriever: Vector similarity search

Submodules:
- processing: PDF and embedding processing
- storage: File I/O and vector storage
- retrieval: Hybrid retrieval and search
"""

from .rag_pipeline import RAGPipeline
from .storage.vector_store import VectorStore
from .storage.summary_generator import SummaryGenerator
from .retrieval.retriever import Retriever

__all__ = [
    "RAGPipeline",
    "VectorStore", 
    "SummaryGenerator",
    "Retriever"
]