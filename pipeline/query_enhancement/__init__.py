"""
Query Enhancement Module package.

Provides functionality to expand user queries prior to retrieval in order
to improve recall across FAISS and BM25 backends.
"""

from .qem_core import QueryEnhancementModule, load_qem_settings

__all__ = ["QueryEnhancementModule", "load_qem_settings"]
