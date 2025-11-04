"""
Chunker Providers - Preprocessing before embedding
==================================================
Provider cho SemanticChunker để xử lý text trước khi embedding:
- Text normalization
- Language detection
- Entity extraction with spaCy
- Metadata enrichment
"""

from .semantic_provider import SemanticChunkerProvider, create_semantic_provider

__all__ = [
    'SemanticChunkerProvider',
    'create_semantic_provider',
]
