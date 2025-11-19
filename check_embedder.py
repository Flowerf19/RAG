#!/usr/bin/env python3
"""
Check pipeline embedder
"""

from pipeline.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
print(f'Pipeline embedder type: {type(pipeline.embedder)}')
print(f'Embedder model: {getattr(pipeline.embedder, "model_name", "unknown")}')
print(f'Embedder: {pipeline.embedder}')