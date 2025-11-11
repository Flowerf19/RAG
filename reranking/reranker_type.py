"""
Reranker Type Enumeration
==========================
Định nghĩa các loại reranker có sẵn
"""

from enum import Enum


class RerankerType(Enum):
    """Enum for available reranker types"""
    BGE_M3_OLLAMA = "bge_m3_ollama"  # BGE-M3 via Ollama
    BGE_M3_HF_API = "bge_m3_hf_api"  # BGE-M3 via HuggingFace API
    BGE_M3_HF_LOCAL = "bge_m3_hf_local"  # BGE-M3 via HuggingFace Local (v2-m3)
    JINA_V2_MULTILINGUAL = "jina_v2_multilingual"  # Jina V2 Multilingual Local
    GTE_MULTILINGUAL = "gte_multilingual"  # GTE Multilingual Local
    BGE_BASE = "bge_base"  # BGE Base Local
