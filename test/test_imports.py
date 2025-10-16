"""
Test Import Validation
======================
Kiểm tra tất cả các import quan trọng trong hệ thống RAG.
"""

import pytest
import sys
import os
from pathlib import Path


class TestImportValidation:
    """Test class để kiểm tra import của tất cả modules chính"""

    def test_project_root_in_path(self):
        """Kiểm tra project root có trong sys.path"""
        project_root = Path(__file__).parent.parent
        assert str(project_root) in sys.path, f"Project root {project_root} not in sys.path"

    def test_rag_system_package_import(self):
        """Test import package RAG_system"""
        try:
            import RAG_system
            assert RAG_system is not None
        except ImportError as e:
            pytest.fail(f"Cannot import RAG_system package: {e}")

    def test_loader_module_imports(self):
        """Test import các module trong LOADER"""
        try:
            from RAG_system.LOADER.pdf_loader import PDFLoader
            from RAG_system.LOADER.model.document import PDFDocument
            from RAG_system.LOADER.model.page import PDFPage
            from RAG_system.LOADER.model.block import Block, TableBlock
            from RAG_system.LOADER.model.table import TableSchema

            # Test factory methods
            loader = PDFLoader.create_default()
            assert isinstance(loader, PDFLoader)

        except ImportError as e:
            pytest.fail(f"Cannot import LOADER modules: {e}")

    def test_chunker_module_imports(self):
        """Test import các module trong CHUNKERS"""
        try:
            from RAG_system.CHUNKERS.hybrid_chunker import HybridChunker
            from RAG_system.CHUNKERS.fixed_size_chunker import FixedSizeChunker
            from RAG_system.CHUNKERS.semantic_chunker import SemanticChunker
            from RAG_system.CHUNKERS.rule_based_chunker import RuleBasedChunker
            from RAG_system.CHUNKERS.model.chunk import Chunk
            from RAG_system.CHUNKERS.model.chunk_set import ChunkSet

            # Test instantiation
            chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
            assert chunker is not None

        except ImportError as e:
            pytest.fail(f"Cannot import CHUNKERS modules: {e}")

    def test_embedder_module_imports(self):
        """Test import các module trong EMBEDDERS"""
        try:
            from RAG_system.EMBEDDERS.embedder_factory import EmbedderFactory
            from RAG_system.EMBEDDERS.i_embedder import IEmbedder
            from RAG_system.EMBEDDERS.embedder_type import EmbedderType
            from RAG_system.EMBEDDERS.providers.ollama.model_switcher import OllamaModelSwitcher
            from RAG_system.EMBEDDERS.providers.ollama import OllamaModelType

            # Test factory instantiation
            factory = EmbedderFactory()
            assert factory is not None

            # Test model switcher
            switcher = OllamaModelSwitcher()
            assert switcher is not None

        except ImportError as e:
            pytest.fail(f"Cannot import EMBEDDERS modules: {e}")

    def test_pipeline_module_imports(self):
        """Test import các module trong pipeline"""
        try:
            from RAG_system.pipeline.rag_pipeline import RAGPipeline
            from RAG_system.pipeline.vector_store import VectorStore
            from RAG_system.pipeline.summary_generator import SummaryGenerator
            from RAG_system.pipeline.retriever import Retriever

            # Test pipeline instantiation (without actual processing)
            # Note: This might fail if Ollama is not running, so we just test import
            assert RAGPipeline is not None

        except ImportError as e:
            pytest.fail(f"Cannot import pipeline modules: {e}")

    def test_llm_module_imports(self):
        """Test import LLM module"""
        try:
            from RAG_system.LLM.llm_client import LLMClient
            assert LLMClient is not None
        except ImportError as e:
            pytest.fail(f"Cannot import LLM modules: {e}")

    def test_retrieval_module_imports(self):
        """Test import RETRIEVAL modules"""
        try:
            from RAG_system.RETRIEVAL.VECTOR.vector_store import VectorStore as RetrievalVectorStore
            assert RetrievalVectorStore is not None
        except ImportError as e:
            pytest.fail(f"Cannot import RETRIEVAL modules: {e}")

    def test_factory_method_patterns(self):
        """Test các factory method patterns"""
        try:
            from RAG_system.LOADER.pdf_loader import PDFLoader
            from RAG_system.EMBEDDERS.providers.ollama.model_switcher import OllamaModelSwitcher

            # Test PDFLoader factory methods
            default_loader = PDFLoader.create_default()
            text_only_loader = PDFLoader.create_text_only()
            tables_only_loader = PDFLoader.create_tables_only()

            assert isinstance(default_loader, PDFLoader)
            assert isinstance(text_only_loader, PDFLoader)
            assert isinstance(tables_only_loader, PDFLoader)

            # Test embedder switching
            switcher = OllamaModelSwitcher()
            gemma_embedder = switcher.switch_to_gemma()
            bge_embedder = switcher.switch_to_bge_m3()

            assert gemma_embedder is not None
            assert bge_embedder is not None

        except ImportError as e:
            pytest.fail(f"Factory method patterns failed: {e}")

    def test_model_classes_import(self):
        """Test import các model classes"""
        try:
            from RAG_system.LOADER.model.base import LoaderBaseModel
            from RAG_system.LOADER.model.text import Text
            from RAG_system.CHUNKERS.base_chunker import BaseChunker
            from RAG_system.EMBEDDERS.model.embedding_profile import EmbeddingProfile

            assert LoaderBaseModel is not None
            assert Text is not None
            assert BaseChunker is not None
            assert EmbeddingProfile is not None

        except ImportError as e:
            pytest.fail(f"Cannot import model classes: {e}")

    def test_normalizer_imports(self):
        """Test import normalizer modules"""
        try:
            from RAG_system.LOADER.normalizers.table_utils import clean_table
            from RAG_system.LOADER.normalizers.text_utils import normalize_whitespace
            from RAG_system.LOADER.normalizers.block_utils import merge_blocks_list

            assert clean_table is not None
            assert normalize_whitespace is not None
            assert merge_blocks_list is not None

        except ImportError as e:
            pytest.fail(f"Cannot import normalizer modules: {e}")

    def test_rerank_module_import(self):
        """Test import RERANK module (if exists)"""
        try:
            import RAG_system.RERANK
            assert RAG_system.RERANK is not None
        except ImportError:
            # RERANK module might not be implemented yet, this is OK
            pytest.skip("RERANK module not implemented yet")

    def test_circular_import_check(self):
        """Test kiểm tra circular imports bằng cách import tất cả cùng lúc"""
        try:
            # Import all major modules at once to check for circular dependencies
            import RAG_system.LOADER.pdf_loader
            import RAG_system.CHUNKERS.hybrid_chunker
            import RAG_system.EMBEDDERS.embedder_factory
            import RAG_system.pipeline.rag_pipeline

            # If we get here without ImportError, no circular imports detected
            assert True

        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_version_compatibility(self):
        """Test version compatibility của các thư viện chính"""
        try:
            import fitz  # PyMuPDF
            import pdfplumber
            import numpy as np
            import faiss

            # Basic functionality tests
            assert fitz is not None
            assert pdfplumber is not None
            assert np is not None
            assert faiss is not None

        except ImportError as e:
            pytest.fail(f"Version compatibility issue: {e}")