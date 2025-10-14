import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import faiss

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.rag_pipeline import RAGPipeline
from pipeline.vector_store import VectorStore
from pipeline.retriever import Retriever
from pipeline.summary_generator import SummaryGenerator
from embedders.providers.ollama import OllamaModelType


class TestVectorStore:
    """Test cơ bản cho VectorStore"""

    def test_initialization(self):
        """Test constructor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))
            assert store.vectors_dir == Path(tmp_dir)
            assert store.vectors_dir.exists()

    def test_create_index_basic(self):
        """Test create_index với data cơ bản"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))

            # Mock embeddings data
            embeddings_data = [
                {
                    'chunk_id': 'chunk_1',
                    'text': 'test text 1',
                    'embedding': np.random.rand(768).tolist(),
                    'embedding_dimension': 768,
                    'text_length': 10,
                    'file_name': 'test.pdf',
                    'file_path': '/test/test.pdf',
                    'page_number': 1,
                    'page_numbers': [1],
                    'chunk_index': 0,
                    'block_type': 'text',
                    'block_ids': [],
                    'is_table': False,
                    'token_count': 5
                },
                {
                    'chunk_id': 'chunk_2',
                    'text': 'test text 2',
                    'embedding': np.random.rand(768).tolist(),
                    'embedding_dimension': 768,
                    'text_length': 10,
                    'file_name': 'test.pdf',
                    'file_path': '/test/test.pdf',
                    'page_number': 2,
                    'page_numbers': [2],
                    'chunk_index': 1,
                    'block_type': 'text',
                    'block_ids': [],
                    'is_table': False,
                    'token_count': 5
                }
            ]

            # Create index
            faiss_file, metadata_file = store.create_index(embeddings_data, "test_doc", "20241201_120000")

            # Verify files exist
            assert faiss_file.exists()
            assert metadata_file.exists()

    def test_load_index(self):
        """Test load_index method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))

            # First create an index
            embeddings_data = [
                {
                    'chunk_id': 'chunk_1',
                    'text': 'test text',
                    'embedding': np.random.rand(768).tolist(),
                    'embedding_dimension': 768,
                    'text_length': 9,
                    'file_name': 'test.pdf',
                    'file_path': '/test/test.pdf',
                    'page_number': 1,
                    'page_numbers': [1],
                    'chunk_index': 0,
                    'block_type': 'text',
                    'block_ids': [],
                    'is_table': False,
                    'token_count': 5
                }
            ]

            faiss_file, metadata_file = store.create_index(embeddings_data, "test_doc", "20241201_120000")

            # Load the index
            loaded_index, loaded_metadata = store.load_index(faiss_file, metadata_file)

            # Verify loaded data
            assert isinstance(loaded_index, faiss.Index)
            assert isinstance(loaded_metadata, dict)
            assert len(loaded_metadata) > 0


class TestRetriever:
    """Test cơ bản cho Retriever"""

    def test_initialization(self):
        """Test constructor"""
        mock_embedder = Mock()
        retriever = Retriever(mock_embedder)
        assert retriever.embedder == mock_embedder

    @patch.object(Retriever, '_load_index_and_metadata')
    def test_search_similar(self, mock_load_method):
        """Test search_similar method"""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.random.rand(768)

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # similarities
            np.array([[0, 1]])      # indices
        )
        mock_metadata = {0: {'text': 'result 1', 'page_number': 1}, 1: {'text': 'result 2', 'page_number': 2}}
        mock_load_method.return_value = (mock_index, mock_metadata)

        retriever = Retriever(mock_embedder)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy files
            faiss_file = Path(tmp_dir) / "test.faiss"
            metadata_file = Path(tmp_dir) / "test.pkl"
            faiss_file.touch()
            metadata_file.touch()

            results = retriever.search_similar(
                faiss_file=faiss_file,
                metadata_map_file=metadata_file,
                query_text="test query",
                top_k=2
            )

            # Verify results
            assert len(results) == 2
            assert results[0]['text'] == 'result 1'
            assert results[0]['similarity_score'] == 0.9
            assert results[1]['text'] == 'result 2'
            assert results[1]['similarity_score'] == 0.8

            # Verify calls
            mock_embedder.embed.assert_called_once_with("test query")
            mock_load_method.assert_called_once_with(faiss_file, metadata_file)
            mock_index.search.assert_called_once()


class TestSummaryGenerator:
    """Test cơ bản cho SummaryGenerator"""

    def test_initialization(self):
        """Test constructor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_dir = Path(tmp_dir) / "metadata"
            output_dir = Path(tmp_dir) / "output"

            generator = SummaryGenerator(metadata_dir, output_dir)
            assert generator.metadata_dir == metadata_dir
            assert generator.output_dir == output_dir

    def test_save_document_summary(self):
        """Test save_document_summary method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_dir = Path(tmp_dir) / "metadata"
            output_dir = Path(tmp_dir) / "output"

            generator = SummaryGenerator(metadata_dir, output_dir)

            # Test data
            summary = {
                'filename': 'test.pdf',
                'total_pages': 10,
                'total_chunks': 25,
                'total_tokens': 5000
            }

            summary_file = generator.save_document_summary(summary, "test_doc", "20241201_120000")

            assert summary_file.exists()
            assert summary_file.suffix == '.json'


class TestRAGPipeline:
    """Test cơ bản cho RAGPipeline"""

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_initialization(self, mock_summary_gen, mock_vector_store,
                           mock_embedder_factory, mock_chunker, mock_loader):
        """Test constructor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir, model_type=OllamaModelType.GEMMA)

            assert pipeline.output_dir == Path(tmp_dir)
            assert pipeline.model_type == OllamaModelType.GEMMA

    def test_process_directory_with_no_pdfs(self):
        """Test process_directory với empty directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir)
            results = pipeline.process_directory(tmp_dir)

            assert results == []