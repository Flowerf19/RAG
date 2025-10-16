#!/usr/bin/env python3
"""
Pipeline Test Runner
====================
Chạy các test pipeline một cách thủ công
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vector_store():
    """Test VectorStore functionality"""
    print("🧪 Testing VectorStore...")

    try:
        from RAG_system.pipeline.vector_store import VectorStore
        import tempfile
        import numpy as np
        import faiss

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(Path(temp_dir))

            # Test data
            embeddings_data = [
                {
                    "chunk_id": "chunk_1",
                    "chunk_index": 0,
                    "text": "test text 1",
                    "text_length": 11,
                    "token_count": 3,
                    "embedding": [0.1] * 768,
                    "embedding_dimension": 768,
                    "file_name": "test.pdf",
                    "file_path": "test.pdf",
                    "page_number": 1,
                    "page_numbers": [1],
                    "block_type": "text",
                    "block_ids": [],
                    "is_table": False
                },
                {
                    "chunk_id": "chunk_2",
                    "chunk_index": 1,
                    "text": "test text 2",
                    "text_length": 11,
                    "token_count": 3,
                    "embedding": [0.2] * 768,
                    "embedding_dimension": 768,
                    "file_name": "test.pdf",
                    "file_path": "test.pdf",
                    "page_number": 2,
                    "page_numbers": [2],
                    "block_type": "text",
                    "block_ids": [],
                    "is_table": False
                }
            ]

            # Test create_index
            faiss_file, metadata_file = store.create_index(embeddings_data, "test_doc", "20241201_120000")

            assert faiss_file.exists(), "FAISS file should be created"
            assert metadata_file.exists(), "Metadata file should be created"

            print("✅ VectorStore tests passed")
            return True

    except Exception as e:
        print(f"❌ VectorStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retriever():
    """Test Retriever functionality"""
    print("🧪 Testing Retriever...")

    try:
        from RAG_system.pipeline.retriever import Retriever
        from RAG_system.EMBEDDERS.providers.ollama.gemma_embedder import GemmaEmbedder
        from unittest.mock import patch

        # Mock the embedder
        with patch('embedders.providers.ollama.base_ollama_embedder.BaseOllamaEmbedder._generate_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 768

            embedder = GemmaEmbedder.create_default()
            retriever = Retriever(embedder)

            # Test search (would need actual FAISS index)
            print("✅ Retriever initialization passed")
            return True

    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_summary_generator():
    """Test SummaryGenerator functionality"""
    print("🧪 Testing SummaryGenerator...")

    try:
        from RAG_system.pipeline.summary_generator import SummaryGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = SummaryGenerator(Path(temp_dir), Path(temp_dir))

            # Mock summary data
            summary = {
                "filename": "test.pdf",
                "total_pages": 5,
                "total_chunks": 10,
                "total_tokens": 1000,
                "processing_time": 30.5
            }

            summary_file = generator.save_document_summary(summary, "test_doc", "20241201_120000")

            assert summary_file.exists(), "Summary file should be created"

            print("✅ SummaryGenerator tests passed")
            return True

    except Exception as e:
        print(f"❌ SummaryGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_pipeline():
    """Test RAGPipeline initialization"""
    print("🧪 Testing RAGPipeline...")

    try:
        from RAG_system.pipeline.rag_pipeline import RAGPipeline
        from RAG_system.EMBEDDERS.providers.ollama import OllamaModelType

        # Test initialization only (don't process actual PDF)
        pipeline = RAGPipeline(model_type=OllamaModelType.GEMMA)

        assert pipeline.loader is not None, "Loader should be initialized"
        assert pipeline.chunker is not None, "Chunker should be initialized"
        assert pipeline.embedder is not None, "Embedder should be initialized"
        assert pipeline.vector_store is not None, "VectorStore should be initialized"

        print("✅ RAGPipeline initialization passed")
        return True

    except Exception as e:
        print(f"❌ RAGPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Running Pipeline Tests Manually")
    print("=" * 50)

    results = []

    # Run individual component tests
    results.append(("VectorStore", test_vector_store()))
    results.append(("Retriever", test_retriever()))
    results.append(("SummaryGenerator", test_summary_generator()))
    results.append(("RAGPipeline", test_rag_pipeline()))

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} PASSED")

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    if passed == total:
        print("\n🎉 All Pipeline Tests PASSED!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")

if __name__ == "__main__":
    main()