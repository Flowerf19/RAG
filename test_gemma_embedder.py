#!/usr/bin/env python3
"""Test script for Gemma HuggingFace Local Embedder."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedders.embedder_factory import EmbedderFactory

def test_gemma_embedder():
    """Test Gemma embedder with secrets.toml token."""
    print("🧪 Testing Gemma HuggingFace Local Embedder...")

    try:
        # Create Gemma embedder
        embedder = EmbedderFactory.create_gemma_hf_local()

        # Test embedding
        test_text = "This is a test sentence for embedding."
        print(f"📝 Embedding text: {test_text}")

        embedding = embedder.embed(test_text)
        print(f"✅ Embedding generated successfully! Shape: {len(embedding)}")

        # Test batch embedding
        test_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence with more content."
        ]
        print(f"📝 Embedding batch of {len(test_texts)} texts...")

        embeddings = embedder.embed_batch(test_texts)
        print(f"✅ Batch embedding successful! Shape: {len(embeddings)} x {len(embeddings[0])}")

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_gemma_embedder()
    sys.exit(0 if success else 1)
