#!/usr/bin/env python3
"""
Test script to verify new multilingual embedding models work in the RAG pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

def test_new_embedders():
    """Test that new multilingual embedders are properly switched in the pipeline"""

    # Test each new model
    new_models = [
        "e5_large_instruct",
        "e5_base",
        "gte_multilingual_base",
        "paraphrase_mpnet_base_v2",
        "paraphrase_minilm_l12_v2"
    ]

    for model in new_models:
        print(f"\n🧪 Testing {model}...")

        try:
            # Create retrieval with the model
            results = fetch_retrieval(
                query_text="test query",
                top_k=1,
                embedder_type=model
            )
            print(f"   ✅ Retrieval successful, got {len(results.get('results', []))} results")

        except Exception as e:
            print(f"   ❌ Error with {model}: {e}")

    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_new_embedders()