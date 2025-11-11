"""
Test script to verify RAG evaluation with different rerankers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

def test_reranker_evaluation():
    """Test that evaluation logging works with different rerankers"""

    print("🧪 Testing RAG Evaluation with Rerankers")
    print("=" * 50)

    # Test query
    query = "What is artificial intelligence?"

    print(f"📝 Query: {query}")

    # Test with different rerankers
    rerankers = ["none", "bge_m3_hf_local"]

    for reranker in rerankers:
        print(f"\n🔄 Testing with reranker: {reranker}")

        try:
            # Call retrieval
            result = fetch_retrieval(
                query_text=query,
                top_k=3,
                embedder_type="huggingface_local",
                reranker_type=reranker,
                use_query_enhancement=False,
                llm_model="gemini"
            )

            print("✅ Retrieval successful"            print(f"📄 Context length: {len(result['context'])} characters")
            print(f"📚 Sources found: {len(result['sources'])}")

        except Exception as e:
            print(f"❌ Retrieval failed: {e}")

    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_reranker_evaluation()