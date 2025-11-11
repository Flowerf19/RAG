"""
Test script to verify RAG evaluation integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

def test_evaluation_integration():
    """Test that evaluation logging works with RAG pipeline"""

    print("🧪 Testing RAG Evaluation Integration")
    print("=" * 50)

    # Test query
    query = "What is machine learning?"

    print(f"📝 Query: {query}")

    try:
        # Call retrieval (this should trigger evaluation logging)
        result = fetch_retrieval(
            query_text=query,
            top_k=3,
            embedder_type="huggingface_local",
            reranker_type="bge_m3_hf_local",
            use_query_enhancement=True,  # Test with query enhancement
            llm_model="gemini"
        )

        print("✅ Retrieval successful")
        print(f"📄 Context length: {len(result['context'])} characters")
        print(f"📚 Sources found: {len(result['sources'])}")
        print(f"🔍 Retrieval info: {result['retrieval_info']}")

        # Check if evaluation was logged
        print("\n📊 Checking evaluation metrics...")
        try:
            from evaluation.metrics.database import MetricsDB
            db = MetricsDB()
            recent_metrics = db.get_metrics(limit=5)

            if recent_metrics:
                latest = recent_metrics[0]
                print("✅ Evaluation logged successfully!")
                print(f"   Model: {latest['model']}")
                latency = latest.get('latency', 0)
                faithfulness = latest.get('faithfulness', 0)
                relevance = latest.get('relevance', 0)
                print(f"   Latency: {latency}")
                print(f"   Faithfulness: {faithfulness}")
                print(f"   Relevance: {relevance}")
                print(f"   Error: {latest.get('error', False)}")
            else:
                print("⚠️  No evaluation metrics found")

        except Exception as e:
            print(f"❌ Error checking metrics: {e}")

    except Exception as e:
        print(f"❌ Retrieval failed: {e}")

    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_evaluation_integration()