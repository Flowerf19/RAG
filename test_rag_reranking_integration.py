"""
Test RAG System Integration with Reranking
===========================================
Test toàn bộ flow: Embedding Retrieval -> Reranking -> Final Results
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rag_integration_with_reranking():
    """Test RAG system với reranking integration"""
    
    try:
        from pipeline.backend_connector import fetch_retrieval
        
        # Test queries
        queries = [
            "What are the benefits of regular exercise?",
            "Explain machine learning algorithms",
            "How does risk management work?"
        ]
        
        # Test configurations
        configs = [
            {
                "name": "No Reranking",
                "top_k_embed": 10,
                "top_k_rerank": 5,
                "embedder_type": "huggingface_local",
                "reranker_type": "none"
            },
            {
                "name": "BGE-M3 HF Local Reranking",
                "top_k_embed": 10,
                "top_k_rerank": 5,
                "embedder_type": "huggingface_local",
                "reranker_type": "bge_m3_hf_local"
            },
            {
                "name": "BGE Local Reranking",
                "top_k_embed": 15,
                "top_k_rerank": 3,
                "embedder_type": "huggingface_local",
                "reranker_type": "bge_local"
            }
        ]
        
        print("\n" + "="*80)
        print("RAG SYSTEM INTEGRATION TEST WITH RERANKING")
        print("="*80)
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            print("-"*80)
            
            for config in configs:
                print(f"\n🔧 Configuration: {config['name']}")
                print(f"   Top K Embed: {config['top_k_embed']}")
                print(f"   Top K Rerank: {config['top_k_rerank']}")
                print(f"   Embedder: {config['embedder_type']}")
                print(f"   Reranker: {config['reranker_type']}")
                
                try:
                    result = fetch_retrieval(
                        query_text=query,
                        top_k_embed=config['top_k_embed'],
                        top_k_rerank=config['top_k_rerank'],
                        max_chars=2000,
                        embedder_type=config['embedder_type'],
                        reranker_type=config['reranker_type']
                    )
                    
                    # Display results
                    context = result.get("context", "")
                    sources = result.get("sources", [])
                    retrieval_info = result.get("retrieval_info", {})
                    
                    print("\n   📊 Retrieval Info:")
                    print(f"      Total Retrieved: {retrieval_info.get('total_retrieved', 0)}")
                    print(f"      Final Count: {retrieval_info.get('final_count', 0)}")
                    print(f"      Reranked: {retrieval_info.get('reranked', False)}")
                    print(f"      Reranker Used: {retrieval_info.get('reranker', 'none')}")
                    
                    print(f"\n   📚 Sources ({len(sources)}):")
                    for i, src in enumerate(sources[:3], 1):  # Show top 3
                        file_name = src.get("file_name", "?")
                        page = src.get("page_number", "?")
                        sim_score = src.get("similarity_score", 0.0)
                        rerank_score = src.get("rerank_score")
                        
                        if rerank_score is not None:
                            print(f"      [{i}] {file_name} - Page {page}")
                            print(f"          Similarity: {sim_score:.4f} | Rerank: {rerank_score:.4f}")
                        else:
                            print(f"      [{i}] {file_name} - Page {page} (Score: {sim_score:.4f})")
                    
                    print(f"\n   📝 Context Preview ({len(context)} chars):")
                    preview = context[:200] + "..." if len(context) > 200 else context
                    print(f"      {preview}")
                    
                    print(f"\n   ✅ Test passed for {config['name']}")
                    
                except Exception as e:
                    print(f"\n   ❌ Test failed for {config['name']}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print("\n" + "-"*80)
        
        print("\n" + "="*80)
        print("✅ RAG INTEGRATION TEST COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_rerank_score_comparison():
    """Test so sánh điểm số trước và sau reranking"""
    
    try:
        from pipeline.backend_connector import fetch_retrieval
        
        query = "Benefits of exercise and physical activity"
        
        print("\n" + "="*80)
        print("RERANK SCORE COMPARISON TEST")
        print("="*80)
        print(f"\n📝 Query: {query}")
        
        # Test without reranking
        print("\n🔧 Test 1: WITHOUT Reranking")
        result_no_rerank = fetch_retrieval(
            query_text=query,
            top_k_embed=10,
            top_k_rerank=5,
            embedder_type="huggingface_local",
            reranker_type="none"
        )
        
        print("\n   Top 5 Results (No Reranking):")
        for i, src in enumerate(result_no_rerank.get("sources", [])[:5], 1):
            print(f"   [{i}] Similarity: {src.get('similarity_score', 0.0):.4f}")
            print(f"       {src.get('file_name', '?')} - Page {src.get('page_number', '?')}")
        
        # Test with reranking
        print("\n🔧 Test 2: WITH BGE-M3 HF Local Reranking")
        result_with_rerank = fetch_retrieval(
            query_text=query,
            top_k_embed=10,
            top_k_rerank=5,
            embedder_type="huggingface_local",
            reranker_type="bge_m3_hf_local"
        )
        
        print("\n   Top 5 Results (With Reranking):")
        for i, src in enumerate(result_with_rerank.get("sources", [])[:5], 1):
            rerank_score = src.get('rerank_score', 0.0)
            sim_score = src.get('similarity_score', 0.0)
            print(f"   [{i}] Similarity: {sim_score:.4f} | Rerank: {rerank_score:.4f}")
            print(f"       {src.get('file_name', '?')} - Page {src.get('page_number', '?')}")
        
        print("\n" + "="*80)
        print("✅ RERANK COMPARISON TEST COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting RAG System Integration Tests...")
    print("="*80)
    
    # Run tests
    test_rag_integration_with_reranking()
    print("\n" + "="*80 + "\n")
    test_rerank_score_comparison()
    
    print("\n✅ All tests completed!")
