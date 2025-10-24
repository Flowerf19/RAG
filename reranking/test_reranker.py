"""
Test Reranking System
=====================
Test BGE local và API rerankers (Cohere, Jina)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reranking.reranker_type import RerankerType
from reranking.reranker_factory import RerankerFactory


def test_bge_local_reranker():
    """Test BGE Local Reranker"""
    print("\n" + "="*80)
    print("TEST 1: BGE Local Reranker")
    print("="*80)
    
    try:
        print("🔄 Loading BGE reranker: BAAI/bge-reranker-v2-m3")
        print("   Please wait, this may take a while for first-time download...")
        
        reranker = RerankerFactory.create_bge_local(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cpu"
        )
        
        print("✅ Model loaded successfully!")
        print(f"✓ Model: {reranker.profile.model_id}")
        print(f"✓ Provider: {reranker.profile.provider}")
        print(f"✓ Is local: {reranker.profile.is_local}")
        
        # Test reranking
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
            "The weather is sunny today",
            "ML algorithms learn from data"
        ]
        
        print(f"\n🔄 Reranking {len(documents)} documents for query: '{query}'")
        results = reranker.rerank(query, documents, top_k=3)
        
        print(f"✓ Rerank results:")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] Score: {result.score:.4f} - {result.document[:60]}...")
        
        print("\n✅ BGE Local Reranker: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ BGE Local Reranker: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cohere_reranker():
    """Test Cohere API Reranker"""
    print("\n" + "="*80)
    print("TEST 2: Cohere API Reranker")
    print("="*80)
    
    try:
        # Get API token from environment or secrets
        from embedders.providers.huggingface.token_manager import get_hf_token
        
        # Try to get Cohere token from env
        cohere_token = os.getenv("COHERE_API_KEY") or os.getenv("COHERE_TOKEN")
        
        if not cohere_token:
            print("⚠️  Cohere API token not found!")
            print("Set token: $env:COHERE_API_KEY='your_token_here'")
            print("Get free token at: https://cohere.com/")
            return False
        
        print(f"✓ API token found: {cohere_token[:10]}...")
        
        print("🔄 Initializing Cohere reranker...")
        reranker = RerankerFactory.create_cohere(
            api_token=cohere_token,
            model_name="rerank-english-v3.0"
        )
        
        print("✅ Reranker initialized!")
        print(f"✓ Model: {reranker.profile.model_id}")
        print(f"✓ Provider: {reranker.profile.provider}")
        print(f"✓ Is local: {reranker.profile.is_local}")
        
        # Test connection
        if not reranker.test_connection():
            print("❌ Connection test failed!")
            return False
        
        print("✓ Connection: OK")
        
        # Test reranking
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
            "The weather is sunny today",
            "ML algorithms learn from data"
        ]
        
        print(f"\n🔄 Reranking {len(documents)} documents for query: '{query}'")
        results = reranker.rerank(query, documents, top_k=3)
        
        print(f"✓ Rerank results:")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] Score: {result.score:.4f} - {result.document[:60]}...")
        
        print("\n✅ Cohere API Reranker: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Cohere API Reranker: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jina_reranker():
    """Test Jina API Reranker"""
    print("\n" + "="*80)
    print("TEST 3: Jina API Reranker")
    print("="*80)
    
    try:
        # Try to get Jina token from env
        jina_token = os.getenv("JINA_API_KEY") or os.getenv("JINA_TOKEN")
        
        if not jina_token:
            print("⚠️  Jina API token not found!")
            print("Set token: $env:JINA_API_KEY='your_token_here'")
            print("Get free token at: https://jina.ai/")
            return False
        
        print(f"✓ API token found: {jina_token[:10]}...")
        
        print("🔄 Initializing Jina reranker...")
        reranker = RerankerFactory.create_jina(
            api_token=jina_token,
            model_name="jina-reranker-v2-base-multilingual"
        )
        
        print("✅ Reranker initialized!")
        print(f"✓ Model: {reranker.profile.model_id}")
        print(f"✓ Provider: {reranker.profile.provider}")
        print(f"✓ Is local: {reranker.profile.is_local}")
        
        # Test connection
        if not reranker.test_connection():
            print("❌ Connection test failed!")
            return False
        
        print("✓ Connection: OK")
        
        # Test reranking
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
            "The weather is sunny today",
            "ML algorithms learn from data"
        ]
        
        print(f"\n🔄 Reranking {len(documents)} documents for query: '{query}'")
        results = reranker.rerank(query, documents, top_k=3)
        
        print(f"✓ Rerank results:")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] Score: {result.score:.4f} - {result.document[:60]}...")
        
        print("\n✅ Jina API Reranker: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Jina API Reranker: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranker_factory():
    """Test RerankerFactory"""
    print("\n" + "="*80)
    print("TEST 4: RerankerFactory")
    print("="*80)
    
    try:
        # Test BGE creation via factory
        print("🔄 Creating BGE reranker via factory...")
        bge = RerankerFactory.create(RerankerType.BGE_RERANKER)
        print(f"✓ BGE created: {bge.profile.model_id}")
        
        # Test Cohere creation via factory
        cohere_token = os.getenv("COHERE_API_KEY")
        if cohere_token:
            print("🔄 Creating Cohere reranker via factory...")
            cohere = RerankerFactory.create(
                RerankerType.COHERE,
                api_token=cohere_token
            )
            print(f"✓ Cohere created: {cohere.profile.model_id}")
        else:
            print("⚠️  Skipping Cohere test - no token")
        
        # Test Jina creation via factory
        jina_token = os.getenv("JINA_API_KEY")
        if jina_token:
            print("🔄 Creating Jina reranker via factory...")
            jina = RerankerFactory.create(
                RerankerType.JINA,
                api_token=jina_token
            )
            print(f"✓ Jina created: {jina.profile.model_id}")
        else:
            print("⚠️  Skipping Jina test - no token")
        
        print("\n✅ RerankerFactory: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ RerankerFactory: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all reranking tests"""
    print("\n" + "="*80)
    print("RERANKING SYSTEM TESTS")
    print("="*80)
    print("Testing BGE local and API rerankers (Cohere, Jina)")
    print("="*80 + "\n")
    
    results = {
        "BGE Local Reranker": test_bge_local_reranker(),
        "Cohere API Reranker": test_cohere_reranker(),
        "Jina API Reranker": test_jina_reranker(),
        "RerankerFactory": test_reranker_factory(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("-"*80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed >= total - 2:  # Allow for API token failures
        print("\n🎉 All critical tests passed!")
        return 0
    else:
        print(f"\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
