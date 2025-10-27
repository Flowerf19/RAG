"""Test score display fix"""
import sys
import io
from pipeline.backend_connector import fetch_retrieval
from embedders.providers.huggingface.token_manager import get_hf_token

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    query = "risk management"
    hf_token = get_hf_token()
    
    print("="*80)
    print("Testing Score Display")
    print("="*80)
    
    # Test with reranking
    result = fetch_retrieval(
        query_text=query,
        top_k=3,
        reranker_type="bge_m3_hf_api",
        embedder_type="huggingface",
        api_tokens={"hf": hf_token}
    )
    
    print(f"\nQuery: {query}")
    print(f"Results: {len(result['sources'])}\n")
    
    for i, src in enumerate(result['sources'], 1):
        hybrid = src.get('similarity_score', 0)
        vector = src.get('vector_similarity')
        rerank = src.get('rerank_score')
        page = src.get('page_number', '?')
        
        print(f"[{i}] Page {page}")
        print(f"  Hybrid Score (z-score): {hybrid:.4f}")
        if vector is not None:
            print(f"  Vector Similarity (cosine): {vector:.4f}")
        if rerank is not None:
            print(f"  Rerank Score: {rerank:.4f}")
        print(f"  Snippet: {src.get('snippet', '')[:60]}...")
        print()
    
    # Verification
    print("="*80)
    print("VERIFICATION")
    print("="*80)
    
    checks = []
    
    # Check 1: All results have rerank_score
    has_rerank = all('rerank_score' in s for s in result['sources'])
    if has_rerank:
        print("✅ All results have rerank_score")
        checks.append(True)
    else:
        print("❌ Some results missing rerank_score")
        checks.append(False)
    
    # Check 2: Vector similarity exists
    has_vector = any('vector_similarity' in s and s['vector_similarity'] is not None for s in result['sources'])
    if has_vector:
        print("✅ Vector similarity available")
        checks.append(True)
    else:
        print("❌ Vector similarity missing")
        checks.append(False)
    
    # Check 3: Rerank scores are different from hybrid scores
    diff_scores = False
    for s in result['sources']:
        hybrid = s.get('similarity_score', 0)
        rerank = s.get('rerank_score', 0)
        if abs(hybrid - rerank) > 0.001:
            diff_scores = True
            break
    
    if diff_scores:
        print("✅ Rerank scores differ from hybrid scores")
        checks.append(True)
    else:
        print("❌ Rerank scores same as hybrid scores (incorrect)")
        checks.append(False)
    
    # Check 4: Rerank scores descending
    rerank_scores = [s.get('rerank_score', 0) for s in result['sources']]
    is_desc = all(rerank_scores[i] >= rerank_scores[i+1] for i in range(len(rerank_scores)-1))
    if is_desc:
        print(f"✅ Rerank scores descending: {[f'{s:.4f}' for s in rerank_scores]}")
        checks.append(True)
    else:
        print(f"❌ Rerank scores NOT descending: {[f'{s:.4f}' for s in rerank_scores]}")
        checks.append(False)
    
    print("\n" + "="*80)
    if all(checks):
        print("✅ ALL CHECKS PASSED")
    else:
        print(f"❌ FAILED: {checks.count(False)}/{len(checks)} checks")
    print("="*80)

if __name__ == "__main__":
    main()
