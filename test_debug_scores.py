"""Debug scores in UI context"""
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
    print("DEBUG: Fetching with reranking")
    print("="*80)
    
    result = fetch_retrieval(
        query_text=query,
        top_k=5,
        reranker_type="bge_m3_hf_api",
        embedder_type="huggingface",
        api_tokens={"hf": hf_token}
    )
    
    sources = result.get('sources', [])
    print(f"\nGot {len(sources)} sources from fetch_retrieval\n")
    
    for i, src in enumerate(sources, 1):
        print(f"[{i}] {src.get('file_name')} - page {src.get('page_number')}")
        print(f"    Keys: {list(src.keys())}")
        print(f"    similarity_score: {src.get('similarity_score')}")
        print(f"    vector_similarity: {src.get('vector_similarity')}")
        print(f"    rerank_score: {src.get('rerank_score')}")
        print()

if __name__ == "__main__":
    main()
