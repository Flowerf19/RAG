"""
Test Semantic Similarity with Source Evaluation
"""
from evaluation.backend_dashboard.api import BackendDashboard

def test_semantic_similarity_evaluation():
    """Test semantic similarity evaluation with true source."""

    # Initialize backend
    backend = BackendDashboard()

    # Get ground truth data
    rows = backend.get_ground_truth_list(limit=5)
    print(f"Found {len(rows)} ground truth rows")

    if not rows:
        print("No ground truth data found!")
        return

    # Test parameters
    embedder_choice = "ollama"

    # Test semantic similarity for first row
    test_row = rows[0]
    gt_id = test_row.get('id')
    question = test_row.get('question')
    true_source = test_row.get('source', '')

    print(f"\n=== Testing Semantic Similarity for ID {gt_id} ===")
    print(f"Question: {question}")
    print(f"True Source (first 200 chars): {true_source[:200]}...")

    try:
        # First, get retrieval results
        from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

        result = fetch_retrieval(
            query_text=question,
            top_k=10,
            embedder_type=embedder_choice,
            reranker_type="none",
            use_query_enhancement=True
        )

        sources = result.get('sources', [])
        print(f"Retrieved {len(sources)} chunks")

        if sources:
            # Now evaluate semantic similarity with true source
            similarity_result = backend.evaluate_semantic_similarity_with_source(
                retrieved_sources=sources,
                ground_truth_id=gt_id,
                embedder_type=embedder_choice
            )

            print("\n=== Semantic Similarity Results ===")
            print(f"Average Semantic Similarity: {similarity_result.get('semantic_similarity', 0):.4f}")
            print(f"Best Match Score: {similarity_result.get('best_match_score', 0):.4f}")
            print(f"Total Chunks Evaluated: {similarity_result.get('total_chunks_evaluated', 0)}")
            print(f"Chunks Above Threshold (0.5): {similarity_result.get('chunks_above_threshold', 0)}")
            print(f"True Source Length: {similarity_result.get('true_source_length', 0)} chars")
            print(f"Embedder Used: {similarity_result.get('embedder_used', 'unknown')}")

            if similarity_result.get('error'):
                print(f"Error: {similarity_result['error']}")

            # Show top matched chunks
            matched_chunks = similarity_result.get('matched_chunks', [])
            if matched_chunks:
                print("\n=== Top Matched Chunks ===")
                for i, chunk in enumerate(matched_chunks[:3], 1):  # Show top 3
                    print(f"{i}. Score: {chunk['similarity_score']:.4f}")
                    print(f"   File: {chunk['file_name']} (page {chunk['page_number']})")
                    print(f"   Text: {chunk['chunk_text'][:100]}...")
                    print()
            else:
                print("\nNo chunks matched above threshold (0.5)")
        else:
            print("No sources retrieved to evaluate!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_semantic_similarity_evaluation()