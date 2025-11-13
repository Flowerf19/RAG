"""
Test Enhanced Ground Truth Evaluation with Semantic Similarity
"""
from evaluation.backend_dashboard.api import BackendDashboard

def test_enhanced_ground_truth_evaluation():
    """Test enhanced ground truth evaluation with semantic similarity."""

    # Initialize backend
    backend = BackendDashboard()

    # Test parameters
    embedder_choice = "ollama"
    reranker_choice = "none"
    use_qem = True
    limit = 2  # Test with just 2 questions for speed

    print("=== Testing Enhanced Ground Truth Evaluation with Semantic Similarity ===")
    print(f"Embedder: {embedder_choice}")
    print(f"Reranker: {reranker_choice}")
    print(f"Query Enhancement: {use_qem}")
    print(f"Limit: {limit} questions")
    print()

    try:
        # Run enhanced evaluation
        result = backend.evaluate_ground_truth_with_semantic_similarity(
            embedder_type=embedder_choice,
            reranker_type=reranker_choice,
            use_query_enhancement=use_qem,
            top_k=10,
            limit=limit
        )

        # Print summary
        summary = result.get('summary', {})
        print("=== SUMMARY ===")
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Processed: {summary.get('processed', 0)}")
        print(f"Errors: {summary.get('errors', 0)}")
        print(f"Average Semantic Similarity: {summary.get('avg_semantic_similarity', 0):.4f}")
        print(f"Average Best Match Score: {summary.get('avg_best_match_score', 0):.4f}")
        print(f"Total Chunks Above Threshold: {summary.get('total_chunks_above_threshold', 0)}")
        print(f"Embedder Used: {summary.get('embedder_used', 'unknown')}")
        print(f"Reranker Used: {summary.get('reranker_used', 'unknown')}")
        print(f"Query Enhancement Used: {summary.get('query_enhancement_used', False)}")
        print()

        # Print detailed results
        results = result.get('results', [])
        print("=== DETAILED RESULTS ===")
        for i, res in enumerate(results, 1):
            print(f"{i}. Question ID {res.get('ground_truth_id', 'unknown')}:")
            print(f"   Semantic Similarity: {res.get('semantic_similarity', 0):.4f}")
            print(f"   Best Match Score: {res.get('best_match_score', 0):.4f}")
            print(f"   Retrieved Chunks: {res.get('retrieved_chunks', 0)}")
            print(f"   Chunks Above Threshold: {res.get('chunks_above_threshold', 0)}")

            if res.get('error'):
                print(f"   ❌ Error: {res['error']}")
            else:
                matched_chunks = res.get('matched_chunks', [])
                if matched_chunks:
                    print("   ✅ Top matches:")
                    for j, chunk in enumerate(matched_chunks[:2], 1):  # Show top 2
                        print(f"      {j}. Score {chunk['similarity_score']:.4f} - {chunk['file_name']} (page {chunk['page_number']})")
                else:
                    print("   ⚠️ No chunks matched above threshold")
            print()

        # Print errors if any
        errors_list = result.get('errors_list', [])
        if errors_list:
            print("=== ERRORS ===")
            for error in errors_list[:5]:  # Show first 5 errors
                print(f"- {error}")
            if len(errors_list) > 5:
                print(f"... and {len(errors_list) - 5} more errors")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_ground_truth_evaluation()