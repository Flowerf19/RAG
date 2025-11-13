"""
Test Recall and Relevance Evaluation Metrics
"""
from evaluation.backend_dashboard.api import BackendDashboard

def test_recall_evaluation():
    """Test recall evaluation metric."""

    backend = BackendDashboard()

    print("=== Testing Recall Evaluation ===")
    print("Parameters: embedder=ollama, reranker=none, qem=True, top_k=10, threshold=0.5, limit=2")

    try:
        result = backend.evaluate_recall(
            embedder_type="ollama",
            reranker_type="none",
            use_query_enhancement=True,
            top_k=10,
            similarity_threshold=0.5,
            limit=2
        )

        summary = result.get('summary', {})
        print("\n=== SUMMARY ===")
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Processed: {summary.get('processed', 0)}")
        print(f"Errors: {summary.get('errors', 0)}")
        print(f"Overall Recall: {summary.get('overall_recall', 0):.4f}")
        print(f"Overall Precision: {summary.get('overall_precision', 0):.4f}")
        print(f"Overall F1 Score: {summary.get('overall_f1_score', 0):.4f}")
        print(f"Average Recall: {summary.get('avg_recall', 0):.4f}")
        print(f"Average Precision: {summary.get('avg_precision', 0):.4f}")
        print(f"Total True Positives: {summary.get('total_true_positives', 0)}")
        print(f"Total False Positives: {summary.get('total_false_positives', 0)}")
        print(f"Total False Negatives: {summary.get('total_false_negatives', 0)}")

        results = result.get('results', [])
        print("\n=== DETAILED RESULTS ===")
        for i, res in enumerate(results, 1):
            print(f"{i}. Question ID {res.get('ground_truth_id', 'unknown')}:")
            print(f"   Recall: {res.get('recall', 0):.4f}")
            print(f"   Precision: {res.get('precision', 0):.4f}")
            print(f"   F1 Score: {res.get('f1_score', 0):.4f}")
            print(f"   TP: {res.get('true_positives', 0)}, FP: {res.get('false_positives', 0)}, FN: {res.get('false_negatives', 0)}")
            if res.get('error'):
                print(f"   ❌ Error: {res['error']}")
            print()

    except Exception as e:
        print(f"❌ Recall evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def test_relevance_evaluation():
    """Test relevance evaluation metric."""

    backend = BackendDashboard()

    print("\n=== Testing Relevance Evaluation ===")
    print("Parameters: embedder=ollama, reranker=none, qem=True, top_k=10, limit=2")

    try:
        result = backend.evaluate_relevance(
            embedder_type="ollama",
            reranker_type="none",
            use_query_enhancement=True,
            top_k=10,
            limit=2
        )

        summary = result.get('summary', {})
        print("\n=== SUMMARY ===")
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Processed: {summary.get('processed', 0)}")
        print(f"Errors: {summary.get('errors', 0)}")
        print(f"Average Overall Relevance: {summary.get('avg_overall_relevance', 0):.4f}")
        print(f"Average Chunk Relevance: {summary.get('avg_chunk_relevance', 0):.4f}")
        print(f"Average Semantic Similarity: {summary.get('avg_semantic_similarity', 0):.4f}")
        print(f"Global Average Relevance: {summary.get('global_avg_relevance', 0):.4f}")
        print(f"Global High Relevance Ratio (>0.8): {summary.get('global_high_relevance_ratio', 0):.4f}")
        print(f"Global Relevant Ratio (>0.5): {summary.get('global_relevant_ratio', 0):.4f}")
        print(f"Total Chunks Evaluated: {summary.get('total_chunks_evaluated', 0)}")

        dist = summary.get('relevance_distribution', {})
        print("Relevance Distribution:")
        for bucket, count in dist.items():
            print(f"   {bucket}: {count}")

        results = result.get('results', [])
        print("\n=== DETAILED RESULTS ===")
        for i, res in enumerate(results, 1):
            print(f"{i}. Question ID {res.get('ground_truth_id', 'unknown')}:")
            print(f"   Overall Relevance: {res.get('overall_relevance', 0):.4f}")
            print(f"   Avg Chunk Relevance: {res.get('avg_chunk_relevance', 0):.4f}")
            print(f"   Semantic Similarity: {res.get('semantic_similarity', 0):.4f}")
            print(f"   Relevant Chunks Ratio: {res.get('relevant_chunks_ratio', 0):.4f}")
            print(f"   High Relevance Chunks: {res.get('high_relevance_chunks', 0)}")
            if res.get('error'):
                print(f"   ❌ Error: {res['error']}")
            print()

    except Exception as e:
        print(f"❌ Relevance evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recall_evaluation()
    test_relevance_evaluation()