"""
Test Database Storage for Evaluation Metrics
"""
from evaluation.backend_dashboard.api import BackendDashboard

def test_database_storage():
    """Test that evaluation results are saved to database."""

    backend = BackendDashboard()

    print("=== Testing Database Storage for Evaluation Results ===")

    # Get initial metrics count
    initial_metrics = backend.db.get_metrics(limit=1000)
    initial_count = len(initial_metrics)
    print(f"Initial metrics in database: {initial_count}")

    # Get initial ground truth count
    initial_gt = backend.db.get_ground_truth_list(limit=1000)
    initial_gt_count = len(initial_gt)
    print(f"Initial ground truth entries: {initial_gt_count}")

    try:
        # Run semantic similarity evaluation with database storage
        print("\n--- Running Semantic Similarity Evaluation (with DB storage) ---")
        semantic_result = backend.evaluate_ground_truth_with_semantic_similarity(
            embedder_type="ollama",
            reranker_type="none",
            use_query_enhancement=True,
            top_k=10,
            limit=2,
            save_to_db=True
        )

        sem_summary = semantic_result['summary']
        print(f"✅ Semantic evaluation completed: {sem_summary['processed']} processed")

        # Run recall evaluation with database storage
        print("\n--- Running Recall Evaluation (with DB storage) ---")
        recall_result = backend.evaluate_recall(
            embedder_type="ollama",
            reranker_type="none",
            use_query_enhancement=True,
            top_k=10,
            similarity_threshold=0.5,
            limit=2,
            save_to_db=True
        )

        rec_summary = recall_result['summary']
        print(f"✅ Recall evaluation completed: {rec_summary['processed']} processed")

        # Run relevance evaluation with database storage
        print("\n--- Running Relevance Evaluation (with DB storage) ---")
        relevance_result = backend.evaluate_relevance(
            embedder_type="ollama",
            reranker_type="none",
            use_query_enhancement=True,
            top_k=10,
            limit=2,
            save_to_db=True
        )

        rel_summary = relevance_result['summary']
        print(f"✅ Relevance evaluation completed: {rel_summary['processed']} processed")

        # Check final metrics count
        final_metrics = backend.db.get_metrics(limit=1000)
        final_count = len(final_metrics)
        print(f"\nFinal metrics in database: {final_count}")
        print(f"New metrics added: {final_count - initial_count}")

        # Check if new metrics were added
        if final_count > initial_count:
            print("✅ SUCCESS: New evaluation metrics were saved to database!")

            # Show recent metrics
            recent_metrics = backend.db.get_metrics(limit=5)
            print("\n--- Recent Metrics in Database ---")
            for i, metric in enumerate(recent_metrics[:3], 1):
                print(f"{i}. {metric.get('query', 'Unknown')} - {metric.get('model', 'Unknown')} - {metric.get('timestamp', 'Unknown')[:19]}")
                if metric.get('recall'):
                    print(f"   Recall: {metric['recall']}")
                if metric.get('relevance'):
                    print(f"   Relevance: {metric['relevance']}")
        else:
            print("❌ WARNING: No new metrics were added to database")

        # Check ground truth updates
        final_gt = backend.db.get_ground_truth_list(limit=1000)
        updated_gt = [gt for gt in final_gt if gt.get('evaluated_at')]
        print(f"\nGround truth entries with evaluation results: {len(updated_gt)}")

        if updated_gt:
            print("✅ SUCCESS: Ground truth entries were updated with evaluation results!")
            # Show one example
            example = updated_gt[0]
            print(f"Example GT ID {example.get('id')}: evaluated_at={example.get('evaluated_at')}, retrieval_chunks={example.get('retrieval_chunks')}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_storage()