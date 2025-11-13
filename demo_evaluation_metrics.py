"""
Demo: 3 Core RAG Evaluation Metrics
Ground-truth, Recall, Relevance - Complete Implementation
"""
from evaluation.backend_dashboard.api import BackendDashboard
import pandas as pd

def demo_evaluation_metrics():
    """Demonstrate all 3 core evaluation metrics."""

    backend = BackendDashboard()

    print("🚀 RAG Evaluation Metrics Demo")
    print("=" * 50)
    print("Testing with: embedder=ollama, reranker=none, qem=True, top_k=10, limit=3")
    print()

    # Test parameters
    embedder = "ollama"
    reranker = "none"
    use_qem = True
    top_k = 10
    limit = 3

    try:
        # 1. Ground-truth + Semantic Similarity
        print("1️⃣ GROUND-TRUTH + SEMANTIC SIMILARITY EVALUATION")
        print("-" * 50)

        semantic_result = backend.evaluate_ground_truth_with_semantic_similarity(
            embedder_type=embedder,
            reranker_type=reranker,
            use_query_enhancement=use_qem,
            top_k=top_k,
            limit=limit
        )

        sem_sum = semantic_result['summary']
        print(f"✅ Processed: {sem_sum['processed']}/{sem_sum['total_questions']} questions")
        print(f"📊 Avg Semantic Similarity: {sem_sum['avg_semantic_similarity']:.4f}")
        print(f"🎯 Avg Best Match Score: {sem_sum['avg_best_match_score']:.4f}")
        print(f"📈 Chunks Above Threshold: {sem_sum['total_chunks_above_threshold']}")
        print()

        # 2. Recall Evaluation
        print("2️⃣ RECALL EVALUATION")
        print("-" * 50)

        recall_result = backend.evaluate_recall(
            embedder_type=embedder,
            reranker_type=reranker,
            use_query_enhancement=use_qem,
            top_k=top_k,
            similarity_threshold=0.5,
            limit=limit
        )

        rec_sum = recall_result['summary']
        print(f"📊 Overall Recall: {rec_sum['overall_recall']:.4f}")
        print(f"🎯 Overall Precision: {rec_sum['overall_precision']:.4f}")
        print(f"⚖️  Overall F1 Score: {rec_sum['overall_f1_score']:.4f}")
        print(f"✅ True Positives: {rec_sum['total_true_positives']}")
        print(f"❌ False Positives: {rec_sum['total_false_positives']}")
        print(f"❓ False Negatives: {rec_sum['total_false_negatives']}")
        print()

        # 3. Relevance Evaluation
        print("3️⃣ RELEVANCE EVALUATION")
        print("-" * 50)

        relevance_result = backend.evaluate_relevance(
            embedder_type=embedder,
            reranker_type=reranker,
            use_query_enhancement=use_qem,
            top_k=top_k,
            limit=limit
        )

        rel_sum = relevance_result['summary']
        print(f"🎯 Avg Overall Relevance: {rel_sum['avg_overall_relevance']:.4f}")
        print(f"📊 Avg Chunk Relevance: {rel_sum['avg_chunk_relevance']:.4f}")
        print(f"🌍 Global Avg Relevance: {rel_sum['global_avg_relevance']:.4f}")
        print(f"⭐ High Relevance Ratio (>0.8): {rel_sum['global_high_relevance_ratio']:.4f}")
        print(f"✅ Relevant Ratio (>0.5): {rel_sum['global_relevant_ratio']:.4f}")
        print(f"📈 Total Chunks Evaluated: {rel_sum['total_chunks_evaluated']}")

        # Show relevance distribution
        dist = rel_sum['relevance_distribution']
        print("Relevance Distribution:")
        for bucket, count in dist.items():
            print(f"   {bucket}: {count} chunks")
        print()

        # 4. Comparative Analysis
        print("4️⃣ COMPARATIVE ANALYSIS")
        print("-" * 50)

        comparison = {
            'Metric': [
                'Ground-truth Coverage',
                'Semantic Similarity (0-1)',
                'Recall (0-1)',
                'Precision (0-1)',
                'F1 Score (0-1)',
                'Overall Relevance (0-1)',
                'High Relevance Ratio (>0.8)',
                'Relevant Ratio (>0.5)'
            ],
            'Score': [
                f"{sem_sum['processed']}/{sem_sum['total_questions']}",
                f"{sem_sum['avg_semantic_similarity']:.4f}",
                f"{rec_sum['overall_recall']:.4f}",
                f"{rec_sum['overall_precision']:.4f}",
                f"{rec_sum['overall_f1_score']:.4f}",
                f"{rel_sum['avg_overall_relevance']:.4f}",
                f"{rel_sum['global_high_relevance_ratio']:.1%}",
                f"{rel_sum['global_relevant_ratio']:.1%}"
            ],
            'Interpretation': [
                'Questions processed successfully',
                'Avg similarity to ground truth source',
                'Fraction of relevant chunks retrieved',
                'Fraction of retrieved chunks that are relevant',
                'Harmonic mean of recall and precision',
                'Overall content relevance score',
                'Chunks with very high relevance',
                'Chunks with good relevance'
            ]
        }

        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        print()

        # 5. Sample Detailed Results
        print("5️⃣ SAMPLE DETAILED RESULTS (First Question)")
        print("-" * 50)

        if semantic_result['results']:
            sample = semantic_result['results'][0]
            print(f"Question ID: {sample['ground_truth_id']}")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Retrieved Chunks: {sample['retrieved_chunks']}")
            print(f"Semantic Similarity: {sample['semantic_similarity']:.4f}")
            print(f"Best Match Score: {sample['best_match_score']:.4f}")
            print(f"Chunks Above Threshold: {sample['chunks_above_threshold']}")

            if recall_result['results']:
                rec_sample = recall_result['results'][0]
                print(f"Recall: {rec_sample['recall']:.4f}")
                print(f"Precision: {rec_sample['precision']:.4f}")
                print(f"F1 Score: {rec_sample['f1_score']:.4f}")

            if relevance_result['results']:
                rel_sample = relevance_result['results'][0]
                print(f"Overall Relevance: {rel_sample['overall_relevance']:.4f}")
                print(f"Avg Chunk Relevance: {rel_sample['avg_chunk_relevance']:.4f}")
                print(f"High Relevance Chunks: {rel_sample['high_relevance_chunks']}")

        print()
        print("🎉 Demo completed successfully!")
        print("All 3 core RAG evaluation metrics are now implemented and working!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_evaluation_metrics()