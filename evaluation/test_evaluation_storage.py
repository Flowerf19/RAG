#!/usr/bin/env python3
"""
Test script to verify evaluation data storage and computation.
Tests the complete evaluation pipeline with a specific ground truth question.
"""

import sys
import os
import logging
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.backend_dashboard.api import BackendDashboard

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_evaluation_storage():
    """Test evaluation data storage and computation."""
    pytest.skip("This test is for the old evaluation system, now using Ragas")
    print("🚀 Starting evaluation storage test...")

    # Initialize backend
    backend = BackendDashboard()

    # Test ground truth data
    test_question = "Chủ sở hữu của Quy trình Quản lý Rủi ro (ISMS/PR_RSK001) là ai?"
    test_answer = "Information Security Committee (Ủy ban An ninh Thông tin)."
    test_source = """Risk Management Owner: Information Security Committee
ISMS/PR_RSK001 Version: 5.0 Company:
Revision History
Date Version Description Author Reviewer
19/05/2015 1.0 Issue version
23/07/2019 2.0 Generalization for company use and for ISO
27001
19/04/2021 3.0 3rd issue version
02/06/2022 4.0 4th issue version, minor change related
requirements of the certificate
28/05/2024 5.0 - Updated according to the requirements
of ISO 27001:2022 Standard
- Updated according to 's
Process_Risk Management.pdf"""

    # Clear existing ground truth for clean test
    print("🧹 Clearing existing ground truth data...")
    try:
        # Note: This assumes there's a method to clear data, if not we'll work with existing
        pass
    except Exception as e:
        print(f"⚠️ Could not clear existing data: {e}")

    # Insert test ground truth
    print("📝 Inserting test ground truth...")
    ground_truth_rows = [{
        'question': test_question,
        'answer': test_answer,
        'source': test_source
    }]

    inserted_count = backend.insert_ground_truth_rows(ground_truth_rows)
    print(f"✅ Inserted {inserted_count} ground truth rows")

    # Verify ground truth was inserted
    gt_list = backend.get_ground_truth_list()
    print(f"📊 Current ground truth count: {len(gt_list)}")
    if gt_list:
        print(f"📋 Latest ground truth: {gt_list[-1]['question'][:50]}...")

    # Run batch evaluation with specified parameters
    print("🔬 Running batch evaluation...")
    print("Parameters:")
    print("  - embedder_type: huggingface_local")
    print("  - reranker_type: bge-m3")
    print("  - llm_choice: gemini")
    print("  - use_query_enhancement: True")
    print("  - top_k: 10")
    print("  - limit: 1")
    print("  - save_to_db: True")

    try:
        results = backend.evaluate_all_metrics_batch(
            embedder_type="huggingface_local",
            reranker_type="bge-m3",
            llm_choice="gemini",
            use_query_enhancement=True,
            top_k=10,
            limit=1,  # Only test with 1 question
            save_to_db=True
        )

        print("✅ Batch evaluation completed!")
        print("\n📈 Results Summary:")

        # Print results for each metric
        for metric_name, metric_result in results.items():
            if isinstance(metric_result, dict):
                summary = metric_result.get('summary', {})
                print(f"\n🔹 {metric_name.upper()}:")
                if 'error' in metric_result:
                    print(f"   ❌ Error: {metric_result['error']}")
                else:
                    # Print key metrics
                    if metric_name == 'semantic_similarity':
                        print(f"   📊 Avg Semantic Similarity: {summary.get('avg_semantic_similarity', 'N/A')}")
                        print(f"   📊 Best Match Score: {summary.get('avg_best_match_score', 'N/A')}")
                        print(f"   📊 Chunks Above Threshold: {summary.get('total_chunks_above_threshold', 'N/A')}")
                    elif metric_name == 'recall':
                        print(f"   📊 Overall Recall: {summary.get('overall_recall', 'N/A')}")
                        print(f"   📊 Overall Precision: {summary.get('overall_precision', 'N/A')}")
                        print(f"   📊 F1 Score: {summary.get('overall_f1_score', 'N/A')}")
                    elif metric_name == 'relevance':
                        print(f"   📊 Avg Overall Relevance: {summary.get('avg_overall_relevance', 'N/A')}")
                        print(f"   📊 High Relevance Ratio: {summary.get('global_high_relevance_ratio', 'N/A')}")
                    elif metric_name == 'faithfulness':
                        print(f"   📊 Avg Faithfulness: {summary.get('avg_faithfulness', 'N/A')}")
                        print(f"   📊 Faithful Ratio (>0.5): {summary.get('global_faithful_ratio', 'N/A')}")

                    print(f"   📊 Processed: {summary.get('processed', 'N/A')}")
                    print(f"   📊 Errors: {summary.get('errors', 'N/A')}")

    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Batch evaluation failed: {e}"

    # Check if results were saved to database
    print("\n💾 Checking database storage...")

    try:
        # Check ground truth results (individual question results)
        gt_list = backend.get_ground_truth_list(limit=10)
        print(f"📊 Ground truth entries in DB: {len(gt_list)}")

        # Check if our test question has evaluation results
        test_gt_found = False
        for gt in gt_list:
            if gt.get('question') == test_question:
                test_gt_found = True
                print(f"✅ Test ground truth found with ID: {gt.get('id')}")
                print(f"   📝 Faithfulness: {gt.get('faithfulness', 'Not set')}")
                print(f"   📝 Relevance: {gt.get('relevance', 'Not set')}")
                print(f"   📝 Predicted Answer: {(gt.get('predicted_answer') or 'Not set')[:100]}...")
                print(f"   📝 Retrieval Chunks: {gt.get('retrieval_chunks', 'Not set')}")
                print(f"   📝 Evaluated At: {gt.get('evaluated_at', 'Not set')}")
                break

        if not test_gt_found:
            print("❌ Test ground truth not found in database")

        # Check metrics table (aggregated results)
        metrics = backend.db.get_metrics(limit=20)
        print(f"📊 Metrics entries in DB: {len(metrics)}")

        # Check for recent evaluation metrics
        recent_metrics = [m for m in metrics if m.get('embedder_model') == 'huggingface_local' and m.get('reranker_model') == 'bge-m3' and m.get('llm_model') == 'gemini']
        print(f"📊 Recent evaluation metrics (hf local + bge-m3 + gemini): {len(recent_metrics)}")

        if recent_metrics:
            latest = recent_metrics[0]  # Most recent first
            print("✅ Found recent evaluation metrics:")
            print(f"   📈 Faithfulness: {latest.get('faithfulness', 'N/A')}")
            print(f"   📈 Relevance: {latest.get('relevance', 'N/A')}")
            print(f"   📈 Recall: {latest.get('recall', 'N/A')}")
            print(f"   📈 Query: {latest.get('query', 'N/A')[:50]}...")
            print(f"   📈 Timestamp: {latest.get('timestamp', 'N/A')}")

        # Overall success check
        success = test_gt_found and len(recent_metrics) > 0

        if success:
            print("✅ Evaluation results successfully saved to database!")
            assert success
        else:
            print("❌ Evaluation results not found in database")
            assert False, "Evaluation results not found in database"

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    setup_logging()

    print("🧪 RAG Evaluation Storage Test")
    print("=" * 50)

    success = test_evaluation_storage()

    print("\n" + "=" * 50)
    if success:
        print("✅ Test PASSED: Evaluation data computed and stored successfully!")
    else:
        print("❌ Test FAILED: Issues with evaluation data computation/storage")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())