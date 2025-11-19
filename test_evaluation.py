#!/usr/bin/env python3
"""
Test script for ground truth evaluation
"""

print('=== Testing Ground Truth Ragas Evaluation ===')
try:
    from evaluation.backend_dashboard.api import BackendDashboard
    backend = BackendDashboard()

    print('Running Ragas evaluation...')
    result = backend.evaluate_ground_truth_with_ragas(
        limit=2,  # Test with just 2 questions first
        save_to_db=False
    )

    print('✅ Ragas evaluation completed')
    summary = result.get('summary', {})
    print(f'Processed: {len(result.get("results", []))} questions')
    print(f'Average faithfulness: {summary.get("faithfulness", 0):.4f}')
    print(f'Average context recall: {summary.get("context_recall", 0):.4f}')
    print(f'Average context relevance: {summary.get("context_relevance", 0):.4f}')

    print('✅ Ragas evaluation test completed successfully!')

except Exception as e:
    print(f'❌ Ragas evaluation test failed: {e}')
    import traceback
    traceback.print_exc()