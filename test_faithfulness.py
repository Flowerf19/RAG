#!/usr/bin/env python3
"""
Test script for faithfulness evaluation functionality.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

import pytest

from evaluation.backend_dashboard.api import BackendDashboard

def test_faithfulness_evaluation():
    """Test Ragas evaluation with ground truth data."""
    pytest.skip("This test performs actual API calls and is too slow for CI")
    """Test Ragas evaluation with ground truth data."""
    print("Testing Ragas evaluation...")

    # Initialize backend dashboard
    dashboard = BackendDashboard()

    # Run Ragas evaluation
    result = dashboard.evaluate_ground_truth_with_ragas(
        limit=3,  # Test with first 3 questions for speed
        save_to_db=True
    )

    print("Ragas evaluation completed!")
    print(f"Summary: {result.get('summary', {})}")
    print(f"Number of results: {len(result.get('results', []))}")
    print(f"Errors: {len(result.get('errors_list', []))}")

    if result.get('errors_list'):
        print("Errors encountered:")
        for error in result['errors_list'][:3]:  # Show first 3 errors
            print(f"  - {error}")

    # Show sample results
    if result.get('results'):
        print("\nSample results:")
        for i, res in enumerate(result['results'][:3]):  # Show first 3 results
            print(f"  Question {i+1}: Faithfulness={res.get('faithfulness', 0):.3f}, Context Recall={res.get('context_recall', 0):.3f}, Context Relevance={res.get('context_relevance', 0):.3f}")
            print(f"    Generated answer: {res.get('generated_answer', '')[:100]}...")

    return None

if __name__ == "__main__":
    test_faithfulness_evaluation()