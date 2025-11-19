#!/usr/bin/env python3
"""
Test RAG Metrics Visualizations
==============================

Test script for the RAG metrics visualization system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging

# Import visualization modules
from evaluation.visualizations import RAGMetricsVisualizer
from evaluation.visualizations.utils.table_output import print_metrics_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sample_data():
    """Test with sample data."""
    print("🧪 Testing RAG Metrics Visualizations with Sample Data")
    print("=" * 60)

    # Create sample data
    sample_data = {
        'Configuration': [
            'Local + No Re-ranking',
            'API + Query Rewrite + Re-ranking',
            'Local + API Hybrid + No Rewrite'
        ],
        'Faithfulness': [1.000, 0.950, 0.980],
        'Context_Recall': [1.000, 0.920, 0.990],
        'Context_Relevance': [1.000, 0.940, 0.970],
        'Answer_Relevancy': [1.000, 0.960, 0.985]
    }
    df = pd.DataFrame(sample_data)

    # Print table
    print_metrics_table(df, "Sample RAG Metrics Comparison")

    # Create visualizer
    visualizer = RAGMetricsVisualizer("data/test_visualizations")

    # Generate all charts (save but don't show)
    results = visualizer.generate_all_charts(df, "Test RAG Evaluation", save_charts=True, show_charts=False)

    print("\n📊 Generated Visualizations:")
    for chart_type, path in results.items():
        if chart_type != "error":
            print(f"✅ {chart_type}: {path}")
        else:
            print(f"❌ Error: {path}")

    assert "error" not in results, f"Visualization generation failed: {results.get('error', 'Unknown error')}"


def test_from_ragas_output():
    """Test with mock Ragas output."""
    print("\n🧪 Testing with Mock Ragas Output")
    print("=" * 40)

    # Mock Ragas summary output
    mock_ragas_summary = {
        'faithfulness': {'mean': 0.875, 'scores': [0.9, 0.85]},
        'context_recall': {'mean': 1.000, 'scores': [1.0, 1.0]},
        'context_relevance': {'mean': 0.950, 'scores': [0.95, 0.95]},
        'answer_relevancy': {'mean': 0.920, 'scores': [0.93, 0.91]}
    }

    visualizer = RAGMetricsVisualizer("data/test_visualizations")

    results = visualizer.visualize_from_ragas_output(
        mock_ragas_summary,
        config_name="Mock Ollama Config",
        title_prefix="Mock Ragas Test"
    )

    print("📊 Generated Visualizations from Ragas:")
    for chart_type, path in results.items():
        if chart_type != "error":
            print(f"✅ {chart_type}: {path}")
        else:
            print(f"❌ Error: {path}")

    assert "error" not in results, f"Ragas visualization failed: {results.get('error', 'Unknown error')}"


def test_configuration_comparison():
    """Test comparing multiple configurations."""
    print("\n🧪 Testing Configuration Comparison")
    print("=" * 40)

    # Mock evaluation results for different configs
    mock_results = [
        {
            'faithfulness': 0.95,
            'context_recall': 0.98,
            'context_relevance': 0.92,
            'answer_relevancy': 0.94
        },
        {
            'faithfulness': 0.88,
            'context_recall': 1.00,
            'context_relevance': 0.95,
            'answer_relevancy': 0.89
        },
        {
            'faithfulness': 0.92,
            'context_recall': 0.96,
            'context_relevance': 0.98,
            'answer_relevancy': 0.91
        }
    ]

    config_names = ['Config A', 'Config B', 'Config C']

    visualizer = RAGMetricsVisualizer("data/test_visualizations")

    results = visualizer.compare_configurations(
        mock_results,
        config_names,
        title_prefix="Multi-Config Comparison Test"
    )

    print("📊 Generated Comparison Visualizations:")
    for chart_type, path in results.items():
        if chart_type != "error":
            print(f"✅ {chart_type}: {path}")
        else:
            print(f"❌ Error: {path}")

    assert "error" not in results, f"Configuration comparison failed: {results.get('error', 'Unknown error')}"


def main():
    """Run all tests."""
    print("🚀 RAG Metrics Visualization Test Suite")
    print("=" * 50)

    try:
        # Test 1: Sample data
        results1 = test_sample_data()

        # Test 2: From Ragas output
        results2 = test_from_ragas_output()

        # Test 3: Configuration comparison
        results3 = test_configuration_comparison()

        # Summary
        print("\n🎉 Test Suite Completed!")
        print("=" * 30)

        total_charts = sum(len([k for k in r.keys() if k != "error"])
                          for r in [results1, results2, results3] if "error" not in r)

        print(f"📊 Total visualizations generated: {total_charts}")
        print("📁 Check data/test_visualizations/ for output files")

        if any("error" in r for r in [results1, results2, results3]):
            print("⚠️  Some tests had errors - check logs above")
        else:
            print("✅ All tests passed successfully!")

    except Exception as e:
        logger.exception(f"Test suite failed: {e}")
        print(f"❌ Test suite failed: {e}")


if __name__ == "__main__":
    main()