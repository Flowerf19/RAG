#!/usr/bin/env python3
"""
Demo: Complete RAG Evaluation with Visualizations
================================================

Demonstrates the full RAG evaluation pipeline with automatic visualization generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from evaluation.backend_dashboard.api import BackendDashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_full_evaluation_with_visualizations():
    """Demo full RAG evaluation with automatic visualizations."""
    print("🚀 Demo: Complete RAG Evaluation with Visualizations")
    print("=" * 60)

    # Initialize backend
    backend = BackendDashboard()

    # Check if we have ground truth data
    gt_data = backend.get_ground_truth_list(limit=5)
    if not gt_data:
        print("❌ No ground truth data found. Please upload some data first via the UI dashboard.")
        return

    print(f"✅ Found {len(gt_data)} ground truth samples")

    # Run evaluation with visualizations
    print("\n🔍 Running Ragas evaluation with Ollama LLM...")
    print("(This may take several minutes)")

    results = backend.evaluate_ground_truth_with_ragas(
        llm_provider='ollama',
        model_name='gemma3:1b',
        limit=5,  # Limit for demo
        save_to_db=True,
        generate_visualizations=True
    )

    if "error" in results:
        print(f"❌ Evaluation failed: {results['error']}")
        return

    # Display results
    print("\n📊 Evaluation Results:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    # Check visualizations
    if "visualizations" in results:
        viz = results["visualizations"]
        print(f"\n📈 Generated {len(viz)} visualizations:")
        for viz_type, path in viz.items():
            print(f"   ✅ {viz_type}: {path}")

        print("\n💡 To view visualizations:")
        print("   1. Check data/visualizations/ directory")
        print("   2. Or run the UI dashboard: streamlit run ui/dashboard/app.py")
    else:
        print("\n⚠️ Visualizations were not generated")

    print("\n🎉 Demo completed successfully!")
    print("The RAG system now includes comprehensive evaluation with automatic visualizations.")


def demo_visualization_only():
    """Demo visualization generation from existing results."""
    print("\n🧪 Demo: Generate Visualizations from Sample Data")
    print("=" * 50)

    from evaluation.visualizations import RAGMetricsVisualizer

    # Create sample data
    import pandas as pd
    sample_data = {
        'Configuration': [
            'Local Ollama + No Re-ranking',
            'Gemini API + Query Rewrite',
            'Hybrid Local-API + Re-ranking'
        ],
        'Faithfulness': [0.95, 0.88, 0.92],
        'Context_Recall': [1.00, 0.94, 0.98],
        'Context_Relevance': [0.97, 0.89, 0.95],
        'Answer_Relevancy': [0.96, 0.91, 0.93]
    }
    df = pd.DataFrame(sample_data)

    # Generate visualizations
    visualizer = RAGMetricsVisualizer("data/demo_visualizations")

    print("Generating visualizations...")
    results = visualizer.generate_all_charts(
        df,
        "Demo RAG Configuration Comparison",
        save_charts=True,
        show_charts=False
    )

    if "error" in results:
        print(f"❌ Visualization failed: {results['error']}")
        return

    print(f"✅ Generated {len(results)} visualizations:")
    for viz_type, path in results.items():
        print(f"   📊 {viz_type}: {path}")

    print("\n📁 Check data/demo_visualizations/ for all output files")


def main():
    """Run demos."""
    try:
        # Demo 1: Full evaluation with visualizations
        demo_full_evaluation_with_visualizations()

        # Demo 2: Visualization generation only
        demo_visualization_only()

        print("\n🎊 All demos completed!")
        print("\nNext steps:")
        print("1. Upload ground truth data via UI: streamlit run ui/dashboard/app.py")
        print("2. Run evaluations to see automatic visualizations")
        print("3. Compare different RAG configurations")

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()