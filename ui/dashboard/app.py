"""
RAG Evaluation Dashboard
Streamlit app for visualizing and comparing RAG model performance.
Refactored to use modular components for better maintainability.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from evaluation.backend_dashboard.api import BackendDashboard
from evaluation.visualizations.visualizer import RAGMetricsVisualizer
from ui.dashboard.components import GroundTruthComponent

# Simple development logging setup
if os.getenv("RAG_DEV_LOGGING", "1") != "0":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s"
    )

import streamlit as st



class RAGEvaluationDashboard:
    """Streamlit dashboard for RAG model evaluation and comparison."""

    def __init__(self):
        """Initialize dashboard with backend API and components."""
        self.backend = BackendDashboard()
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",
            page_icon="📊",
            layout="wide"
        )

        # Initialize only essential components
        self.ground_truth = GroundTruthComponent(self.backend)

    def run(self):
        """Run the dashboard application."""
        st.title("RAG Evaluation Dashboard")
        st.markdown("Compare performance and accuracy across different RAG models")

        # Display charts immediately at the top
        self._display_charts_section()

        # Add refresh button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🔄 Refresh Data"):
                self.backend.refresh_data()
                st.rerun()

        with col2:
            if st.button("🚀 Auto Test (5 samples)"):
                self._run_auto_test()

        # Display simplified ground truth evaluation
        self._display_evaluation_section()


    def _display_charts_section(self):
        """Display charts immediately when dashboard loads."""
        # Show overview stats first
        stats = self.backend.get_overview_stats()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📊 Total Queries", stats.get('total_queries', 0))
        col2.metric("🎯 Ground Truth", stats.get('model_count', 0))
        col3.metric("⚡ Avg Latency", f"{stats.get('avg_latency', 0):.2f}s")
        col4.metric("🎯 Answer Correctness", f"{stats.get('avg_relevance', 0):.3f}")
        col5.metric("❌ Error Rate", f"{stats.get('error_rate', 0):.1f}%")

        st.header("📊 Model Performance Charts")

        try:
            # Get data from database
            stats = self.backend.get_model_comparison_data()
            df = pd.DataFrame(stats['llm'])

            if df.empty:
                st.info("📊 No evaluation data available yet.")
                st.markdown("""
                **To get started:**
                1. Click **🚀 Auto Test** to run evaluation on 5 sample questions
                2. Or upload your own ground truth data in the **Advanced Evaluation Options** below
                """)
                return

            # Show data table
            with st.expander("📋 View Raw Data", expanded=False):
                st.dataframe(df[['model', 'accuracy', 'faithfulness', 'relevance', 'recall', 'latency', 'error_rate']])

            # Prepare DataFrame for visualizer
            viz_df = pd.DataFrame({
                'Configuration': df['model'],
                'Faithfulness': df.get('faithfulness', pd.Series([0]*len(df))).astype(float),
                'Context_Recall': df.get('recall', pd.Series([0]*len(df))).astype(float),
                'Answer_Correctness': df.get('relevance', pd.Series([0]*len(df))).astype(float),
                'Answer_Relevancy': df.get('answer_relevancy', df.get('relevance', pd.Series([0]*len(df)))).astype(float)
            })

            # Generate and display charts
            visualizer = RAGMetricsVisualizer(output_dir="data/visualizations")
            viz_results = visualizer.generate_all_charts(viz_df, title_prefix="Model Performance", save_charts=True, show_charts=False)

            if "error" in viz_results:
                st.warning(f"Could not generate visualizations: {viz_results['error']}")
            else:
                # Display charts in a grid
                chart_files = {k: v for k, v in viz_results.items() if k not in ['table', 'error']}
                if chart_files:
                    cols = st.columns(2)
                    for i, (chart_type, chart_path) in enumerate(chart_files.items()):
                        with cols[i % 2]:
                            st.subheader(f"📈 {chart_type.replace('_', ' ').title()}")
                            try:
                                st.image(chart_path, width='stretch')
                            except Exception as e:
                                st.error(f"Could not load chart {chart_type}: {e}")
                else:
                    st.info("No charts available.")

        except Exception as e:
            st.error(f"Failed to load charts: {e}")


    def _run_auto_test(self):
        """Run automatic test with 5 samples."""
        with st.spinner("🚀 Running auto test with 5 ground truth samples using Ollama LLM..."):
            try:
                # Use Ollama for more reliable testing
                res = self.backend.evaluate_ground_truth_with_ragas(
                    llm_provider='ollama',
                    model_name='gemma3:1b',
                    limit=5,
                    save_to_db=True,
                    generate_visualizations=True
                )
                st.success(f"✅ Auto test completed! Processed {len(res.get('results', []))} samples.")
                st.rerun()  # Refresh to show updated charts
            except Exception as e:
                st.error(f"❌ Auto test failed: {e}")
                st.info("💡 Make sure Ollama is running with 'gemma3:1b' model: `ollama pull gemma3:1b`")


    def _display_evaluation_section(self):
        """Display simplified evaluation controls."""
        st.header("🧪 Manual Evaluation")

        with st.expander("Advanced Evaluation Options", expanded=False):
            # Keep the full ground truth component for advanced users
            self.ground_truth.display()


def main():
    """Main entry point for the dashboard."""
    dashboard = RAGEvaluationDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()