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

from evaluation.backend_dashboard.api import BackendDashboard
from ui.dashboard.components import (
    ModelPerformanceComponent,
    RecentActivityComponent,
    TokenUsageComponent,
    GroundTruthComponent
)

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

        # Initialize components
        self.model_performance = ModelPerformanceComponent(self.backend)
        self.recent_activity = RecentActivityComponent(self.backend)
        self.token_usage = TokenUsageComponent(self.backend)
        self.ground_truth = GroundTruthComponent(self.backend)

    def run(self):
        """Run the dashboard application."""
        st.title("RAG Evaluation Dashboard")
        st.markdown("Compare performance and accuracy across different RAG models")

        # Add refresh button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("Refresh Data"):
                self.backend.refresh_data()
                # Force refresh all cached component data
                st.session_state["force_refresh_recent"] = True
                st.session_state["force_refresh_overview"] = True
                st.session_state["force_refresh_performance"] = True
                st.rerun()

        with col2:
            # Placeholder for future model filter
            st.empty()

        # Display components
        self.model_performance.display()
        self.recent_activity.display()
        self.ground_truth.display()


        # Optional: Display detailed token usage if needed
        # self.token_usage.display_detailed()


def main():
    """Main entry point for the dashboard."""
    dashboard = RAGEvaluationDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()