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
import os

# Development logging override: enable terminal logging for key modules
# Set `RAG_DEV_LOGGING=0` to disable this behavior in other environments.
if os.getenv("RAG_DEV_LOGGING", "1") != "0":
    root_logger = logging.getLogger()
    # Add a stream handler if none present (Streamlit may remove handlers on import)
    if not root_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(ch)
    root_logger.setLevel(logging.INFO)
    # Ensure important package loggers propagate to root so they appear in terminal
    for name in ("pipeline", "evaluation", "embedders", "llm", "BM25", "ui"):
        try:
            lg = logging.getLogger(name)
            lg.propagate = True
            lg.setLevel(logging.INFO)
        except Exception:
            pass

import streamlit as st


def _force_dev_logging_after_streamlit():
    """Reapply development logging configuration after Streamlit import.

    Streamlit may reconfigure loggers on import; call this to ensure project
    logs propagate to the terminal and a StreamHandler is attached.
    """
    import sys
    root_logger = logging.getLogger()

    # Force basicConfig with a stdout handler to guarantee we have a handler
    try:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logging.basicConfig(level=logging.INFO, handlers=[stdout_handler], force=True)
    except Exception:
        pass

    # Ensure project-related loggers propagate and have proper level
    prefixes = ("pipeline", "evaluation", "embedders", "llm", "BM25", "ui")
    mgr = logging.root.manager
    for name in list(mgr.loggerDict.keys()):
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                try:
                    lg = logging.getLogger(name)
                    lg.propagate = True
                    if not lg.handlers:
                        lg.addHandler(stdout_handler)
                    lg.setLevel(logging.INFO)
                except Exception:
                    pass


# Reapply logging configuration after Streamlit import to counteract
# Streamlit's logger reconfiguration.
_force_dev_logging_after_streamlit()

from evaluation.backend_dashboard.api import BackendDashboard
from ui.dashboard.components import (
    OverviewStatsComponent,
    ModelPerformanceComponent,
    PerformanceChartsComponent,
    RecentActivityComponent,
    TokenUsageComponent,
    GroundTruthComponent
)


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
        self.overview_stats = OverviewStatsComponent(self.backend)
        self.model_performance = ModelPerformanceComponent(self.backend)
        self.performance_charts = PerformanceChartsComponent(self.backend)
        self.recent_activity = RecentActivityComponent(self.backend)
        self.token_usage = TokenUsageComponent(self.backend)
        self.ground_truth = GroundTruthComponent(self.backend)

    def run(self):
        """Run the dashboard application."""
        st.title("🚀 RAG Evaluation Dashboard")
        st.markdown("Compare performance and accuracy across different RAG models")

        # Add refresh button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🔄 Refresh Data"):
                self.backend.refresh_data()
                st.rerun()

        with col2:
            # Placeholder for future model filter
            st.empty()

        # Display components
        self.overview_stats.display()
        self.model_performance.display()
        self.performance_charts.display()
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