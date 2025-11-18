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
    OverviewStatsComponent,
    ModelPerformanceComponent,
    PerformanceChartsComponent,
    RecentActivityComponent,
    TokenUsageComponent,
    GroundTruthComponent
)

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

    # Ensure a single stdout StreamHandler exists to avoid duplicates.
    try:
        # Reuse an existing StreamHandler writing to stdout if present
        stdout_handler = None
        root_handlers = logging.getLogger().handlers
        for h in root_handlers:
            if isinstance(h, logging.StreamHandler):
                stdout_handler = h
                break

        if stdout_handler is None:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            logging.getLogger().addHandler(stdout_handler)

        # Set root level but avoid calling basicConfig(force=True) which can
        # reattach handlers in some environments (Streamlit). Adjust level
        # on the root logger instead.
        logging.getLogger().setLevel(logging.INFO)
    except Exception:
        pass

    # Ensure project-related loggers propagate and have proper level without
    # adding duplicate handlers on individual loggers.
    prefixes = ("pipeline", "evaluation", "embedders", "llm", "BM25", "ui")
    mgr = logging.root.manager
    for name in list(mgr.loggerDict.keys()):
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                try:
                    lg = logging.getLogger(name)
                    lg.propagate = True
                    # Do not attach handlers to child loggers; rely on root
                    # handler to emit records and just set an appropriate level.
                    lg.setLevel(logging.INFO)
                except Exception:
                    pass


# Reapply logging configuration after Streamlit import to counteract
# Streamlit's logger reconfiguration.
_force_dev_logging_after_streamlit()

# Debug: emit a single info log showing how many handlers are attached to the
# root logger and their types. This helps verify the duplicate-handler issue
# without repeatedly attaching new handlers.
try:
    root_logger = logging.getLogger()
    handler_types = [type(h).__name__ for h in root_logger.handlers]
    root_logger.info("[DEBUG] root handlers=%d types=%s", len(root_logger.handlers), handler_types)
except Exception:
    pass



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