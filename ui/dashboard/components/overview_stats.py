"""
Overview Statistics Component
Displays key metrics and statistics for the RAG evaluation dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from evaluation.backend_dashboard.api import BackendDashboard


class OverviewStatsComponent:
    """Component for displaying overview statistics."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend

    def display(self):
        """Display overview statistics in header."""
        st.header("📈 Overview Statistics")

        stats = self.backend.get_overview_stats()

        # Main metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Queries", f"{stats['total_queries']:,}")

        with col2:
            st.metric("Avg Accuracy", f"{stats['avg_accuracy']:.3f}")

        with col3:
            st.metric("Avg Faithfulness", f"{stats.get('avg_faithfulness', 0):.3f}")

        with col4:
            st.metric("Avg Relevance", f"{stats.get('avg_relevance', 0):.3f}")

        with col5:
            st.metric("Avg Recall", f"{stats.get('avg_recall', 0):.3f}")

        with col6:
            st.metric("Avg Latency", f"{stats['avg_latency']:.3f}s")