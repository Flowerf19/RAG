"""
Recent Activity Component
Displays recent evaluation activity.
"""

import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard


class RecentActivityComponent:
    """Component for displaying recent evaluation activity."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend

    def display(self):
        """Display recent evaluation activity."""
        st.header("📋 Recent Activity")

        recent_metrics = self.backend.get_recent_metrics(limit=50)

        if recent_metrics:
            # Convert to DataFrame for display
            df = pd.DataFrame(recent_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Create detailed model description
            def format_model_info(row):
                llm = row.get('llm_model', 'Unknown')
                embedder = row.get('embedder_model', 'Unknown')
                reranker = row.get('reranker_model', 'none')
                qe = "QE" if row.get('query_enhanced', False) else "NoQE"
                return f"{llm} + {embedder} + {reranker} + {qe}"

            df['model_config'] = df.apply(format_model_info, axis=1)

            # Select and reorder columns
            display_cols = ['timestamp', 'model_config', 'query', 'latency', 'faithfulness', 'relevance', 'error']
            df_display = df[display_cols].copy()
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_display = df_display.head(10)

            st.dataframe(df_display, width='stretch')

            # Show summary stats
            total_recent = len(recent_metrics)
            error_recent = sum(1 for m in recent_metrics if m.get('error', False))
            avg_latency_recent = sum(m.get('latency', 0) for m in recent_metrics) / total_recent if total_recent > 0 else 0

            st.caption(f"Showing last 10 of {total_recent} recent evaluations | "
                      f"Avg Latency: {avg_latency_recent:.3f}s | Error Rate: {error_recent/total_recent*100:.1f}%")
        else:
            st.info("No recent evaluation data available")