import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard


class RecentActivityComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self, limit: int = 25):
        st.header("Recent Activity")
        # Backend provides `get_recent_metrics` for recent metric records
        rows = self.backend.get_recent_metrics(limit=limit)
        if not rows:
            st.info("No recent activity available.")
            return

        df = pd.DataFrame(rows)

        # Keep only useful columns if present. `get_recent_metrics` returns metric records
        # which typically include: id, query, model, latency, error_rate, metadata, created_at
        candidate_cols = ['id', 'query', 'model', 'latency', 'error_rate', 'status', 'error', 'created_at', 'user', 'session_id', 'metadata']
        display_cols = [c for c in candidate_cols if c in df.columns]
        if not display_cols:
            st.dataframe(df.head(limit), use_container_width=True)
            return

        st.write(f"Showing {len(df):,} recent rows (limit {limit})")
        # If created_at exists, sort by it; otherwise show raw order
        if 'created_at' in df.columns:
            # convert to datetime if needed
            try:
                df['created_at'] = pd.to_datetime(df['created_at'])
            except Exception:
                pass
            out_df = df[display_cols].sort_values('created_at', ascending=False).head(limit)
        else:
            out_df = df[display_cols].head(limit)

        st.dataframe(out_df, use_container_width=True)
