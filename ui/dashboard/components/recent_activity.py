import streamlit as st
import pandas as pd
import logging
from evaluation.backend_dashboard.api import BackendDashboard


class RecentActivityComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend
        try:
            self.logger = logging.getLogger(self.__class__.__module__)
        except Exception:
            self.logger = logging.getLogger(__name__)

    def display(self, limit: int = 25):
        st.header("Recent Activity")
        # Backend provides `get_recent_metrics` for recent metric records
        rows = self.backend.get_recent_metrics(limit=limit)
        try:
            self.logger.info("RecentActivity: fetched %d rows (limit=%d)", len(rows) if rows else 0, limit)
        except Exception:
            pass
        if not rows:
            st.info("No recent activity available.")
            return

        df = pd.DataFrame(rows)

        # Keep only useful columns if present. `get_recent_metrics` returns metric records
        # which typically include: id, query, model, latency, error_rate, metadata, created_at
        candidate_cols = ['id', 'query', 'model', 'latency', 'error_rate', 'status', 'error', 'created_at', 'user', 'session_id', 'metadata']
        display_cols = [c for c in candidate_cols if c in df.columns]
        if not display_cols:
            # Show up to `limit` rows but also indicate total available
            total_rows = len(df)
            shown = min(total_rows, limit)
            st.write(f"Showing {shown} of {total_rows} recent rows (limit {limit})")
            st.dataframe(df.head(limit), width='stretch')
            return

        total_rows = len(df)
        # If created_at exists, sort by it; otherwise show raw order
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

        # Show counts: displayed vs total available
        displayed_count = len(out_df)
        st.write(f"Showing {displayed_count} of {total_rows} recent rows (limit {limit})")
        st.dataframe(out_df, width='stretch')

        # Add a developer button to print recent activity to terminal/log
        if st.button("Print recent activity to terminal"):
            try:
                self.logger.info("--- Recent Activity (top %d rows) ---", len(out_df))
                for i, r in enumerate(out_df.to_dict(orient='records'), start=1):
                    # Log a concise representation per row
                    self.logger.info("%d: id=%s model=%s query=%s latency=%s error=%s created_at=%s",
                                     i,
                                     r.get('id'),
                                     r.get('model'),
                                     (r.get('query') or '')[:120],
                                     r.get('latency'),
                                     r.get('error'),
                                     r.get('created_at'))
                st.success(f"Printed {len(out_df)} recent rows to terminal/log.")
            except Exception as e:
                st.error(f"Failed to print recent activity: {e}")
                try:
                    self.logger.exception("Failed to print recent activity: %s", e)
                except Exception:
                    pass
