import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard


class TokenUsageComponent:
    """Minimal token usage component: total, avg, simple breakdown."""

    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Token Usage")
        token_stats = self.backend.get_token_usage_stats()
        st.metric("Total Tokens", f"{token_stats.get('total_tokens', 0):,}")
        st.metric("Avg Tokens / Query", f"{token_stats.get('avg_total_tokens', 0):.1f}")

        df = pd.DataFrame([
            {'Component': 'Embedding', 'Tokens': token_stats.get('total_embedding_tokens', 0)},
            {'Component': 'Reranking', 'Tokens': token_stats.get('total_reranking_tokens', 0)},
            {'Component': 'LLM', 'Tokens': token_stats.get('total_llm_tokens', 0)}
        ])

        if not df.empty:
            st.bar_chart(df.set_index('Component'))
        else:
            st.info("No token breakdown data available")

