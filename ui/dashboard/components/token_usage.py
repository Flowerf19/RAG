import streamlit as st
import pandas as pd
import altair as alt
from evaluation.backend_dashboard.api import BackendDashboard


class TokenUsageComponent:
    """Token usage component: overview and stacked per-model breakdown."""

    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Token Usage")
        token_stats = self.backend.get_token_usage_stats()
        st.metric("Total Tokens", f"{token_stats.get('total_tokens', 0):,}")
        st.metric("Avg Tokens / Query", f"{token_stats.get('avg_total_tokens', 0):.1f}")

        # Attempt to fetch per-model breakdown (prefer embedder-level then llm)
        try:
            per_model = self.backend.get_token_usage_by_embedder()
        except Exception:
            try:
                per_model = self.backend.get_token_usage_by_llm()
            except Exception:
                per_model = []

        if per_model:
            df = pd.DataFrame(per_model)
            # Expect columns like total_embedding_tokens, total_reranking_tokens, total_llm_tokens
            value_cols = [c for c in ['total_embedding_tokens', 'total_reranking_tokens', 'total_llm_tokens'] if c in df.columns]
            if value_cols:
                melt = df.melt(id_vars=['model'], value_vars=value_cols, var_name='component', value_name='tokens')
                # Clean component names
                melt['component'] = melt['component'].str.replace('total_', '').str.replace('_tokens', '').str.replace('_', ' ').str.title()
                chart = alt.Chart(melt).mark_bar().encode(
                    y=alt.Y('model:N', sort=alt.EncodingSortField(field='tokens', op='sum', order='descending'), title='Model'),
                    x=alt.X('tokens:Q', title='Tokens'),
                    color=alt.Color('component:N', title='Component'),
                    tooltip=['model', 'component', 'tokens']
                ).properties(height=420)
                st.altair_chart(chart, use_container_width=True)
                return

        # Fallback simple breakdown
        df_simple = pd.DataFrame([
            {'Component': 'Embedding', 'Tokens': token_stats.get('total_embedding_tokens', 0)},
            {'Component': 'Reranking', 'Tokens': token_stats.get('total_reranking_tokens', 0)},
            {'Component': 'LLM', 'Tokens': token_stats.get('total_llm_tokens', 0)}
        ])

        if not df_simple.empty:
            st.bar_chart(df_simple.set_index('Component'))
        else:
            st.info("No token breakdown data available")

