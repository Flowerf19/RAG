import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

                # Create stacked horizontal bar chart per model
                try:
                    models = list(melt['model'].unique())
                    components = list(melt['component'].unique())
                    # Prepare values matrix
                    vals_by_comp = {}
                    for comp in components:
                        comp_series = melt[melt['component'] == comp].set_index('model')['tokens']
                        comp_vals = [float(comp_series.get(m, 0)) for m in models]
                        vals_by_comp[comp] = comp_vals

                    fig, ax = plt.subplots(figsize=(8, max(2, 0.6 * len(models))))
                    left = [0.0] * len(models)
                    for comp in components:
                        vals = vals_by_comp[comp]
                        ax.barh(models, vals, left=left, label=comp)
                        left = [l + v for l, v in zip(left, vals)]

                    ax.set_xlabel('Tokens')
                    ax.set_ylabel('Model')
                    ax.set_title('Token Usage by Model and Component')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not render token breakdown chart: {e}")
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

