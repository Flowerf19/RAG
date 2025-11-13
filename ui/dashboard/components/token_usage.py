"""
Token Usage Component
Displays token usage statistics and visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from evaluation.backend_dashboard.api import BackendDashboard


class TokenUsageComponent:
    """Component for displaying token usage statistics and charts."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend

    def display_overview(self):
        """Display token usage overview in model comparison tab."""
        st.subheader("💰 Token Usage Overview")

        # Get token usage overview
        token_overview = self.backend.get_token_usage_overview()

        if token_overview['total_queries'] == 0:
            st.info("No token usage data available yet")
            return

        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", f"{token_overview['total_queries']:,}")

        with col2:
            st.metric("Total Tokens", f"{token_overview['total_tokens']:,}")

        with col3:
            st.metric("Avg Tokens/Query", f"{token_overview['avg_total_tokens']:.1f}")

        with col4:
            # Calculate cost estimate (rough estimate)
            costs = self.backend.get_token_costs()
            st.metric("Est. Total Cost", f"${costs['total_cost']:.4f}")

        # Token breakdown by component
        st.subheader("🔍 Token Usage by Component")

        component_data = {
            'Component': ['Embedding', 'Reranking', 'LLM'],
            'Total Tokens': [
                token_overview['total_embedding_tokens'],
                token_overview['total_reranking_tokens'],
                token_overview['total_llm_tokens']
            ],
            'Avg Tokens/Query': [
                token_overview['avg_embedding_tokens'],
                token_overview['avg_reranking_tokens'],
                token_overview['avg_llm_tokens']
            ]
        }

        df_components = pd.DataFrame(component_data)
        # Horizontal bar chart for better comparison
        fig = px.bar(df_components, y='Component', x='Total Tokens',
                    title='Token Usage by Component',
                    color='Component',
                    color_discrete_map={
                        'Embedding': '#1f77b4',
                        'Reranking': '#ff7f0e',
                        'LLM': '#2ca02c'
                    },
                    orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')

    def display_detailed(self):
        """Display detailed token usage analytics."""
        st.header("💰 Token Usage Analytics")

        # Get token usage data
        token_overview = self.backend.get_token_usage_overview()
        token_by_embedder = self.backend.get_token_usage_by_embedder()
        token_by_llm = self.backend.get_token_usage_by_llm()
        token_by_reranker = self.backend.get_token_usage_by_reranker()
        token_over_time = self.backend.get_token_usage_over_time()
        token_costs = self.backend.get_token_costs()

        # Overview metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Queries", f"{token_overview['total_queries']:,}")

        with col2:
            st.metric("Total Tokens", f"{token_overview['total_tokens']:,}")

        with col3:
            st.metric("Embedding Tokens", f"{token_overview['total_embedding_tokens']:,}")

        with col4:
            st.metric("LLM Tokens", f"{token_overview['total_llm_tokens']:,}")

        with col5:
            st.metric("Retrieval Chunks", f"{token_overview['total_retrieval_chunks']:,}")

        # Cost estimates
        st.subheader("💵 Estimated Costs")
        cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)

        with cost_col1:
            st.metric("Embedding Cost", f"${token_costs['embedding_cost']:.4f}")

        with cost_col2:
            st.metric("Reranking Cost", f"${token_costs['reranking_cost']:.4f}")

        with cost_col3:
            st.metric("LLM Cost", f"${token_costs['llm_cost']:.4f}")

        with cost_col4:
            st.metric("Total Cost", f"${token_costs['total_cost']:.4f}")

        st.caption(f"Cost per query: ${token_costs['cost_per_query']:.6f}")

        # Cost breakdown stacked bar chart
        st.subheader("💵 Cost Breakdown by Component")
        cost_data = {
            'Component': ['Embedding', 'Reranking', 'LLM'],
            'Cost': [token_costs['embedding_cost'], token_costs['reranking_cost'], token_costs['llm_cost']]
        }
        df_cost = pd.DataFrame(cost_data)
        fig = px.bar(df_cost, y='Component', x='Cost',
                    title='Cost Breakdown by Component',
                    color='Component',
                    color_discrete_map={
                        'Embedding': '#1f77b4',
                        'Reranking': '#ff7f0e',
                        'LLM': '#2ca02c'
                    },
                    orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')

        # Token usage by component tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔗 By Embedder", "🤖 By LLM", "📊 By Reranker", "📈 Over Time"])

        with tab1:
            self._display_embedder_tokens(token_by_embedder)

        with tab2:
            self._display_llm_tokens(token_by_llm)

        with tab3:
            self._display_reranker_tokens(token_by_reranker)

        with tab4:
            self._display_tokens_over_time(token_over_time)

    def _display_embedder_tokens(self, token_by_embedder):
        """Display token usage by embedder."""
        if token_by_embedder:
            df = pd.DataFrame(token_by_embedder)
            st.dataframe(df, width='stretch')

            # Token usage chart for embedders
            st.subheader("🔗 Token Usage by Embedder Model")
            fig = px.bar(df, y='model', x='total_tokens',
                       title='Total Tokens Used by Embedder',
                       labels={'total_tokens': 'Total Tokens', 'model': 'Embedder Model'},
                       color='total_tokens',
                       color_continuous_scale='blues',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')

            # Average tokens per query
            st.subheader("📊 Average Tokens per Query by Embedder")
            fig = px.bar(df, y='model', x='avg_total_tokens',
                       title='Average Tokens per Query by Embedder',
                       labels={'avg_total_tokens': 'Avg Tokens/Query', 'model': 'Embedder Model'},
                       color='avg_total_tokens',
                       color_continuous_scale='greens',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')

            # Retrieval chunks by embedder
            st.subheader("📦 Retrieval Chunks by Embedder")
            fig = px.bar(df, y='model', x='total_retrieval_chunks',
                       title='Total Retrieval Chunks by Embedder',
                       labels={'total_retrieval_chunks': 'Total Chunks', 'model': 'Embedder Model'},
                       color='total_retrieval_chunks',
                       color_continuous_scale='purples',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No embedder token usage data available")

    def _display_llm_tokens(self, token_by_llm):
        """Display token usage by LLM."""
        if token_by_llm:
            df = pd.DataFrame(token_by_llm)
            st.dataframe(df, width='stretch')

            # Token usage chart for LLMs
            st.subheader("🤖 Token Usage by LLM Model")
            fig = px.bar(df, y='model', x='total_llm_tokens',
                       title='LLM Tokens Used by Model',
                       labels={'total_llm_tokens': 'LLM Tokens', 'model': 'LLM Model'},
                       color='total_llm_tokens',
                       color_continuous_scale='reds',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')

            # Average tokens per query
            st.subheader("📊 Average LLM Tokens per Query")
            fig = px.bar(df, y='model', x='avg_llm_tokens',
                       title='Average LLM Tokens per Query by Model',
                       labels={'avg_llm_tokens': 'Avg LLM Tokens/Query', 'model': 'LLM Model'},
                       color='avg_llm_tokens',
                       color_continuous_scale='oranges',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No LLM token usage data available")

    def _display_reranker_tokens(self, token_by_reranker):
        """Display token usage by reranker."""
        if token_by_reranker:
            df = pd.DataFrame(token_by_reranker)
            st.dataframe(df, width='stretch')

            # Token usage chart for rerankers
            st.subheader("📊 Token Usage by Reranker Model")
            fig = px.bar(df, y='model', x='total_reranking_tokens',
                       title='Reranking Tokens Used by Model',
                       labels={'total_reranking_tokens': 'Reranking Tokens', 'model': 'Reranker Model'},
                       color='total_reranking_tokens',
                       color_continuous_scale='purples',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')

            # Average tokens per query
            st.subheader("📊 Average Reranking Tokens per Query")
            fig = px.bar(df, y='model', x='avg_reranking_tokens',
                       title='Average Reranking Tokens per Query by Model',
                       labels={'avg_reranking_tokens': 'Avg Reranking Tokens/Query', 'model': 'Reranker Model'},
                       color='avg_reranking_tokens',
                       color_continuous_scale='teals',
                       orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No reranker token usage data available")

    def _display_tokens_over_time(self, token_over_time):
        """Display token usage over time."""
        if token_over_time:
            df = pd.DataFrame(token_over_time)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Token usage over time
            st.subheader("📈 Token Usage Over Time")
            fig = px.line(df, x='timestamp', y=['embedding_tokens', 'reranking_tokens', 'llm_tokens', 'total_tokens'],
                        title='Token Usage Trends Over Time',
                        labels={'value': 'Tokens', 'timestamp': 'Time', 'variable': 'Token Type'},
                        color_discrete_map={
                            'embedding_tokens': 'blue',
                            'reranking_tokens': 'purple',
                            'llm_tokens': 'red',
                            'total_tokens': 'black'
                        })
            st.plotly_chart(fig, width='stretch')

            # Query count over time
            st.subheader("📊 Query Volume Over Time")
            fig = px.bar(df, x='timestamp', y='query_count',
                       title='Number of Queries Over Time',
                       labels={'query_count': 'Query Count', 'timestamp': 'Time'},
                       color='query_count',
                       color_continuous_scale='viridis')
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No time-series token usage data available")
