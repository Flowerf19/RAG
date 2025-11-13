"""
Model Comparison Component
Displays comparison tables and charts for different model types.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from evaluation.backend_dashboard.api import BackendDashboard


class ModelComparisonComponent:
    """Component for displaying model comparison data."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend

    def display(self):
        """Display comparison tables for different model types."""
        st.header("🔍 Model Comparison")

        comparison_data = self.backend.get_model_comparison_data()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🤖 LLM Models",
            "🔗 Embedding Models",
            "📊 Reranking Models",
            "✨ Query Enhancement",
            "💰 Token Usage",
            "📥 Ground Truth"
        ])

        with tab1:
            self._display_llm_tab(comparison_data)

        with tab2:
            self._display_embedding_tab(comparison_data)

        with tab3:
            self._display_reranking_tab(comparison_data)

        with tab4:
            self._display_query_enhancement_tab(comparison_data)

        with tab5:
            self._display_token_usage_tab()

        with tab6:
            self._display_ground_truth_tab()

    def _add_radar_chart(self, df, title_prefix):
        """Add radar chart for multi-metric comparison."""
        import plotly.graph_objects as go

        # Check if we have multiple metrics
        metrics_cols = ['faithfulness', 'relevance', 'recall', 'accuracy']
        available_metrics = [col for col in metrics_cols if col in df.columns]

        if len(available_metrics) >= 2 and len(df) > 1:
            st.subheader(f"🕸️ {title_prefix} - Multi-Metric Comparison")

            # Create radar chart
            fig = go.Figure()

            for _, row in df.iterrows():
                model_name = row.get('model', f'Model {_}')
                values = [row.get(metric, 0) for metric in available_metrics]
                values.append(values[0])  # Close the radar

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=available_metrics + [available_metrics[0]],
                    fill='toself',
                    name=model_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"{title_prefix} Performance Radar"
            )

            st.plotly_chart(fig, width='stretch')

    def _display_llm_tab(self, comparison_data):
        """Display LLM models comparison."""
        if comparison_data['llm']:
            df = pd.DataFrame(comparison_data['llm'])
            st.dataframe(df, width='stretch')

            # Add accuracy chart for LLM models
            llm_accuracy = self.backend.get_llm_accuracy()
            if llm_accuracy:
                st.subheader("🎯 LLM Model Accuracy Comparison")
                df_acc = pd.DataFrame(llm_accuracy)

                # Horizontal bar chart for better comparison
                fig = px.bar(df_acc, y='model', x='accuracy',
                           title='LLM Model Accuracy Comparison',
                           labels={'accuracy': 'Accuracy Score', 'model': 'LLM Model'},
                           color='accuracy',
                           color_continuous_scale='viridis',
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, width='stretch')

                # Add radar chart for multi-metric comparison if available
                self._add_radar_chart(df_acc, 'LLM Models')
        else:
            st.info("No LLM model data available")

    def _display_embedding_tab(self, comparison_data):
        """Display embedding models comparison."""
        if comparison_data['embedding']:
            df = pd.DataFrame(comparison_data['embedding'])
            st.dataframe(df, width='stretch')

            # Add accuracy chart for embedding models
            embedder_accuracy = self.backend.get_embedder_accuracy()
            if embedder_accuracy:
                st.subheader("🎯 Embedding Model Accuracy Comparison")
                df_acc = pd.DataFrame(embedder_accuracy)

                # Horizontal bar chart for better comparison
                fig = px.bar(df_acc, y='model', x='accuracy',
                           title='Embedding Model Accuracy Comparison',
                           labels={'accuracy': 'Accuracy Score', 'model': 'Embedding Model'},
                           color='accuracy',
                           color_continuous_scale='viridis',
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, width='stretch')

                # Add radar chart for multi-metric comparison if available
                self._add_radar_chart(df_acc, 'Embedding Models')
        else:
            st.info("No embedding model data available")

    def _display_reranking_tab(self, comparison_data):
        """Display reranking models comparison."""
        if comparison_data['reranking']:
            df = pd.DataFrame(comparison_data['reranking'])
            st.dataframe(df, width='stretch')

            # Add accuracy chart for reranking models
            reranker_accuracy = self.backend.get_reranker_accuracy()
            if reranker_accuracy:
                st.subheader("🎯 Reranking Model Accuracy Comparison")
                df_acc = pd.DataFrame(reranker_accuracy)

                # Horizontal bar chart for better comparison
                fig = px.bar(df_acc, y='model', x='accuracy',
                           title='Reranking Model Accuracy Comparison',
                           labels={'accuracy': 'Accuracy Score', 'model': 'Reranking Model'},
                           color='accuracy',
                           color_continuous_scale='viridis',
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, width='stretch')

                # Add radar chart for multi-metric comparison if available
                self._add_radar_chart(df_acc, 'Reranking Models')
        else:
            st.info("No reranking model data available")

    def _display_query_enhancement_tab(self, comparison_data):
        """Display query enhancement comparison."""
        if comparison_data.get('query_enhancement'):
            df = pd.DataFrame(comparison_data['query_enhancement'])
            st.dataframe(df, width='stretch')

            # Add accuracy chart for query enhancement comparison
            qe_accuracy = self.backend.get_query_enhancement_accuracy()
            if qe_accuracy:
                st.subheader("🎯 Query Enhancement Accuracy Comparison")
                df_acc = pd.DataFrame(qe_accuracy)

                # Horizontal bar chart for better comparison
                fig = px.bar(df_acc, y='model', x='accuracy',
                           title='Query Enhancement Accuracy Comparison',
                           labels={'accuracy': 'Accuracy Score', 'model': 'Query Enhancement'},
                           color='accuracy',
                           color_continuous_scale='viridis',
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, width='stretch')

                # Add radar chart for multi-metric comparison if available
                self._add_radar_chart(df_acc, 'Query Enhancement')
        else:
            st.info("No query enhancement data available")

    def _display_token_usage_tab(self):
        """Display token usage overview in model comparison tab."""
        from .token_usage import TokenUsageComponent
        token_comp = TokenUsageComponent(self.backend)
        token_comp.display_overview()

    def _display_ground_truth_tab(self):
        """Display ground truth import and evaluation."""
        from .ground_truth import GroundTruthComponent
        gt_comp = GroundTruthComponent(self.backend)
        gt_comp.display()
