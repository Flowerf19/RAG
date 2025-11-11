"""
RAG Evaluation Dashboard
Streamlit app for visualizing and comparing RAG model performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation.backend_dashboard.api import BackendDashboard


class RAGEvaluationDashboard:
    """Streamlit dashboard for RAG model evaluation and comparison."""

    def __init__(self):
        """Initialize dashboard with backend API."""
        self.backend = BackendDashboard()
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",
            page_icon="📊",
            layout="wide"
        )

    def run(self):
        """Run the dashboard application."""
        st.title("🚀 RAG Evaluation Dashboard")
        st.markdown("Compare performance and accuracy across different RAG models")

        # Add refresh button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🔄 Refresh Data"):
                self.backend.refresh_data()
                st.rerun()

        with col2:
            # model_filter = st.selectbox(
            #     "Filter by Model",
            #     ["All Models", "GPT-4", "GPT-3.5", "Claude", "Mistral", "DeepSeek", "Local Models"],
            #     key="model_filter"
            # )
            st.empty()  # Placeholder for future model filter

        # Overview Statistics
        self._display_overview_stats()

        # Model Comparison Tables
        self._display_model_comparison()

        # Performance Charts
        self._display_performance_charts()

        # Recent Activity
        self._display_recent_activity()

    def _display_overview_stats(self):
        """Display overview statistics in header."""
        st.header("📈 Overview Statistics")

        stats = self.backend.get_overview_stats()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Queries", f"{stats['total_queries']:,}")

        with col2:
            st.metric("Avg Accuracy", f"{stats['avg_accuracy']:.3f}")

        with col3:
            st.metric("Avg Latency", f"{stats['avg_latency']:.3f}s")

        with col4:
            st.metric("Error Rate", f"{stats['error_rate']:.1f}%")

        with col5:
            st.metric("Models Tested", stats['model_count'])

    def _display_model_comparison(self):
        """Display comparison tables for different model types."""
        st.header("🔍 Model Comparison")

        comparison_data = self.backend.get_model_comparison_data()

        tab1, tab2, tab3, tab4 = st.tabs(["🤖 LLM Models", "🔗 Embedding Models", "📊 Reranking Models", "✨ Query Enhancement"])

        with tab1:
            if comparison_data['llm']:
                df = pd.DataFrame(comparison_data['llm'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for LLM models
                llm_accuracy = self.backend.get_llm_accuracy()
                if llm_accuracy:
                    st.subheader("🎯 LLM Model Accuracy")
                    df_acc = pd.DataFrame(llm_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='LLM Model Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'LLM Model'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No LLM model data available")

        with tab2:
            if comparison_data['embedding']:
                df = pd.DataFrame(comparison_data['embedding'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for embedding models
                embedder_accuracy = self.backend.get_embedder_accuracy()
                if embedder_accuracy:
                    st.subheader("🎯 Embedding Model Accuracy")
                    df_acc = pd.DataFrame(embedder_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='Embedding Model Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'Embedding Model'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No embedding model data available")

        with tab3:
            if comparison_data['reranking']:
                df = pd.DataFrame(comparison_data['reranking'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for reranking models
                reranker_accuracy = self.backend.get_reranker_accuracy()
                if reranker_accuracy:
                    st.subheader("🎯 Reranking Model Accuracy")
                    df_acc = pd.DataFrame(reranker_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='Reranking Model Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'Reranking Model'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reranking model data available")

        with tab4:
            if comparison_data.get('query_enhancement'):
                df = pd.DataFrame(comparison_data['query_enhancement'])
                st.dataframe(df, width='stretch')

                # Add accuracy chart for query enhancement comparison
                qe_accuracy = self.backend.get_query_enhancement_accuracy()
                if qe_accuracy:
                    st.subheader("🎯 Query Enhancement Accuracy")
                    df_acc = pd.DataFrame(qe_accuracy)
                    fig = px.bar(df_acc, x='model', y='accuracy',
                               title='Query Enhancement Accuracy Comparison',
                               labels={'accuracy': 'Accuracy Score', 'model': 'Query Enhancement'},
                               color='accuracy',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query enhancement data available")

    def _display_performance_charts(self):
        """Display performance visualization charts."""
        st.header("📊 Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Latency Over Time")
            latency_data = self.backend.get_latency_over_time()
            if latency_data:
                df = pd.DataFrame(latency_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                fig = px.line(df, x='timestamp', y='avg_latency',
                            title='Average Latency Trend',
                            labels={'avg_latency': 'Latency (s)', 'timestamp': 'Time'})
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No latency data available")

        with col2:
            st.info("Model-specific accuracy charts are now available in the Model Comparison tabs above.")

    def _display_recent_activity(self):
        """Display recent evaluation activity."""
        st.header("📋 Recent Activity")

        recent_metrics = self.backend.get_recent_metrics(limit=20)

        if recent_metrics:
            # Convert to DataFrame for display
            df = pd.DataFrame(recent_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Create detailed model description
            def format_model_info(row):
                llm = row.get('llm_model', 'Unknown')
                embedder = row.get('embedder_model', 'Unknown')
                reranker = row.get('reranker_model', 'none')
                qe = "QE" if row.get('query_enhanced', False) else "NoQE"
                return f"{llm} + {embedder} + {reranker} + {qe}"

            df['model_config'] = df.apply(format_model_info, axis=1)

            df['model_config'] = df.apply(format_model_info, axis=1)

            df['model_config'] = df.apply(format_model_info, axis=1)

            # Select and reorder columns
            display_cols = ['timestamp', 'model_config', 'query', 'latency', 'faithfulness', 'relevance', 'error']
            df_display = df[display_cols].head(10)

            st.dataframe(df_display, width='stretch')

            # Show summary stats
            total_recent = len(recent_metrics)
            error_recent = sum(1 for m in recent_metrics if m.get('error', False))
            avg_latency_recent = sum(m.get('latency', 0) for m in recent_metrics) / total_recent if total_recent > 0 else 0

            st.caption(f"Showing last 10 of {total_recent} recent evaluations | "
                      f"Avg Latency: {avg_latency_recent:.3f}s | Error Rate: {error_recent/total_recent*100:.1f}%")
        else:
            st.info("No recent evaluation data available")


def main():
    """Main entry point for the dashboard."""
    dashboard = RAGEvaluationDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()