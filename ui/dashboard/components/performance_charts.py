"""
Performance Charts Component
Displays performance visualization charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from evaluation.backend_dashboard.api import BackendDashboard


class PerformanceChartsComponent:
    """Component for displaying performance visualization charts."""

    def __init__(self, backend: BackendDashboard):
        """Initialize with backend API."""
        self.backend = backend

    def display(self):
        """Display performance visualization charts."""
        st.header("📊 Performance Analysis")

        # Get recent metrics data for charts
        recent_metrics = self.backend.get_recent_metrics(limit=200)

        if recent_metrics:
            df = pd.DataFrame(recent_metrics)

            # Row 1: Correlation only
            col1 = st.columns(1)[0]

            with col1:
                st.subheader("🔗 Metric Correlations")
                metrics_cols = ['faithfulness', 'relevance', 'recall', 'latency']
                available_metrics = [col for col in metrics_cols if col in df.columns]

                if len(available_metrics) >= 2:
                    corr_matrix = df[available_metrics].corr()

                    fig = px.imshow(corr_matrix,
                                  text_auto=True,
                                  aspect="auto",
                                  title="Performance Metrics Correlation Matrix",
                                  color_continuous_scale='RdBu_r',
                                  labels=dict(color="Correlation"))
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Need at least 2 metrics for correlation analysis")

            # Row 2: Distribution analysis
            st.subheader("📈 Score Distributions")
            dist_col1, dist_col2, dist_col3 = st.columns(3)

            with dist_col1:
                if 'faithfulness' in df.columns:
                    fig = px.histogram(df, x='faithfulness',
                                     title='Faithfulness Distribution',
                                     labels={'faithfulness': 'Faithfulness Score'},
                                     nbins=20,
                                     color_discrete_sequence=['#636EFA'])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No faithfulness data")

            with dist_col2:
                if 'relevance' in df.columns:
                    fig = px.histogram(df, x='relevance',
                                     title='Relevance Distribution',
                                     labels={'relevance': 'Relevance Score'},
                                     nbins=20,
                                     color_discrete_sequence=['#EF553B'])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No relevance data")

            with dist_col3:
                if 'recall' in df.columns:
                    fig = px.histogram(df, x='recall',
                                     title='Recall Distribution',
                                     labels={'recall': 'Recall Score'},
                                     nbins=20,
                                     color_discrete_sequence=['#00CC96'])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No recall data")

            # Row 3: Box plots and scatter plots
            st.subheader("📊 Advanced Analytics")
            adv_col1, adv_col2 = st.columns(2)

            with adv_col1:
                st.subheader("📦 Metric Box Plots")
                # Prepare data for box plot
                box_data = []
                for metric in ['faithfulness', 'relevance', 'recall']:
                    if metric in df.columns:
                        metric_df = df[[metric]].copy()
                        metric_df['metric'] = metric.capitalize()
                        metric_df = metric_df.rename(columns={metric: 'value'})
                        box_data.append(metric_df)

                if box_data:
                    box_df = pd.concat(box_data, ignore_index=True)
                    fig = px.box(box_df, x='metric', y='value',
                               title='Metric Distributions (Box Plot)',
                               labels={'value': 'Score', 'metric': 'Metric'},
                               color='metric')
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No metric data for box plot")

            with adv_col2:
                st.subheader("🎯 Performance vs Latency")
                if 'latency' in df.columns and 'faithfulness' in df.columns:
                    # Calculate average performance score
                    perf_cols = ['faithfulness', 'relevance', 'recall']
                    available_perf = [col for col in perf_cols if col in df.columns]

                    if available_perf:
                        df_plot = df.copy()
                        df_plot['avg_performance'] = df_plot[available_perf].mean(axis=1)

                        # Filter out NaN values for plotting
                        df_plot = df_plot.dropna(subset=['latency', 'avg_performance'])

                        if not df_plot.empty:
                            fig = px.scatter(df_plot, x='latency', y='avg_performance',
                                           title='Performance vs Latency Trade-off',
                                           labels={'latency': 'Latency (s)', 'avg_performance': 'Avg Performance Score'},
                                           color='avg_performance',
                                           color_continuous_scale='viridis',
                                           size='avg_performance')
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.info("No valid data for performance vs latency plot")
                    else:
                        st.info("No performance metrics available")
                else:
                    st.info("Need both latency and performance data")

        else:
            st.info("No performance data available")
