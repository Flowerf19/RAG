import streamlit as st
import pandas as pd
import altair as alt
from evaluation.backend_dashboard.api import BackendDashboard


class ModelPerformanceComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Hiệu năng theo model")
        stats = self.backend.get_model_comparison_data()
        df = pd.DataFrame(stats['embedding'] + stats['llm'] + stats['reranking'])

        if df.empty:
            st.info("Chưa có dữ liệu model.")
            return

        # Ensure numeric columns exist and fillna
        for col in ['accuracy', 'recall', 'latency', 'error_rate', 'faithfulness', 'relevance']:
            if col not in df.columns:
                df[col] = pd.NA

        # Show table preview
        st.dataframe(df[['model', 'accuracy', 'faithfulness', 'relevance', 'recall', 'latency', 'error_rate']])

        # Scatter: Accuracy vs Recall — size by latency, color by error_rate
        scatter_df = df[['model', 'accuracy', 'recall', 'latency', 'error_rate']].fillna(0)
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle()
            .encode(
                x=alt.X('accuracy:Q', title='Accuracy'),
                y=alt.Y('recall:Q', title='Recall'),
                size=alt.Size('latency:Q', title='Latency', scale=alt.Scale(range=[30, 400])),
                color=alt.Color('error_rate:Q', title='Error Rate', scale=alt.Scale(scheme='redblue')),
                tooltip=['model', 'accuracy', 'recall', 'latency', 'error_rate']
            )
            .interactive()
            .properties(height=420)
        )

        st.altair_chart(scatter, use_container_width=True)
