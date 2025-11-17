import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard

class ModelPerformanceComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Hiệu năng theo model")
        stats = self.backend.get_model_comparison_data()
        df = pd.DataFrame(stats['embedding'] + stats['llm'] + stats['reranking'])
        if not df.empty:
            st.dataframe(df[['model','accuracy','faithfulness','relevance','recall','latency','error_rate']])
            st.bar_chart(df.set_index('model')[['accuracy','recall']])
        else:
            st.info("Chưa có dữ liệu model.")
