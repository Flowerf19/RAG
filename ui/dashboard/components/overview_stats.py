import streamlit as st
from evaluation.backend_dashboard.api import BackendDashboard

class OverviewStatsComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        st.header("Tổng quan hệ thống")
        stats = self.backend.get_overview_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Số truy vấn", stats.get('total_queries', 0))
        col2.metric("Số ground-truth", stats.get('model_count', 0))
        col3.metric("Tỷ lệ lỗi (%)", stats.get('error_rate', 0))
        st.write("---")
        st.write(f"Trung bình: Accuracy={stats.get('avg_accuracy', 0):.3f}, Faithfulness={stats.get('avg_faithfulness', 0):.3f}, Relevance={stats.get('avg_relevance', 0):.3f}, Recall={stats.get('avg_recall', 0):.3f}, Latency={stats.get('avg_latency', 0):.3f}s")
