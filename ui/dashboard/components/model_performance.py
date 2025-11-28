import streamlit as st
import pandas as pd
from evaluation.backend_dashboard.api import BackendDashboard
from evaluation.visualizations.visualizer import RAGMetricsVisualizer


class ModelPerformanceComponent:
    def __init__(self, backend: BackendDashboard):
        self.backend = backend

    def display(self):
        # Overview stats section
        st.header("Tổng quan hệ thống")
        stats = self.backend.get_overview_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Số truy vấn", stats.get('total_queries', 0))
        col2.metric("Số ground-truth", stats.get('model_count', 0))
        col3.metric("Tỷ lệ lỗi (%)", stats.get('error_rate', 0))
        st.write("---")
        st.write(f"Trung bình: Accuracy={stats.get('avg_accuracy', 0):.3f}, Faithfulness={stats.get('avg_faithfulness', 0):.3f}, Relevance={stats.get('avg_relevance', 0):.3f}, Recall={stats.get('avg_recall', 0):.3f}, Latency={stats.get('avg_latency', 0):.3f}s")

        st.header("Hiệu năng theo model")

        # Giải thích về tên model
        with st.expander("ℹ️ Hiểu về tên Model", expanded=False):
            st.markdown("""
            **Tên model trong bảng được tạo theo quy tắc thống nhất:**

            **🔹 Định dạng thống nhất: `{embedder}_{reranker}_{llm}`**
            - Ví dụ: `huggingface_local_none_gemini` = embedder huggingface_local + reranker none + llm gemini
            - Ý nghĩa: Tất cả các loại đánh giá (semantic similarity, recall, relevance, faithfulness) từ cùng một cấu hình sẽ được gộp lại

            **🔹 Các thành phần:**
            - **Embedder**: `huggingface_local` (BGE-M3, 1024d), `ollama` (768d), etc.
            - **Reranker**: `none` (không rerank), `bge_m3_hf_local`, `jina_v2_multilingual`, etc.
            - **LLM**: `gemini`, `ollama`, `lmstudio`, `openai` (được dùng cho faithfulness evaluation)

            **💡 Lưu ý:** Bảng này hiện gộp tất cả metrics từ cùng một cấu hình model để dễ so sánh!
            """)

        stats = self.backend.get_model_comparison_data()
        # Combine all models from the unified stats (now all in 'llm' list)
        df = pd.DataFrame(stats['llm'])

        if df.empty:
            st.info("Chưa có dữ liệu model.")
            return

        # Ensure numeric columns exist and fillna
        for col in ['accuracy', 'recall', 'latency', 'error_rate', 'faithfulness', 'relevance']:
            if col not in df.columns:
                df[col] = pd.NA

        # Show table preview
        st.dataframe(df[['model', 'accuracy', 'faithfulness', 'relevance', 'recall', 'latency', 'error_rate']])

        # Use new visualizations module to display metrics comparisons
        st.header("Visual Comparison of Model Configurations")

        try:
            # Prepare DataFrame expected by visualizer
            viz_df = pd.DataFrame({
                'Configuration': df['model'],
                'Faithfulness': df.get('faithfulness', pd.Series([0]*len(df))).astype(float),
                'Context_Recall': df.get('recall', pd.Series([0]*len(df))).astype(float),
                'Answer_Correctness': df.get('relevance', pd.Series([0]*len(df))).astype(float),
                'Answer_Relevancy': df.get('answer_relevancy', df.get('relevance', pd.Series([0]*len(df)))).astype(float)
            })

            visualizer = RAGMetricsVisualizer(output_dir="data/visualizations")
            viz_results = visualizer.generate_all_charts(viz_df, title_prefix="Model Performance Comparison", save_charts=True, show_charts=False)

            if "error" in viz_results:
                st.warning(f"Could not generate visualizations: {viz_results['error']}")
            else:
                # Show generated charts inline
                st.subheader("Charts")
                chart_files = {k: v for k, v in viz_results.items() if k != 'table' and k != 'error'}
                if chart_files:
                    cols = st.columns(2)
                    for i, (chart_type, chart_path) in enumerate(chart_files.items()):
                        with cols[i % 2]:
                            st.subheader(chart_type.replace('_', ' ').title())
                            try:
                                st.image(chart_path, width='stretch')
                            except Exception as e:
                                st.error(f"Could not load chart {chart_type}: {e}")
                else:
                    st.info("No charts generated to display.")

        except Exception as e:
            st.error(f"Visualization failed: {e}")
